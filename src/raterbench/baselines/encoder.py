"""Fine-tuned encoder baseline (e.g. RoBERTa-base) trained on the train role.

Requires the `train` extra (torch, transformers). Model selection uses a dev set
(the published validation split, or a grouped hold-out from train when none exists);
the calibration and eval roles are never seen during training.
"""

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from raterbench.data.schema import Item
from raterbench.metrics.agreement import agreement_metrics


@dataclass
class EncoderSettings:
    model_name: str = "FacebookAI/roberta-base"
    max_length: int = 256
    lr: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 64
    max_epochs: int = 6
    patience: int = 2  # epochs without dev macro-F1 improvement before stopping
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    bf16: bool = True


@dataclass
class TrainResult:
    predictions: dict[str, list[dict[str, Any]]]  # role -> per-item records
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    seconds: float = 0.0
    model_revision: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _encode(tok: PreTrainedTokenizerBase, items: list[Item], max_length: int) -> list[torch.Tensor]:
    enc = tok([it.text for it in items], truncation=True, max_length=max_length)
    return [torch.tensor(ids) for ids in enc["input_ids"]]


def _batches(
    ids: list[torch.Tensor], pad_id: int, batch_size: int, order: list[int]
) -> list[tuple[list[int], torch.Tensor, torch.Tensor]]:
    out = []
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        x = pad_sequence([ids[j] for j in idx], batch_first=True, padding_value=pad_id)
        out.append((idx, x, (x != pad_id).long()))
    return out


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    ids: list[torch.Tensor],
    pad_id: int,
    s: EncoderSettings,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    # Sort by length so batches pad little; restore order afterwards.
    order = sorted(range(len(ids)), key=lambda j: len(ids[j]))
    probs = np.zeros((len(ids), model.config.num_labels), dtype=np.float64)
    use_bf16 = s.bf16 and device.type == "cuda"
    for idx, x, mask in _batches(ids, pad_id, s.eval_batch_size, order):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
        probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return probs


def train_encoder(
    train: list[Item],
    dev: list[Item],
    predict_sets: dict[str, list[Item]],
    labels: tuple[str, ...],
    s: EncoderSettings,
    seed: int,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase | None = None,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    set_seed(seed)
    label2id = {lab: i for i, lab in enumerate(labels)}
    tok = tokenizer or AutoTokenizer.from_pretrained(s.model_name)
    if model_factory is not None:
        model = model_factory()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            s.model_name,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id=label2id,
        )
    model.to(device)
    pad_id = tok.pad_token_id

    train_ids = _encode(tok, train, s.max_length)
    y_train = torch.tensor([label2id[it.label] for it in train])
    dev_ids = _encode(tok, dev, s.max_length)

    steps_per_epoch = math.ceil(len(train) / s.batch_size)
    total_steps = steps_per_epoch * s.max_epochs
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": s.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(params, lr=s.lr)
    sched = get_linear_schedule_with_warmup(opt, int(s.warmup_ratio * total_steps), total_steps)
    use_bf16 = s.bf16 and device.type == "cuda"
    rng = random.Random(seed)

    t0 = time.perf_counter()
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, s.max_epochs + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss = 0.0
        for idx, x, mask in _batches(train_ids, pad_id, s.batch_size, order):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=x.to(device), attention_mask=mask.to(device)).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y_train[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * len(idx)
        dev_pred = predict_probs(model, dev_ids, pad_id, s, device).argmax(axis=1)
        dev_f1 = agreement_metrics([it.label for it in dev], [labels[i] for i in dev_pred], labels)[
            "macro_f1"
        ]
        history.append(
            {"epoch": epoch, "train_loss": total_loss / len(train), "dev_macro_f1": dev_f1}
        )
        log(
            f"    epoch {epoch}: train_loss={total_loss / len(train):.4f} dev_macro_f1={dev_f1:.4f}"
        )
        if dev_f1 > best_f1:
            best_f1, best_epoch, stale = dev_f1, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= s.patience:
                break

    model.load_state_dict(best_state)
    predictions = {}
    for role, items in {**predict_sets, "dev": dev}.items():
        probs = predict_probs(model, _encode(tok, items, s.max_length), pad_id, s, device)
        predictions[role] = [
            {
                "item_id": it.id,
                "label": labels[int(p.argmax())],
                "confidence": float(100 * p.max()),
                "probs": {lab: float(p[i]) for i, lab in enumerate(labels)},
            }
            for it, p in zip(items, probs, strict=True)
        ]
    revision = getattr(model.config, "_commit_hash", None)
    del model, best_state, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return TrainResult(
        predictions=predictions,
        history=history,
        best_epoch=best_epoch,
        seconds=time.perf_counter() - t0,
        model_revision=revision,
    )
