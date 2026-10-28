"""Encoder training on a tiny, randomly initialized model (no downloads; CPU)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from raterbench.baselines.encoder import EncoderSettings, train_encoder  # noqa: E402
from raterbench.data.schema import Item  # noqa: E402

LABELS = ("pos", "neg")
WORDS = ["good", "great", "fine", "bad", "awful", "poor", "the", "movie", "was", "very"]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]"
    )


def _model():
    cfg = transformers.BertConfig(
        vocab_size=len(WORDS) + 2,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
        pad_token_id=0,
    )
    return transformers.BertForSequenceClassification(cfg)


def _items(n: int, split: str) -> list[Item]:
    out = []
    for i in range(n):
        lab = LABELS[i % 2]
        word = ["good", "great", "fine"][i % 3] if lab == "pos" else ["bad", "awful", "poor"][i % 3]
        out.append(
            Item(
                id=f"{split}{i}",
                text=f"the movie was very {word}",
                label=lab,
                subgroup="s",
                split=split,
            )
        )
    return out


def test_train_encoder_learns_toy_task_and_predicts_all_roles() -> None:
    s = EncoderSettings(max_length=16, lr=5e-3, batch_size=8, max_epochs=8, patience=8, bf16=False)
    res = train_encoder(
        _items(64, "train"),
        _items(16, "dev"),
        {"eval": _items(20, "test")},
        LABELS,
        s,
        seed=0,
        device=torch.device("cpu"),
        tokenizer=_tokenizer(),
        model_factory=_model,
        log=lambda _: None,
    )
    assert set(res.predictions) == {"eval", "dev"}
    ev = res.predictions["eval"]
    assert len(ev) == 20
    assert all(abs(sum(r["probs"].values()) - 1) < 1e-5 for r in ev)
    acc = sum(r["label"] == it.label for r, it in zip(ev, _items(20, "test"), strict=True)) / 20
    assert acc >= 0.9
    assert 1 <= res.best_epoch <= len(res.history)


def test_train_encoder_is_deterministic_on_cpu() -> None:
    s = EncoderSettings(max_length=16, lr=5e-3, batch_size=8, max_epochs=2, patience=2, bf16=False)

    def run():
        return train_encoder(
            _items(32, "train"),
            _items(8, "dev"),
            {"eval": _items(8, "test")},
            LABELS,
            s,
            seed=3,
            device=torch.device("cpu"),
            tokenizer=_tokenizer(),
            model_factory=_model,
            log=lambda _: None,
        ).predictions["eval"]

    assert [r["probs"] for r in run()] == [r["probs"] for r in run()]
