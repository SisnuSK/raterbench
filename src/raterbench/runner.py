"""Benchmark runner: (dataset x model x condition x item x sample) -> predictions + manifest."""

import asyncio
import json
import logging
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from raterbench import __version__
from raterbench.config import MainConfig, ModelConfig, RunConfig, load_yaml
from raterbench.data import registry, splits
from raterbench.data.sampling import stratified_sample
from raterbench.data.schema import Item
from raterbench.llm.backends import Backend
from raterbench.llm.cache import ResponseCache
from raterbench.llm.client import LLMClient
from raterbench.llm.cost import BudgetExceeded, CostTracker
from raterbench.llm.types import BackendError, LLMRequest, Message
from raterbench.prompting.parser import label_logprob, parse_output
from raterbench.prompting.render import CONDITIONS, REASONING_CONDITIONS, PromptBuilder

log = logging.getLogger("raterbench.run")


@dataclass(frozen=True)
class Job:
    dataset: str
    role: str
    item: Item
    condition: str
    sample_idx: int  # 0 = main run; 1..n = self-consistency samples


class _JsonFormatter(logging.Formatter):
    _std = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        out.update({k: v for k, v in record.__dict__.items() if k not in self._std})
        return json.dumps(out, default=str)


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return r.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        backends: dict[str, Backend] | None = None,
        sleep: Any = asyncio.sleep,
    ):
        unknown = set(cfg.conditions) - set(CONDITIONS)
        if unknown:
            raise ValueError(f"Unknown conditions {sorted(unknown)}")
        self.cfg = cfg
        self.main = load_yaml(cfg.main, MainConfig)
        self.model_cfgs = [load_yaml(p, ModelConfig) for p in cfg.models]
        self._backends = backends or {}
        self._sleep = sleep
        self.items: dict[str, dict[str, list[Item]]] = {}
        self.builders: dict[str, PromptBuilder] = {}
        self.labels: dict[str, tuple[str, ...]] = {}
        for ds in self.main.dataset_configs():
            info = registry.get(ds.name)
            roles = splits.read_samples(
                info.load(self.main.raw_dir), self.main.samples_dir / f"{ds.name}.jsonl"
            )
            selected = {}
            for role in cfg.roles:
                pool = roles[role]
                if cfg.max_items_per_role is not None:
                    pool = stratified_sample(
                        pool, cfg.max_items_per_role, lambda it: (it.subgroup, it.label), cfg.seed
                    )
                selected[role] = sorted(pool, key=lambda it: it.id)
            self.items[ds.name] = selected
            self.labels[ds.name] = info.labels
            # Few-shot examples come only from the train role.
            self.builders[ds.name] = PromptBuilder(
                ds.name, cfg.prompts_dir, roles["train"], cfg.fewshot_k, cfg.seed
            )

    # ---- job construction -------------------------------------------------------------

    def jobs(self) -> list[Job]:
        out = []
        sc = self.cfg.self_consistency
        for ds, roles in self.items.items():
            for role, items in roles.items():
                for cond in self.cfg.conditions:
                    n_samples = 1 + (sc.n if cond in sc.conditions else 0)
                    for it in items:
                        out.extend(Job(ds, role, it, cond, s) for s in range(n_samples))
        return out

    def request_for(self, job: Job, client: LLMClient) -> LLMRequest:
        builder = self.builders[job.dataset]
        if job.sample_idx == 0:
            temperature = 0.0 if client.supports_temperature else None
        else:
            temperature = (
                self.cfg.self_consistency.temperature if client.supports_temperature else None
            )
        return LLMRequest(
            system=builder.system,
            messages=[Message(role="user", content=builder.render(job.condition, job.item))],
            temperature=temperature,
            seed=self.cfg.seed + job.sample_idx,
            sample_idx=job.sample_idx,
            want_logprobs=client.supports_logprobs,
        )

    def _client(self, mc: ModelConfig, cache: ResponseCache, tracker: CostTracker) -> LLMClient:
        return LLMClient(
            name=mc.name,
            backend_name=mc.backend,
            model=mc.model,
            params=mc.params,
            pricing=mc.pricing,
            concurrency=mc.concurrency,
            cache=cache,
            tracker=tracker,
            backend=self._backends.get(mc.name),
            sleep=self._sleep,
        )

    # ---- execution --------------------------------------------------------------------

    async def _run_job(self, job: Job, mc: ModelConfig, client: LLMClient) -> dict[str, Any]:
        builder = self.builders[job.dataset]
        labels = self.labels[job.dataset]
        want_reasoning = job.condition in REASONING_CONDITIONS
        req = self.request_for(job, client)
        rec: dict[str, Any] = {
            "dataset": job.dataset,
            "role": job.role,
            "item_id": job.item.id,
            "subgroup": job.item.subgroup,
            "gold": job.item.label,
            "model": mc.name,
            "condition": job.condition,
            "sample_idx": job.sample_idx,
            "temperature": req.temperature,
            "prompt_hash": builder.prompt_hash(job.condition),
            "retried": False,
            "cached": True,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        try:
            resp, cached, cost = await client.complete(req)
            rec["cached"] &= cached
            rec["cost_usd"] += cost
            rec["input_tokens"] += resp.input_tokens
            rec["output_tokens"] += resp.output_tokens
            parsed = parse_output(resp.text, labels, want_reasoning)
            if not parsed.ok and resp.stop_reason != "refusal":
                retry_req = req.model_copy(
                    update={
                        "messages": [
                            *req.messages,
                            Message(role="assistant", content=resp.text or "(no output)"),
                            Message(role="user", content=builder.retry_message(parsed.error or "")),
                        ]
                    }
                )
                resp, cached, cost = await client.complete(retry_req)
                rec["retried"] = True
                rec["cached"] &= cached
                rec["cost_usd"] += cost
                rec["input_tokens"] += resp.input_tokens
                rec["output_tokens"] += resp.output_tokens
                parsed = parse_output(resp.text, labels, want_reasoning)
        except BudgetExceeded:
            rec.update(outcome="skipped_budget", cached=False)
            return rec
        except BackendError as e:
            rec.update(outcome="api_error", error=str(e), cached=False)
            log.error("api error", extra={"model": mc.name, "item": job.item.id, "error": str(e)})
            return rec

        lp = label_logprob(resp.text, resp.logprobs) if parsed.ok else None
        if resp.stop_reason == "refusal":
            outcome = "refusal"
        else:
            outcome = "ok" if parsed.ok else "parse_failure"
        rec.update(
            outcome=outcome,
            label=parsed.label if parsed.ok else None,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            parse_error=parsed.error,
            label_logprob=lp,
            label_prob=math.exp(lp) if lp is not None else None,
            raw_text=resp.text,
            served_model=resp.served_model,
            stop_reason=resp.stop_reason,
        )
        return rec

    async def run(self) -> Path:
        cfg = self.cfg
        run_dir = cfg.output_dir / cfg.name
        run_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger("raterbench")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        started = datetime.now(UTC)
        cache = ResponseCache(cfg.cache_path)
        tracker = CostTracker(budget_usd=cfg.budget_usd)
        jobs = self.jobs()
        log.info("run started", extra={"run": cfg.name, "jobs": len(jobs)})
        records: list[dict[str, Any]] = []
        model_info: dict[str, Any] = {}
        try:
            # Models run one after another (local models share one GPU); calls within a
            # model run concurrently up to its configured concurrency.
            for mc in self.model_cfgs:
                client = self._client(mc, cache, tracker)
                model_info[mc.name] = {
                    **mc.model_dump(mode="json"),
                    **await client.backend.describe(),
                }
                done = 0

                async def one(job: Job, mc: ModelConfig = mc, client: LLMClient = client) -> dict:
                    nonlocal done
                    rec = await self._run_job(job, mc, client)
                    done += 1
                    if done % 50 == 0 or done == len(jobs):
                        print(
                            f"  {mc.name}: {done}/{len(jobs)}  spent ${tracker.spent_usd:.4f}",
                            flush=True,
                        )
                    return rec

                print(f"model {mc.name}: {len(jobs)} jobs", flush=True)
                records.extend(await asyncio.gather(*(one(j) for j in jobs)))
                served = Counter(r.get("served_model") for r in records if r["model"] == mc.name)
                model_info[mc.name]["served_models"] = dict(served)
        finally:
            cache.close()
            root.removeHandler(handler)
            handler.close()

        records.sort(
            key=lambda r: (r["model"], r["dataset"], r["condition"], r["item_id"], r["sample_idx"])
        )
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = summarize(records)
        (run_dir / "summary.md").write_text(format_summary(summary), encoding="utf-8")
        manifest = {
            "run": cfg.name,
            "raterbench_version": __version__,
            "started_utc": started.isoformat(),
            "finished_utc": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit() or "unavailable (git not installed or not a repository)",
            "python": sys.version,
            "platform": platform.platform(),
            "config": {
                "run": cfg.model_dump(mode="json"),
                "main": self.main.model_dump(mode="json"),
                "datasets": [d.model_dump(mode="json") for d in self.main.dataset_configs()],
            },
            "models": model_info,
            "prompt_hashes": {
                ds: {c: b.prompt_hash(c) for c in cfg.conditions} for ds, b in self.builders.items()
            },
            "fewshot_ids": {
                ds: {str(k): [it.id for it in v] for k, v in b.fewshot.items()}
                for ds, b in self.builders.items()
            },
            "items": {
                ds: {r: len(v) for r, v in roles.items()} for ds, roles in self.items.items()
            },
            "n_jobs": len(records),
            "cost": tracker.summary(),
            "summary": summary,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        return run_dir

    # ---- cost estimate ----------------------------------------------------------------

    def estimate(self) -> list[dict[str, Any]]:
        """Projected cost of uncached calls (excluding parse retries).

        Upper bound assumes every call uses the model's full max output tokens. If the cache
        holds earlier responses from the same model, their mean token counts give an
        empirical projection as well.
        """
        cache = ResponseCache(self.cfg.cache_path)
        tracker = CostTracker(budget_usd=math.inf)
        rows = []
        try:
            for mc in self.model_cfgs:
                client = self._client(mc, cache, tracker)
                uncached = 0
                upper = 0.0
                in_tok = 0
                seen_in, seen_out, seen_n = 0, 0, 0
                for job in self.jobs():
                    req = self.request_for(job, client)
                    hit = cache.get(client.key(req))
                    if hit is not None:
                        seen_in += hit.input_tokens
                        seen_out += hit.output_tokens
                        seen_n += 1
                        continue
                    uncached += 1
                    upper += client.max_call_cost(req)
                    in_tok += len(req.system + req.messages[0].content) // 3 + 1
                row = {
                    "model": mc.name,
                    "uncached_calls": uncached,
                    "cached_calls": seen_n,
                    "est_input_tokens": in_tok,
                    "upper_bound_usd": round(upper, 4),
                }
                if seen_n:
                    mean_in, mean_out = seen_in / seen_n, seen_out / seen_n
                    row["mean_tokens_in_cache"] = {
                        "input": round(mean_in),
                        "output": round(mean_out),
                    }
                    row["empirical_usd"] = round(
                        uncached * mc.pricing.cost(int(mean_in), int(mean_out)), 4
                    )
                rows.append(row)
        finally:
            cache.close()
        return rows


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        kind = "main" if r["sample_idx"] == 0 else "self_consistency"
        groups[(r["model"], r["dataset"], r["condition"], kind)].append(r)
    rows = []
    for (model, ds, cond, kind), rs in sorted(groups.items()):
        c = Counter(r["outcome"] for r in rs)
        answered = c["ok"] + c["parse_failure"] + c["refusal"]
        rows.append(
            {
                "model": model,
                "dataset": ds,
                "condition": cond,
                "kind": kind,
                "n": len(rs),
                "ok": c["ok"],
                "parse_failure": c["parse_failure"],
                "refusal": c["refusal"],
                "api_error": c["api_error"],
                "skipped_budget": c["skipped_budget"],
                "retried": sum(r["retried"] for r in rs),
                "parse_failure_rate": round((c["parse_failure"] + c["refusal"]) / answered, 4)
                if answered
                else None,
                "agree_with_gold": sum(r.get("label") == r["gold"] for r in rs),
                "with_label_logprob": sum(r.get("label_logprob") is not None for r in rs),
                "cached": sum(r["cached"] for r in rs),
                "cost_usd": round(sum(r["cost_usd"] for r in rs), 6),
            }
        )
    return rows


def format_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No jobs.\n"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def load_run_config(path: Path) -> RunConfig:
    with Path(path).open(encoding="utf-8") as f:
        return RunConfig.model_validate(yaml.safe_load(f))
