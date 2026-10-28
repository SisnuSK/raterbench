import asyncio
import math
from pathlib import Path

import pytest

from fakes import FakeBackend, no_sleep
from raterbench.llm.cache import ResponseCache
from raterbench.llm.client import LLMClient
from raterbench.llm.cost import BudgetExceeded, CostTracker, Pricing
from raterbench.llm.types import (
    BackendError,
    LLMRequest,
    LLMResponse,
    Message,
    TokenLogprob,
    cache_key,
)
from raterbench.prompting.parser import label_logprob, parse_output

LABELS = ("favor", "against", "none")


# ---- parser ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,label,conf",
    [
        ('{"label": "favor", "confidence": 80}', "favor", 80.0),
        ('```json\n{"label": "Against", "confidence": "65"}\n```', "against", 65.0),
        ('Sure! {"label": " none ", "confidence": 0} Hope this helps.', "none", 0.0),
        (
            '{"reasoning": "uses {braces} and \\"quotes\\"", "label": "favor", "confidence": 100}',
            "favor",
            100.0,
        ),
        ('{"label": "favor", "confidence": "70%"}', "favor", 70.0),
    ],
)
def test_parse_valid(text: str, label: str, conf: float) -> None:
    r = parse_output(text, LABELS)
    assert r.ok, r.error
    assert (r.label, r.confidence) == (label, conf)


def test_parse_normalizes_label_spacing() -> None:
    r = parse_output('{"label": "not stress", "confidence": 50}', ("stress", "not_stress"))
    assert r.ok and r.label == "not_stress"


@pytest.mark.parametrize(
    "text,err",
    [
        ("I think it's favor.", "no JSON"),
        ("{label: favor}", "no JSON"),
        ('{"label": "neutral", "confidence": 50}', "not in"),
        ('{"label": "favor", "confidence": 150}', "confidence"),
        ('{"label": "favor"}', "confidence"),
        ('{"label": "favor", "confidence": true}', "confidence"),
        ('{"confidence": 50}', "label"),
    ],
)
def test_parse_invalid(text: str, err: str) -> None:
    r = parse_output(text, LABELS)
    assert not r.ok
    assert err in r.error


def test_parse_requires_reasoning_when_asked() -> None:
    assert not parse_output('{"label": "favor", "confidence": 50}', LABELS, want_reasoning=True).ok
    r = parse_output('{"reasoning": "r", "label": "favor", "confidence": 50}', LABELS, True)
    assert r.ok and r.reasoning == "r"


def test_parse_skips_invalid_object_and_uses_next() -> None:
    r = parse_output('{bad} then {"label": "none", "confidence": 10}', LABELS)
    assert r.ok and r.label == "none"


def test_label_logprob_sums_label_tokens() -> None:
    toks = ['{"', "label", '":', ' "', "ag", "ainst", '",', ' "confidence": 9', "0}"]
    lps = [-0.1, -0.1, -0.1, -0.1, -0.5, -0.25, -0.1, -0.1, -0.1]
    text = "".join(toks)
    got = label_logprob(
        text, [TokenLogprob(token=t, logprob=lp) for t, lp in zip(toks, lps, strict=True)]
    )
    assert got == pytest.approx(-0.75)
    assert math.exp(got) == pytest.approx(0.4724, abs=1e-4)


def test_label_logprob_none_when_tokens_do_not_match_text() -> None:
    assert label_logprob('{"label": "x"}', [TokenLogprob(token="nope", logprob=-1)]) is None
    assert label_logprob('{"label": "x"}', None) is None


# ---- cache key / cache --------------------------------------------------------------


def _req(**kw) -> LLMRequest:
    base = {"system": "s", "messages": [Message(role="user", content="u")]}
    return LLMRequest(**{**base, **kw})


def test_cache_key_depends_on_everything_that_changes_output() -> None:
    k = cache_key("b", "m", {"p": 1}, _req())
    assert k == cache_key("b", "m", {"p": 1}, _req())
    assert k != cache_key("b", "m2", {"p": 1}, _req())
    assert k != cache_key("b", "m", {"p": 2}, _req())
    assert k != cache_key("b", "m", {"p": 1}, _req(temperature=0.7))
    assert k != cache_key("b", "m", {"p": 1}, _req(sample_idx=1))
    assert k != cache_key("b", "m", {"p": 1}, _req(system="s2"))


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    c = ResponseCache(path)
    resp = LLMResponse(
        text="t",
        input_tokens=1,
        output_tokens=2,
        served_model="m",
        logprobs=[TokenLogprob(token="t", logprob=-0.5)],
    )
    c.put("k", resp)
    c.close()
    c2 = ResponseCache(path)
    assert c2.get("k") == resp
    assert c2.get("missing") is None
    assert len(c2) == 1
    c2.close()


# ---- client: cache, retries, budget -----------------------------------------------


def _client(
    tmp_path: Path, backend: FakeBackend, budget: float = 10.0, pricing: Pricing | None = None
) -> LLMClient:
    return LLMClient(
        name="fake",
        backend_name="fake",
        model="fake-1",
        params={},
        pricing=pricing or Pricing(input_per_mtok=1.0, output_per_mtok=10.0),
        concurrency=2,
        cache=ResponseCache(tmp_path / "c.sqlite"),
        tracker=CostTracker(budget_usd=budget),
        backend=backend,
        sleep=no_sleep,
    )


def test_client_caches_and_charges_once(tmp_path: Path) -> None:
    backend = FakeBackend()
    client = _client(tmp_path, backend)
    r1, cached1, cost1 = asyncio.run(client.complete(_req()))
    r2, cached2, cost2 = asyncio.run(client.complete(_req()))
    assert (cached1, cached2) == (False, True)
    assert cost1 == pytest.approx((1000 * 1.0 + 100 * 10.0) / 1e6)
    assert cost2 == 0.0
    assert len(backend.calls) == 1
    assert client.tracker.spent_usd == pytest.approx(cost1)
    assert client.tracker.reserved_usd == pytest.approx(0.0)


def test_client_retries_retryable_errors(tmp_path: Path) -> None:
    backend = FakeBackend(errors=[BackendError("429", True), BackendError("503", True)])
    resp, _, _ = asyncio.run(_client(tmp_path, backend).complete(_req()))
    assert resp.text and len(backend.calls) == 3


def test_client_does_not_retry_permanent_errors(tmp_path: Path) -> None:
    backend = FakeBackend(errors=[BackendError("400 bad request", False)])
    client = _client(tmp_path, backend)
    with pytest.raises(BackendError):
        asyncio.run(client.complete(_req()))
    assert len(backend.calls) == 1
    assert client.tracker.reserved_usd == pytest.approx(0.0)


def test_client_gives_up_after_max_attempts(tmp_path: Path) -> None:
    backend = FakeBackend(errors=[BackendError("503", True)] * 10)
    with pytest.raises(BackendError):
        asyncio.run(_client(tmp_path, backend).complete(_req()))
    assert len(backend.calls) == 6


def test_budget_cap_refuses_before_calling_and_stays_halted(tmp_path: Path) -> None:
    # Worst case per call: (2 tokens in + 100 out) at $10k/Mtok output = ~$1.00
    pricing = Pricing(input_per_mtok=0.0, output_per_mtok=10_000.0)
    backend = FakeBackend()
    client = _client(tmp_path, backend, budget=1.5, pricing=pricing)
    asyncio.run(client.complete(_req()))  # actual cost: 100 tokens -> $1.00
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.complete(_req(sample_idx=1)))
    assert len(backend.calls) == 1
    assert client.tracker.halted
    assert client.tracker.spent_usd <= 1.5
    # Cached calls still work after the halt; new ones do not.
    _, cached, _ = asyncio.run(client.complete(_req()))
    assert cached
    with pytest.raises(BudgetExceeded):
        asyncio.run(client.complete(_req(sample_idx=2)))
