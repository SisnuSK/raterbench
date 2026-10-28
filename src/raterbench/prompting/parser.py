"""Strict JSON output parsing with schema validation."""

import json
import math
import re
from dataclasses import dataclass

from raterbench.llm.types import TokenLogprob


@dataclass
class ParseResult:
    ok: bool
    label: str | None = None
    confidence: float | None = None
    reasoning: str | None = None
    error: str | None = None


def _json_objects(text: str):
    """Yield every balanced top-level {...} substring, respecting JSON string escapes."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def normalize_label(s: str) -> str:
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def parse_output(text: str, labels: tuple[str, ...], want_reasoning: bool = False) -> ParseResult:
    obj = None
    for candidate in _json_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return ParseResult(ok=False, error="no JSON object found")

    raw_label = obj.get("label")
    if not isinstance(raw_label, str):
        return ParseResult(ok=False, error="missing or non-string 'label'")
    by_norm = {normalize_label(lab): lab for lab in labels}
    label = by_norm.get(normalize_label(raw_label))
    if label is None:
        return ParseResult(ok=False, error=f"label {raw_label!r} not in {list(labels)}")

    conf = obj.get("confidence")
    if isinstance(conf, str):
        try:
            conf = float(conf.strip().rstrip("%"))
        except ValueError:
            conf = None
    if isinstance(conf, bool) or not isinstance(conf, int | float) or not 0 <= conf <= 100:
        return ParseResult(ok=False, label=label, error="missing or invalid 'confidence' (0-100)")

    reasoning = obj.get("reasoning")
    if want_reasoning and not isinstance(reasoning, str):
        return ParseResult(ok=False, label=label, error="missing 'reasoning'")
    return ParseResult(
        ok=True,
        label=label,
        confidence=float(conf),
        reasoning=reasoning if isinstance(reasoning, str) else None,
    )


_LABEL_VALUE = re.compile(r'"label"\s*:\s*"([^"]*)"')


def label_logprob(text: str, logprobs: list[TokenLogprob] | None) -> float | None:
    """Sum of log-probabilities of the tokens spelling the label value in the output.

    Returns None if logprobs are unavailable, if the tokens do not reconstruct the text
    exactly, or if no label field is found.
    """
    if not logprobs:
        return None
    if "".join(t.token for t in logprobs) != text:
        return None
    m = _LABEL_VALUE.search(text)
    if m is None:
        return None
    start, end = m.span(1)
    total, pos, hit = 0.0, 0, False
    for t in logprobs:
        t_start, pos = pos, pos + len(t.token)
        if pos > start and t_start < end:
            total += t.logprob
            hit = True
    return total if hit and math.isfinite(total) else None
