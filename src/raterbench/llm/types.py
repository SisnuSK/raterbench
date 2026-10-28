"""Backend-neutral request/response types."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class LLMRequest(BaseModel):
    system: str
    messages: list[Message]
    # Decoding: temperature None means "backend default" (Claude models accept no temperature).
    temperature: float | None = None
    seed: int | None = None
    sample_idx: int = 0  # distinguishes repeated samples of the same prompt in the cache
    want_logprobs: bool = False


class TokenLogprob(BaseModel):
    token: str
    logprob: float


class LLMResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    served_model: str  # model identifier reported by the provider
    stop_reason: str | None = None
    logprobs: list[TokenLogprob] | None = None
    latency_s: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class BackendError(Exception):
    """A failed call. `retryable` marks rate limits, overloads, 5xx, and network errors."""

    def __init__(self, message: str, retryable: bool, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


def cache_key(backend: str, model: str, params: dict[str, Any], req: LLMRequest) -> str:
    """Stable hash over everything that can change a model's output."""
    payload = {
        "backend": backend,
        "model": model,
        "params": params,
        "request": req.model_dump(mode="json"),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
