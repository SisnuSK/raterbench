"""LLM backends. Each turns an LLMRequest into an LLMResponse or raises BackendError.

Implemented: `anthropic` (Claude API) and `ollama` (local models). `openai`, `vllm`, and
`hf` are not implemented yet.
"""

import os
import time
from collections.abc import Callable
from typing import Any, Protocol

import anthropic
import httpx

from raterbench.llm.types import BackendError, LLMRequest, LLMResponse, TokenLogprob


class Backend(Protocol):
    supports_temperature: bool
    supports_logprobs: bool
    max_output_tokens: int

    async def complete(self, req: LLMRequest) -> LLMResponse: ...

    async def describe(self) -> dict[str, Any]: ...


def _retry_after(headers: Any) -> float | None:
    try:
        return float(headers.get("retry-after")) if headers is not None else None
    except (TypeError, ValueError):
        return None


class AnthropicBackend:
    """Claude Messages API. Sampling parameters (temperature etc.) are not accepted by
    current Claude models, so `req.temperature` must be None; repeated samples use the
    model's default sampling. Logprobs are not available."""

    supports_temperature = False
    supports_logprobs = False

    def __init__(self, model: str, params: dict[str, Any]):
        self.model = model
        self.max_output_tokens = int(params.get("max_tokens", 2048))
        self.effort = params.get("effort")
        self.timeout = float(params.get("timeout_s", 120))
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            # Retries are handled uniformly by LLMClient, so the SDK's own are disabled.
            self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self.timeout)
        return self._client

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if req.temperature is not None:
            raise ValueError("Claude models do not accept temperature; pass None")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": req.system,
            "messages": [m.model_dump() for m in req.messages],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.perf_counter()
        try:
            resp = await self._get_client().messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            retryable = e.status_code in (408, 409, 429) or e.status_code >= 500
            raise BackendError(
                f"anthropic {e.status_code}: {e.message}",
                retryable,
                _retry_after(e.response.headers),
            ) from e
        except anthropic.APIConnectionError as e:
            raise BackendError(f"anthropic connection error: {e}", retryable=True) from e
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            served_model=resp.model,
            stop_reason=resp.stop_reason,
            latency_s=time.perf_counter() - t0,
            meta={"request_id": getattr(resp, "_request_id", None)},
        )

    async def describe(self) -> dict[str, Any]:
        return {
            "backend": "anthropic",
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_output_tokens,
            "sdk_version": anthropic.__version__,
        }


class OllamaBackend:
    """Local models served by Ollama's /api/chat endpoint."""

    supports_temperature = True
    supports_logprobs = True

    def __init__(self, model: str, params: dict[str, Any]):
        self.model = model
        self.base_url = (
            params.get("base_url") or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.max_output_tokens = int(params.get("num_predict", 512))
        self.num_ctx = params.get("num_ctx")
        self.think = params.get("think")
        self.timeout = float(params.get("timeout_s", 300))
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def complete(self, req: LLMRequest) -> LLMResponse:
        options: dict[str, Any] = {"num_predict": self.max_output_tokens}
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.seed is not None:
            options["seed"] = req.seed
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}]
            + [m.model_dump() for m in req.messages],
            "stream": False,
            "options": options,
        }
        if req.want_logprobs:
            body["logprobs"] = True
        if self.think is not None:
            body["think"] = self.think
        t0 = time.perf_counter()
        try:
            r = await self._get_client().post("/api/chat", json=body)
        except httpx.ConnectError as e:
            raise BackendError(
                f"cannot reach Ollama at {self.base_url} (is `ollama serve` running?)", False
            ) from e
        except httpx.TransportError as e:
            raise BackendError(f"ollama transport error: {e}", retryable=True) from e
        if r.status_code != 200:
            hint = f" (try `ollama pull {self.model}`)" if r.status_code == 404 else ""
            raise BackendError(
                f"ollama {r.status_code}: {r.text[:300]}{hint}",
                retryable=r.status_code == 429 or r.status_code >= 500,
                retry_after=_retry_after(r.headers),
            )
        data = r.json()
        lps = data.get("logprobs")
        return LLMResponse(
            text=data["message"]["content"],
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            served_model=data.get("model", self.model),
            stop_reason=data.get("done_reason"),
            logprobs=[TokenLogprob(token=t["token"], logprob=t["logprob"]) for t in lps]
            if lps
            else None,
            latency_s=time.perf_counter() - t0,
        )

    async def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"backend": "ollama", "model": self.model, "base_url": self.base_url}
        try:
            r = await self._get_client().get("/api/tags")
            for m in r.json().get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    info["digest"] = m.get("digest")
                    info["details"] = m.get("details")
            v = await self._get_client().get("/api/version")
            info["ollama_version"] = v.json().get("version")
        except (httpx.HTTPError, ValueError) as e:
            info["describe_error"] = str(e)
        return info


BACKENDS: dict[str, Callable[[str, dict[str, Any]], Backend]] = {
    "anthropic": AnthropicBackend,
    "ollama": OllamaBackend,
}


def make_backend(backend: str, model: str, params: dict[str, Any]) -> Backend:
    if backend not in BACKENDS:
        raise ValueError(f"Backend {backend!r} not implemented; available: {sorted(BACKENDS)}")
    return BACKENDS[backend](model, params)
