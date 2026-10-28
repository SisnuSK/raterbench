"""LLMClient: one interface over all backends, with caching, budget, concurrency, retries."""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

from raterbench.llm.backends import Backend, make_backend
from raterbench.llm.cache import ResponseCache
from raterbench.llm.cost import CostTracker, Pricing, estimate_tokens
from raterbench.llm.types import BackendError, LLMRequest, LLMResponse, cache_key

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        name: str,
        backend_name: str,
        model: str,
        params: dict[str, Any],
        pricing: Pricing,
        concurrency: int,
        cache: ResponseCache,
        tracker: CostTracker,
        backend: Backend | None = None,
        max_attempts: int = 6,
        base_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.name = name
        self.backend_name = backend_name
        self.model = model
        self.params = params
        self.pricing = pricing
        self.backend = backend or make_backend(backend_name, model, params)
        self.cache = cache
        self.tracker = tracker
        self._sem = asyncio.Semaphore(concurrency)
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self._sleep = sleep
        self._jitter = random.Random()

    @property
    def supports_temperature(self) -> bool:
        return self.backend.supports_temperature

    @property
    def supports_logprobs(self) -> bool:
        return self.backend.supports_logprobs

    def key(self, req: LLMRequest) -> str:
        return cache_key(self.backend_name, self.model, self.params, req)

    def max_call_cost(self, req: LLMRequest) -> float:
        prompt = req.system + "".join(m.content for m in req.messages)
        return self.pricing.cost(estimate_tokens(prompt), self.backend.max_output_tokens)

    async def complete(self, req: LLMRequest) -> tuple[LLMResponse, bool, float]:
        """Return (response, from_cache, cost_usd). Raises BudgetExceeded or BackendError."""
        key = self.key(req)
        hit = self.cache.get(key)
        if hit is not None:
            self.tracker.record_cached(self.name)
            return hit, True, 0.0

        reserved = self.max_call_cost(req)
        self.tracker.reserve(reserved)
        try:
            async with self._sem:
                resp = await self._with_retries(req)
        except BaseException:
            self.tracker.release(reserved)
            raise
        cost = self.tracker.settle(
            reserved, self.name, resp.input_tokens, resp.output_tokens, self.pricing
        )
        self.cache.put(key, resp)
        return resp, False, cost

    async def _with_retries(self, req: LLMRequest) -> LLMResponse:
        for attempt in range(self.max_attempts):
            try:
                return await self.backend.complete(req)
            except BackendError as e:
                if not e.retryable or attempt == self.max_attempts - 1:
                    raise
                delay = e.retry_after or min(self.base_delay_s * 2**attempt, self.max_delay_s)
                delay *= 0.5 + self._jitter.random()
                log.warning(
                    "retrying call",
                    extra={
                        "model": self.name,
                        "attempt": attempt + 1,
                        "delay_s": round(delay, 2),
                        "error": str(e),
                    },
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")
