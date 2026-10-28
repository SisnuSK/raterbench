from collections.abc import Callable

from raterbench.llm.types import BackendError, LLMRequest, LLMResponse


class FakeBackend:
    supports_temperature = True
    supports_logprobs = False
    max_output_tokens = 100

    def __init__(
        self,
        reply: Callable[[LLMRequest], str] = lambda req: '{"label": "x", "confidence": 90}',
        errors: list[BackendError] | None = None,
    ):
        self.reply = reply
        self.errors = list(errors or [])
        self.calls: list[LLMRequest] = []

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(req)
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(
            text=self.reply(req), input_tokens=1000, output_tokens=100, served_model="fake-1"
        )

    async def describe(self) -> dict:
        return {"backend": "fake"}


async def no_sleep(_: float) -> None:
    return None
