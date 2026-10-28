"""Token/cost accounting with a hard budget cap.

Before each uncached call we reserve a worst-case cost (estimated input + max output
tokens). A call that could push spend past the cap is refused, so the cap holds even with
concurrent requests in flight.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel


class Pricing(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


class BudgetExceeded(Exception):
    pass


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (about 3 characters per token) for reservations."""
    return len(text) // 3 + 1


@dataclass
class ModelUsage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    budget_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    halted: bool = False  # once the cap is hit, every later uncached call is refused
    by_model: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))

    def reserve(self, amount: float) -> None:
        if self.halted:
            raise BudgetExceeded("Run halted: budget cap reached earlier")
        if self.spent_usd + self.reserved_usd + amount > self.budget_usd:
            self.halted = True
            raise BudgetExceeded(
                f"Budget cap ${self.budget_usd:.2f} would be exceeded "
                f"(spent ${self.spent_usd:.4f}, in flight ${self.reserved_usd:.4f}, "
                f"next call up to ${amount:.4f})"
            )
        self.reserved_usd += amount

    def settle(
        self, reserved: float, model: str, input_tokens: int, output_tokens: int, pricing: Pricing
    ) -> float:
        actual = pricing.cost(input_tokens, output_tokens)
        self.reserved_usd -= reserved
        self.spent_usd += actual
        u = self.by_model[model]
        u.calls += 1
        u.input_tokens += input_tokens
        u.output_tokens += output_tokens
        u.cost_usd += actual
        return actual

    def release(self, reserved: float) -> None:
        self.reserved_usd -= reserved

    def record_cached(self, model: str) -> None:
        self.by_model[model].cached_calls += 1

    def summary(self) -> dict:
        return {
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 6),
            "halted_by_budget": self.halted,
            "by_model": {
                m: {**u.__dict__, "cost_usd": round(u.cost_usd, 6)}
                for m, u in self.by_model.items()
            },
        }
