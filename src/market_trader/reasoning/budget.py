"""Daily call budget for LLM use — make the cost gate real (it was an inert setting).

Wraps any :class:`LLMProvider` with a per-process call counter that fails closed once
the budget is spent, so a batch extraction over a large universe can never run away on
cost. Decisions never depend on the LLM (see ``reasoning/__init__``), but the *cost* of
using it must be bounded; this is that bound.
"""

from __future__ import annotations

from market_trader.reasoning.llm import LLMError, LLMProvider


class LLMBudgetExceeded(LLMError):
    """Raised when the daily LLM call budget is exhausted (fail closed)."""


class BudgetedProvider(LLMProvider):
    """An :class:`LLMProvider` that allows at most ``budget`` completions, then fails closed."""

    def __init__(self, inner: LLMProvider, *, budget: int) -> None:
        self._inner = inner
        self._budget = max(0, int(budget))
        self.calls = 0

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self.calls)

    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if self.calls >= self._budget:
            raise LLMBudgetExceeded(f"LLM daily call budget {self._budget} exhausted")
        self.calls += 1
        return self._inner.complete(system=system, prompt=prompt, max_tokens=max_tokens)
