"""The multi-agent analyst panel: role views + deterministic synthesis, fully offline.

Every test uses a canned/scripted provider — no network — so the panel's *logic* (not a
model's output) is what's under test. We assert the panel keeps the tier's hard contract:
it produces a bounded candidate number, never crashes on bad output, never invents a signal
out of nothing, propagates a budget stop, and feeds the model only point-in-time context
with the no-lookahead discipline attached.
"""

from __future__ import annotations

import pytest

from market_trader.reasoning.agents import (
    AgentSignal,
    AnalystContext,
    AnalystRole,
    AnalystView,
    PortfolioManagerAgent,
    run_analyst_panel,
)
from market_trader.reasoning.budget import BudgetedProvider, LLMBudgetExceeded
from market_trader.reasoning.llm import LLMProvider, MockLLMProvider


class ScriptedProvider(LLMProvider):
    """Returns a different canned reply keyed by which analyst is calling.

    The role is identifiable from the ``system`` prompt (each role names itself, e.g. "NEWS
    analyst"), so one stub can voice the whole panel deterministically. Unmatched roles fall
    back to ``default`` and every call is recorded for prompt-content assertions.
    """

    def __init__(self, by_role: dict[AnalystRole, str], *, default: str = "{}") -> None:
        self._by_role = by_role
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append((system, prompt))
        for role, reply in self._by_role.items():
            if f"{role.name} analyst" in system:
                return reply
        return self._default


class RaisingProvider(LLMProvider):
    """Always raises a generic error — exercises the fail-soft path (not budget)."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        raise RuntimeError("provider exploded")


def _bullish_panel() -> ScriptedProvider:
    return ScriptedProvider(
        {
            AnalystRole.NEWS: '{"score": 0.8, "confidence": 0.9, "rationale": "beat + raise"}',
            AnalystRole.TECHNICAL: '{"score": 0.6, "confidence": 0.7, "rationale": "uptrend"}',
            AnalystRole.RISK: '{"score": 0.5, "confidence": 0.6, "rationale": "calm tape"}',
        }
    )


def _ctx() -> AnalystContext:
    return AnalystContext(
        symbol="ABC",
        headlines=["ABC beats and raises guidance", "Analysts upgrade ABC"],
        price_summary="20d return +6.2%, 5d +1.1%, realised vol 17%",
        risk_notes=["earnings already passed; low event risk"],
    )


def test_panel_aggregates_a_sensible_bullish_signal() -> None:
    provider = _bullish_panel()
    sig = run_analyst_panel(provider, symbol="ABC", context=_ctx())

    assert isinstance(sig, AgentSignal)
    assert sig.symbol == "ABC"
    # Confidence-weighted mean of the directional roles: (0.8*0.9 + 0.6*0.7)/(0.9+0.7).
    assert sig.score == pytest.approx((0.8 * 0.9 + 0.6 * 0.7) / (0.9 + 0.7))
    assert 0.0 < sig.score <= 1.0 and 0.0 < sig.confidence <= 1.0
    assert len(sig.views) == 3  # all three lenses are recorded for audit
    assert "news" in sig.rationale and "technical" in sig.rationale
    assert len(provider.calls) == 3  # one LLM call per role, no extra synthesis call


def test_each_analyst_gets_only_point_in_time_context_with_discipline() -> None:
    provider = _bullish_panel()
    run_analyst_panel(provider, symbol="ABC", context=_ctx())

    systems = [s for s, _ in provider.calls]
    prompts = [p for _, p in provider.calls]
    # No-lookahead discipline reaches every role.
    assert all("no lookahead" in s.lower() for s in systems)
    assert all("only the context" in s.lower() for s in systems)
    # Point-in-time data reached the model; the technical lens is scoped (no headlines).
    assert any("ABC beats and raises guidance" in p for p in prompts)
    tech_prompt = next(p for s, p in provider.calls if f"{AnalystRole.TECHNICAL.name} analyst" in s)
    assert "beats and raises" not in tech_prompt  # scoped to its own evidence
    assert "price/return summary" in tech_prompt


def test_risk_analyst_damps_conviction_without_flipping_the_sign() -> None:
    base = run_analyst_panel(_bullish_panel(), symbol="ABC", context=_ctx())
    dangerous = ScriptedProvider(
        {
            AnalystRole.NEWS: '{"score": 0.8, "confidence": 0.9, "rationale": "beat"}',
            AnalystRole.TECHNICAL: '{"score": 0.6, "confidence": 0.7, "rationale": "up"}',
            AnalystRole.RISK: '{"score": -0.9, "confidence": 0.9, "rationale": "crowded, vol"}',
        }
    )
    risky = run_analyst_panel(dangerous, symbol="ABC", context=_ctx())

    assert base is not None and risky is not None
    assert risky.score == pytest.approx(base.score)  # same direction/strength
    assert risky.confidence < base.confidence  # but trusted less under danger
    assert risky.score > 0  # risk never flips the thesis sign


def test_disagreement_between_analysts_lowers_confidence() -> None:
    agree = ScriptedProvider(
        {
            AnalystRole.NEWS: '{"score": 0.7, "confidence": 0.8, "rationale": "x"}',
            AnalystRole.TECHNICAL: '{"score": 0.7, "confidence": 0.8, "rationale": "y"}',
            AnalystRole.RISK: '{"score": 0.0, "confidence": 0.0, "rationale": "n/a"}',
        }
    )
    conflict = ScriptedProvider(
        {
            AnalystRole.NEWS: '{"score": 0.7, "confidence": 0.8, "rationale": "x"}',
            AnalystRole.TECHNICAL: '{"score": -0.7, "confidence": 0.8, "rationale": "y"}',
            AnalystRole.RISK: '{"score": 0.0, "confidence": 0.0, "rationale": "n/a"}',
        }
    )
    a = run_analyst_panel(agree, symbol="ABC", context=_ctx())
    c = run_analyst_panel(conflict, symbol="ABC", context=_ctx())

    assert a is not None and c is not None
    assert c.confidence < a.confidence  # opposing views must not yield a confident call
    assert c.score == pytest.approx(0.0)  # and they cancel toward neutral


def test_malformed_agent_output_is_handled_gracefully() -> None:
    # News lens returns garbage; the panel must not crash and must lean on the survivors.
    provider = ScriptedProvider(
        {
            AnalystRole.NEWS: "I think it's probably fine, hard to say",
            AnalystRole.TECHNICAL: '{"score": 0.5, "confidence": 0.6, "rationale": "trend"}',
            AnalystRole.RISK: '{"score": 0.4, "confidence": 0.5, "rationale": "ok"}',
        }
    )
    sig = run_analyst_panel(provider, symbol="ABC", context=_ctx())

    assert sig is not None
    # The malformed news view abstained (confidence 0) -> score is the surviving technical
    # view's, never dragged toward 0 by the failed lens.
    assert sig.score == pytest.approx(0.5)
    news_view = next(v for v in sig.views if v.role is AnalystRole.NEWS)
    assert news_view.score == 0.0 and news_view.confidence == 0.0


def test_all_lenses_failing_soft_yields_a_neutral_signal_not_a_crash() -> None:
    sig = run_analyst_panel(RaisingProvider(), symbol="ABC", context=_ctx())
    assert sig is not None  # context existed, so we still return a (neutral) signal
    assert sig.score == 0.0 and sig.confidence == 0.0
    assert all(v.confidence == 0.0 for v in sig.views)


def test_empty_context_returns_none_no_call() -> None:
    provider = MockLLMProvider('{"score": 0.9, "confidence": 0.9}')
    empty = AnalystContext(symbol="ABC", headlines=["", "   "], price_summary="  ")
    assert run_analyst_panel(provider, symbol="ABC", context=empty) is None
    assert provider.calls == []  # nothing to read -> no LLM spend at all


def test_budget_exhaustion_propagates_so_the_sweep_can_stop() -> None:
    # Budget of 0: the very first analyst call must fail closed and bubble out of the panel.
    provider = BudgetedProvider(MockLLMProvider('{"score": 0.1, "confidence": 0.2}'), budget=0)
    with pytest.raises(LLMBudgetExceeded):
        run_analyst_panel(provider, symbol="ABC", context=_ctx())


def test_budget_stops_mid_panel_after_partial_spend() -> None:
    # Budget of 1: first analyst succeeds, the second trips the budget and propagates.
    provider = BudgetedProvider(_bullish_panel(), budget=1)
    with pytest.raises(LLMBudgetExceeded):
        run_analyst_panel(provider, symbol="ABC", context=_ctx())
    assert provider.calls == 1  # exactly the budgeted spend reached the inner provider


def test_synthesize_with_no_directional_views_is_neutral() -> None:
    # Direct unit on the PM: only a risk view, no directional opinion -> neutral, low trust.
    pm = PortfolioManagerAgent()
    sig = pm.synthesize("ABC", [AnalystView(AnalystRole.RISK, -0.5, 0.9, "danger")])
    assert sig.score == 0.0 and sig.confidence == 0.0
    assert sig.symbol == "ABC"


def test_panel_normalises_a_mismatched_context_symbol() -> None:
    # A bundle built for the wrong name must not reach the model with that name.
    provider = _bullish_panel()
    ctx = AnalystContext(symbol="WRONG", headlines=["ABC beats guidance"])
    sig = run_analyst_panel(provider, symbol="ABC", context=ctx)
    assert sig is not None and sig.symbol == "ABC"
    assert all("Company ticker: ABC" in p for _, p in provider.calls)
    assert all("WRONG" not in p for _, p in provider.calls)
