"""Multi-agent LLM analyst — a panel of role specialists that proposes ONE number.

This is the breadth factory's "TradingAgents"-style head: instead of a single extraction
call (see ``extraction.py``), several *role-specialised* analysts each read the same
point-in-time context bundle through a focused lens (news, price action, risk), and a
deterministic portfolio-manager step synthesises their views into a single bounded
:class:`AgentSignal`.

The same two hard rules as everywhere else in this tier apply, and they are the whole
reason the design looks the way it does:

* **The panel never sizes or triggers a trade.** Its only output is a structured candidate
  *number* (score in [-1, 1] + confidence in [0, 1]). Downstream that becomes an
  Observation -> a candidate Feature, and like every signal it must earn positive,
  significant out-of-sample IC before anything trusts it. We are producing a number for the
  gate to (dis)trust, not a decision.
* **Point-in-time + fail-soft.** Every system prompt forbids outside/future knowledge and
  demands STRICT JSON; an unparseable or erroring view collapses to *neutral* (score 0), so
  a bad analyst is a *missing* opinion, never a fabricated one. The panel as a whole can
  always produce *something* as long as it had any context to read.

Why a *deterministic* aggregator rather than a final LLM "debate" call? Three reasons that
matter for a quant pipeline: (1) **auditability** — the final number is a transparent,
reproducible function of the role views, so when the IC gate later judges this feature we
can attribute it; (2) **cost/latency** — it spends no extra LLM call beyond the role
analysts and cannot loop; (3) **leakage containment** — synthesis arithmetic cannot smuggle
in outside knowledge the way a free-form debate turn might. The combine is a
confidence-weighted vote of the *directional* analysts, with the risk analyst acting as a
confidence *damper* (it lowers conviction when it sees danger) rather than casting its own
directional vote — a risk read of "this looks dangerous" should shrink position-implying
conviction, not by itself flip the sign of the thesis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from market_trader.observability import get_logger
from market_trader.reasoning.budget import LLMBudgetExceeded
from market_trader.reasoning.extraction import parse_signal
from market_trader.reasoning.llm import LLMProvider

_log = get_logger("llm_agents")


class AnalystRole(StrEnum):
    """The lens each panellist reads the context through.

    ``DIRECTIONAL`` roles (news, technical) vote on the sign/strength of the thesis;
    ``RISK`` is a non-directional damper on conviction (see module docstring).
    """

    NEWS = "news"
    TECHNICAL = "technical"
    RISK = "risk"


# Shared no-lookahead preamble — repeated verbatim into every role prompt so the discipline
# can never drift between roles. It restates the point-in-time and fail-soft contract in the
# words the model must obey, and pins the exact JSON shape ``parse_signal`` expects.
_DISCIPLINE = (
    "Use ONLY the context provided below. Do not use any outside knowledge, and never "
    "reason about what happened after the point-in-time of this context (no lookahead). "
    "If the context is thin, immaterial, or absent, say so via a low confidence and a "
    "score near 0 — do not guess. Respond with STRICT JSON only, no prose and no code "
    'fences: {"score": <float -1..1>, "confidence": <float 0..1>, "rationale": <short '
    "string>}."
)

_NEWS_SYSTEM = (
    "You are the NEWS analyst on an equity research panel. Judge the materiality and "
    "directional sentiment of the recent headlines for this single company: -1 strongly "
    "bearish, 0 neutral/immaterial, +1 strongly bullish. Discount routine or promotional "
    "items. " + _DISCIPLINE
)

_TECHNICAL_SYSTEM = (
    "You are the TECHNICAL analyst on an equity research panel. From the provided recent "
    "price/return summary only, judge directional momentum and trend quality: -1 strongly "
    "bearish (breaking down), 0 no edge/range-bound, +1 strongly bullish (trending up). Do "
    "not infer fundamentals. " + _DISCIPLINE
)

_RISK_SYSTEM = (
    "You are the RISK analyst on an equity research panel. You do NOT pick a direction; you "
    "assess how DANGEROUS it would be to act on a thesis for this name right now given the "
    "provided context (e.g. crowding, volatility, headline/event risk, thin or conflicting "
    "evidence). Encode danger in the SIGN of score: score near -1 means high risk / low "
    "trust, score near +1 means calm / high trust, 0 means neutral. Confidence is how sure "
    "you are of that risk read. " + _DISCIPLINE
)

_DIRECTIONAL_SYSTEMS: dict[AnalystRole, str] = {
    AnalystRole.NEWS: _NEWS_SYSTEM,
    AnalystRole.TECHNICAL: _TECHNICAL_SYSTEM,
}


@dataclass(frozen=True)
class AnalystContext:
    """The per-symbol, point-in-time inputs the panel may read — plain Python only.

    Kept deliberately free of any store/DB handle so the panel is unit-testable with literal
    data. ``headlines`` feeds the news analyst; ``price_summary`` (a short pre-rendered line
    such as ``"20d return +4.1%, 5d -0.8%, realised vol 18%"``) feeds the technical analyst;
    both plus ``risk_notes`` feed the risk analyst. ``extra`` is an escape hatch for future
    context lines without changing the signature.
    """

    symbol: str
    headlines: Sequence[str] = field(default_factory=tuple)
    price_summary: str = ""
    risk_notes: Sequence[str] = field(default_factory=tuple)
    extra: Sequence[str] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        """True when there is nothing material to reason about (-> the panel returns None)."""
        return not (
            any(h.strip() for h in self.headlines if h)
            or self.price_summary.strip()
            or any(n.strip() for n in self.risk_notes if n)
            or any(x.strip() for x in self.extra if x)
        )


@dataclass(frozen=True)
class AnalystView:
    """One panellist's structured read. ``score``/``confidence`` are already bounded."""

    role: AnalystRole
    score: float  # [-1, 1]
    confidence: float  # [0, 1]
    rationale: str


@dataclass(frozen=True)
class AgentSignal:
    """The panel's synthesised candidate signal — the single number we hand downstream.

    Carries the contributing ``views`` purely for auditing/explanation; only ``score`` and
    ``confidence`` are meant to flow into the Observation/Feature (mirroring
    :class:`~market_trader.reasoning.extraction.ExtractedSignal`).
    """

    symbol: str
    score: float  # [-1, 1]
    confidence: float  # [0, 1]
    rationale: str
    views: tuple[AnalystView, ...] = ()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _render_context(ctx: AnalystContext, role: AnalystRole, *, max_headlines: int = 20) -> str:
    """Build the user prompt for ``role`` from ``ctx`` — only the lines that role should see.

    We scope each analyst to its own evidence so a role can't be swayed by material outside
    its remit (the technical analyst never sees headlines, etc.). The risk analyst is the
    one integrator and sees everything.
    """
    lines = [f"Company ticker: {ctx.symbol}"]
    show_headlines = role in (AnalystRole.NEWS, AnalystRole.RISK)
    show_price = role in (AnalystRole.TECHNICAL, AnalystRole.RISK)

    if show_headlines:
        clean = [h.strip() for h in ctx.headlines if h and h.strip()][:max_headlines]
        if clean:
            lines.append("Recent headlines:")
            lines.extend(f"- {h}" for h in clean)
    if show_price and ctx.price_summary.strip():
        lines.append(f"Recent price/return summary: {ctx.price_summary.strip()}")
    if role is AnalystRole.RISK:
        notes = [n.strip() for n in ctx.risk_notes if n and n.strip()]
        if notes:
            lines.append("Risk notes:")
            lines.extend(f"- {n}" for n in notes)
    extra = [x.strip() for x in ctx.extra if x and x.strip()]
    if extra:
        lines.append("Additional context:")
        lines.extend(f"- {x}" for x in extra)
    return "\n".join(lines)


def run_analyst(
    provider: LLMProvider,
    role: AnalystRole,
    ctx: AnalystContext,
    *,
    max_tokens: int = 300,
) -> AnalystView:
    """Run one role analyst over ``ctx`` and return its (always-bounded) view.

    Fail-soft: any malformed/unparseable reply or non-budget provider error becomes a
    *neutral* view (score 0, confidence 0) so one bad lens cannot crash or bias the panel.
    A :class:`LLMBudgetExceeded` is re-raised so the caller's sweep can stop cleanly rather
    than silently neutralising every name (same contract as ``extract_news_signal``).
    """
    system = _RISK_SYSTEM if role is AnalystRole.RISK else _DIRECTIONAL_SYSTEMS[role]
    prompt = _render_context(ctx, role)
    try:
        text = provider.complete(system=system, prompt=prompt, max_tokens=max_tokens)
    except LLMBudgetExceeded:
        raise  # never swallow the budget; let the batch stop
    except Exception as exc:  # one bad lens must not abort the panel
        _log.warning("agent_call_failed", symbol=ctx.symbol, role=str(role), error=str(exc))
        return AnalystView(role, 0.0, 0.0, "")
    parsed = parse_signal(ctx.symbol, text)
    if parsed is None:
        _log.warning("agent_view_unparseable", symbol=ctx.symbol, role=str(role))
        return AnalystView(role, 0.0, 0.0, "")
    return AnalystView(role, parsed.score, parsed.confidence, parsed.rationale)


class PortfolioManagerAgent:
    """Deterministic synthesiser: combine role views into one :class:`AgentSignal`.

    The aggregation (see module docstring for the rationale) is:

    * **Direction & strength** = confidence-weighted mean of the DIRECTIONAL roles' scores.
      A role that abstains (confidence 0 — e.g. a malformed reply) contributes nothing, so a
      single failed lens degrades gracefully toward the surviving ones rather than dragging
      the mean to 0.
    * **Base conviction** = mean confidence of the directional roles, scaled by their
      *agreement* (1 - normalised dispersion). Two analysts pointing opposite ways should not
      yield a confident net call even if each is individually sure.
    * **Risk damping** = the risk analyst trims conviction when it sees danger. Its score in
      [-1, 1] maps to a multiplier in ``[1 - risk_weight, 1]`` (calm -> ~1, dangerous -> as
      low as ``1 - risk_weight``), weighted by the risk analyst's own confidence. Risk never
      changes the *sign* of the thesis — it only lowers how much we trust it.

    The result is fully reproducible from the views, which is what lets the downstream IC
    gate attribute and judge this feature like any other.
    """

    def __init__(self, *, risk_weight: float = 0.5) -> None:
        # How much a maximally-dangerous, fully-confident risk read can cut conviction.
        # 0 disables risk damping entirely; clamp to [0, 1] so a multiplier never goes < 0.
        self.risk_weight = _clip(float(risk_weight), 0.0, 1.0)

    def synthesize(self, symbol: str, views: Sequence[AnalystView]) -> AgentSignal:
        directional = [v for v in views if v.role is not AnalystRole.RISK and v.confidence > 0.0]
        if not directional:
            # No usable directional opinion -> a neutral, low-trust signal (never a guess).
            return AgentSignal(symbol, 0.0, 0.0, "no usable analyst views", tuple(views))

        weight = sum(v.confidence for v in directional)
        score = sum(v.score * v.confidence for v in directional) / weight
        mean_conf = weight / len(directional)

        # Agreement: shrink conviction when the directional analysts disagree. Spread is the
        # confidence-weighted mean absolute deviation of scores from the combined score; the
        # /2.0 normalises it to [0, 1] since scores live in [-1, 1] (max deviation 2).
        spread = sum(abs(v.score - score) * v.confidence for v in directional) / weight
        agreement = _clip(1.0 - spread / 2.0, 0.0, 1.0)
        confidence = mean_conf * agreement

        risk = next((v for v in views if v.role is AnalystRole.RISK), None)
        if risk is not None and risk.confidence > 0.0:
            # score +1 (calm) -> multiplier 1; score -1 (dangerous) -> 1 - risk_weight.
            # Blend toward 1 by how confident the risk analyst is in that read.
            danger_mult = 1.0 - self.risk_weight * (1.0 - risk.score) / 2.0
            confidence *= 1.0 - risk.confidence * (1.0 - danger_mult)

        return AgentSignal(
            symbol=symbol,
            score=_clip(score, -1.0, 1.0),
            confidence=_clip(confidence, 0.0, 1.0),
            rationale=_summarize(views),
            views=tuple(views),
        )


def _summarize(views: Sequence[AnalystView]) -> str:
    """One-line audit trail of which lens said what, bounded like every other rationale."""
    parts = [
        f"{v.role}: {v.score:+.2f}@{v.confidence:.2f}"
        for v in views
        if v.confidence > 0.0 or v.rationale
    ]
    return ("panel — " + "; ".join(parts))[:300] if parts else "panel — no usable views"


def run_analyst_panel(
    provider: LLMProvider,
    *,
    symbol: str,
    context: AnalystContext,
    roles: Sequence[AnalystRole] = (
        AnalystRole.NEWS,
        AnalystRole.TECHNICAL,
        AnalystRole.RISK,
    ),
    pm: PortfolioManagerAgent | None = None,
) -> AgentSignal | None:
    """Run the role panel over ``context`` and synthesise one candidate :class:`AgentSignal`.

    Returns ``None`` when there is nothing to read (empty context) — a *missing* signal, not
    a neutral 0, so the collector can skip the name entirely rather than persist a noise
    point. Otherwise always returns a bounded signal (possibly neutral if every lens
    abstained). A :class:`LLMBudgetExceeded` from any analyst propagates so a universe sweep
    can stop and resume; all other per-analyst errors fail soft inside :func:`run_analyst`.

    ``provider`` may be any :class:`LLMProvider`, including a
    :class:`~market_trader.reasoning.budget.BudgetedProvider`, so the panel's cost is bounded
    exactly like single-call extraction.
    """
    if context.is_empty():
        return None
    if symbol and context.symbol != symbol:
        # Guard against a context bundle built for a different name reaching the model.
        context = AnalystContext(
            symbol=symbol,
            headlines=context.headlines,
            price_summary=context.price_summary,
            risk_notes=context.risk_notes,
            extra=context.extra,
        )

    views = [run_analyst(provider, role, context) for role in roles]
    signal = (pm or PortfolioManagerAgent()).synthesize(context.symbol, views)
    _log.info(
        "analyst_panel",
        symbol=context.symbol,
        score=round(signal.score, 4),
        confidence=round(signal.confidence, 4),
        roles=len(views),
    )
    return signal
