"""The daily pre-market briefing.

``build_briefing_context`` assembles the day's point-in-time data (regime, ranked
signals, flow, news). ``render_brief_markdown`` turns it into a deterministic,
LLM-free brief (the system always produces *something*). ``generate_llm_brief``
adds Claude's narration — constrained to the data, and required to state the case
against. Either way it is decision-support, never a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from market_trader.features import (
    FeatureStore,
    cross_sectional_zscore,
    default_features,
    macro_regime,
)
from market_trader.presentation import build_dashboard_data
from market_trader.reasoning.llm import LLMProvider
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe import PointInTimeUniverse

_SYSTEM = (
    "You are a buy-side analyst producing a pre-market briefing. Strict rules: "
    "(1) assert only what the provided signals/flow/news support; do not invent "
    "facts or prices. (2) For each name give catalysts, risks, AND the explicit "
    "case against the quant score. (3) Quantify uncertainty; never imply "
    "certainty. This is decision-support, not financial advice."
)


def _as_float(x: Any) -> float:
    return float(x)


@dataclass
class BriefingContext:
    as_of: datetime
    regime: dict[str, Any]
    top_signals: list[dict[str, Any]]
    insider: list[dict[str, Any]]
    congress: list[dict[str, Any]]
    news: list[dict[str, Any]]


def build_briefing_context(
    store: BitemporalStore,
    as_of: datetime,
    *,
    feature_store: FeatureStore | None = None,
    universe: PointInTimeUniverse | None = None,
    top_n: int = 5,
) -> BriefingContext:
    dash = build_dashboard_data(store, as_of, universe=universe)
    symbols = dash.watchlist
    fs = feature_store or FeatureStore(store, default_features())
    matrix = fs.compute_matrix(as_of, symbols)

    top_signals: list[dict[str, Any]] = []
    if not matrix.empty and symbols:
        z = matrix.apply(cross_sectional_zscore, axis=0)
        composite = z.mean(axis=1, skipna=True)
        for sym in composite.dropna().sort_values(ascending=False).head(top_n).index:
            feats = {
                c: (
                    None if pd.isna(matrix.loc[sym, c]) else round(_as_float(matrix.loc[sym, c]), 4)
                )
                for c in matrix.columns
            }
            top_signals.append(
                {
                    "symbol": str(sym),
                    "score": round(_as_float(composite[sym]), 3),
                    "features": feats,
                }
            )

    return BriefingContext(
        as_of=as_of,
        regime=macro_regime(store, as_of),
        top_signals=top_signals,
        insider=dash.recent_insider,
        congress=dash.recent_congress,
        news=dash.recent_news,
    )


def render_brief_markdown(ctx: BriefingContext) -> str:
    lines = [
        f"# Pre-market briefing — {ctx.as_of:%Y-%m-%d}",
        "",
        f"**Regime:** {ctx.regime['label']} "
        f"(yield-curve slope: {ctx.regime['yield_curve_slope']})",
        "",
        "## Top signals (composite z-score)",
    ]
    if ctx.top_signals:
        for s in ctx.top_signals:
            feats = ", ".join(f"{k}={v}" for k, v in s["features"].items())
            lines.append(f"- **{s['symbol']}** (score {s['score']}): {feats}")
    else:
        lines.append("- (insufficient data for ranking)")

    lines += [
        "",
        "## Flow",
        f"- Insider (Form 4) items: {len(ctx.insider)}",
        f"- Congressional items: {len(ctx.congress)}",
    ]
    for c in ctx.congress[:5]:
        lines.append(
            f"  - {c.get('entity_id')}: {c.get('transaction_type')} by {c.get('representative')}"
        )

    lines += ["", "## News"]
    if ctx.news:
        lines += [
            f"- {n.get('entity_id')}: {n.get('title')} (tone {n.get('tone')})" for n in ctx.news[:5]
        ]
    else:
        lines.append("- (none)")

    lines += ["", "_Decision-support only. Not financial advice. Signals are inputs to judgement._"]
    return "\n".join(lines)


def generate_llm_brief(
    ctx: BriefingContext, provider: LLMProvider, *, max_tokens: int = 1200
) -> str:
    prompt = (
        "Here is today's point-in-time data block. Write a concise per-name thesis "
        "for the top signals, each with catalysts, risks, and the case against.\n\n"
        f"{render_brief_markdown(ctx)}"
    )
    return provider.complete(system=_SYSTEM, prompt=prompt, max_tokens=max_tokens)
