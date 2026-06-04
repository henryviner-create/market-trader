"""Self-evaluation: judge our own decisions and surface what to learn.

Builds a *decision journal* from logged predictions whose horizon has elapsed
(each entry = a decision + its realised forward return + the regime it was made
in), attributes performance to signals and regimes, recalls analogous past
setups via episodic memory, and (optionally) asks the LLM for an honest
post-mortem. This is the **read side** of the learning loop — measurement and
insight only; it changes no trade behaviour. It reuses the grading machinery in
``runtime/learning`` and the (previously disconnected) ``memory/episodic``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.core.time import DISTANT_FUTURE
from market_trader.features import macro_regime
from market_trader.feedback.prediction_log import load_predictions
from market_trader.memory.episodic import Episode, EpisodicMemory
from market_trader.reasoning.llm import LLMProvider
from market_trader.runtime.learning import _forward_returns_at
from market_trader.storage.bitemporal import BitemporalStore


@dataclass(frozen=True)
class JournalEntry:
    symbol: str
    as_of: datetime
    probability: float
    realized_return: float
    won: bool
    regime: str
    features: dict[str, float]


@dataclass(frozen=True)
class RegimeStats:
    regime: str
    n: int
    hit_rate: float
    mean_return: float
    ic: dict[str, float]


@dataclass(frozen=True)
class AttributionReport:
    n: int
    hit_rate: float
    mean_return: float
    ic: dict[str, float]  # per-signal IC, overall
    by_regime: list[RegimeStats]


# --- the journal ------------------------------------------------------------


def build_trade_journal(
    store: BitemporalStore,
    as_of: datetime,
    *,
    model_version: str,
    horizon_days: int = 5,
) -> list[JournalEntry]:
    """One entry per *matured* decision: its realised forward return + the regime.

    Reuses the prediction log and the same matured-only forward-return rule as
    grading, so an entry only appears once its full horizon has elapsed.
    """
    preds = load_predictions(store, as_of, model_version=model_version)
    panel = observations_to_price_frame(store.as_of(DISTANT_FUTURE, dataset=PRICE_DATASET))
    by_time: dict[datetime, list] = defaultdict(list)
    for p in preds:
        by_time[p.as_of].append(p)

    journal: list[JournalEntry] = []
    for t, group in by_time.items():
        forward = (
            _forward_returns_at(panel, t, horizon_days)
            if not panel.empty
            else pd.Series(dtype=float)
        )
        if forward.empty:
            continue  # horizon not yet elapsed
        regime = str(macro_regime(store, t).get("label", "unknown"))
        for p in group:
            r = forward.get(p.symbol)
            if r is None or pd.isna(r):
                continue
            journal.append(
                JournalEntry(
                    symbol=p.symbol,
                    as_of=t,
                    probability=p.probability,
                    realized_return=float(r),
                    won=float(r) > 0,
                    regime=regime,
                    features=p.features,
                )
            )
    return journal


# --- attribution ------------------------------------------------------------


def _signal_ic(frame: pd.DataFrame, ret: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in frame.columns:
        std = frame[col].std(skipna=True)
        if pd.notna(std) and std > 0:
            v = frame[col].corr(ret)
            if pd.notna(v):
                out[str(col)] = float(v)
    return out


def attribute_performance(journal: list[JournalEntry]) -> AttributionReport:
    """Per-signal IC and hit-rate/mean-return, overall and split by regime."""
    if not journal:
        return AttributionReport(0, 0.0, 0.0, {}, [])
    feat_df = pd.DataFrame([e.features for e in journal])
    ret = pd.Series([e.realized_return for e in journal])
    won = pd.Series([e.won for e in journal])

    by_regime: list[RegimeStats] = []
    for regime in sorted({e.regime for e in journal}):
        sub = [e for e in journal if e.regime == regime]
        sub_ret = pd.Series([e.realized_return for e in sub])
        by_regime.append(
            RegimeStats(
                regime=regime,
                n=len(sub),
                hit_rate=float(sum(e.won for e in sub) / len(sub)),
                mean_return=float(sub_ret.mean()),
                ic=_signal_ic(pd.DataFrame([e.features for e in sub]), sub_ret),
            )
        )
    return AttributionReport(
        n=len(journal),
        hit_rate=float(won.mean()),
        mean_return=float(ret.mean()),
        ic=_signal_ic(feat_df, ret),
        by_regime=by_regime,
    )


# --- episodic recall (wires the existing memory/episodic) -------------------


def _feature_frame(journal: list[JournalEntry]) -> pd.DataFrame:
    return pd.DataFrame([e.features for e in journal]).fillna(0.0)


def _standardizer(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return frame.mean(), frame.std(ddof=0).replace(0.0, 1.0)


def build_episodic_memory(journal: list[JournalEntry]) -> EpisodicMemory:
    """One standardised episode per decision so Euclidean analog distance is meaningful."""
    mem = EpisodicMemory()
    if not journal:
        return mem
    frame = _feature_frame(journal)
    mu, sd = _standardizer(frame)
    z = (frame - mu) / sd
    for i, e in enumerate(journal):
        mem.add(
            Episode(
                key=f"{e.symbol}@{e.as_of:%Y-%m-%d}",
                vector=z.iloc[i].to_numpy(dtype=float),
                outcome=e.realized_return,
                regime=e.regime,
            )
        )
    return mem


def analog_outcomes(
    journal: list[JournalEntry],
    query_features: dict[str, float],
    *,
    regime: str | None = None,
    k: int = 8,
) -> dict:
    """How the k closest past setups resolved (mean/median/q10/q90/share-positive)."""
    if not journal:
        return {"n": 0, "mean": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0, "share_positive": 0.0}
    frame = _feature_frame(journal)
    mu, sd = _standardizer(frame)
    query = pd.Series({c: float(query_features.get(str(c), 0.0)) for c in frame.columns})
    qz = ((query - mu) / sd).to_numpy(dtype=float)
    return build_episodic_memory(journal).outcome_distribution(qz, k, regime=regime)


# --- presentation + reflection ----------------------------------------------


def evaluation_summary_markdown(report: AttributionReport) -> str:
    if report.n == 0:
        return "# Self-evaluation\n\n(nothing matured yet — run cycles, then wait the horizon)"
    lines = [
        f"# Self-evaluation — {report.n} graded decisions",
        "",
        f"- hit-rate: {report.hit_rate:.1%}",
        f"- mean decision return: {report.mean_return:+.2%}",
        "",
        "## Signal IC (vs forward return)",
    ]
    for name, val in sorted(report.ic.items(), key=lambda kv: abs(kv[1]), reverse=True):
        lines.append(f"- {name}: {val:+.3f}")
    lines += ["", "## By regime"]
    for rs in report.by_regime:
        lines.append(
            f"- **{rs.regime}** (n={rs.n}): hit {rs.hit_rate:.0%}, mean {rs.mean_return:+.2%}"
        )
        top = sorted(rs.ic.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        if top:
            lines.append("  - top IC: " + ", ".join(f"{k} {v:+.3f}" for k, v in top))
    lines += ["", "_Decision-level realised returns (point-in-time). Not advice._"]
    return "\n".join(lines)


_REFLECT_SYSTEM = (
    "You are a trading-desk risk reviewer doing an honest post-mortem. Rules: "
    "(1) ground every claim in the provided attribution numbers; invent nothing. "
    "(2) say what worked, what didn't, and the case that any apparent edge is noise. "
    "(3) quantify; never imply certainty. Decision-support, not financial advice."
)


def reflect(
    report: AttributionReport, journal: list[JournalEntry], llm: LLMProvider | None = None
) -> str:
    """Deterministic summary, or an LLM post-mortem when a provider is supplied."""
    summary = evaluation_summary_markdown(report)
    if llm is None:
        return summary
    notable = ""
    if journal:
        ranked = sorted(journal, key=lambda e: e.realized_return)
        picks = ranked[-2:][::-1] + ranked[:2]  # best two, worst two
        notable = "\n\nNotable decisions: " + "; ".join(
            f"{e.symbol} {e.realized_return:+.1%} ({e.regime})" for e in picks
        )
    prompt = (
        "Here is the system's own performance attribution. Write a brief, honest "
        "post-mortem: which signals are carrying, which have decayed, any regime "
        "pattern, and whether the apparent edge could be noise.\n\n" + summary + notable
    )
    return llm.complete(system=_REFLECT_SYSTEM, prompt=prompt, max_tokens=800)
