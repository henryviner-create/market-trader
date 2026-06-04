"""Self-evaluation: a journal from matured decisions, attribution by signal/regime,
episodic recall, and reflection — all offline on a synthetic store."""

from __future__ import annotations

from datetime import date, datetime

from market_trader.core.synthetic import synthetic_price_observations
from market_trader.core.time import UTC
from market_trader.features import FeatureStore, default_features
from market_trader.portfolio import composite_score, equal_weights
from market_trader.reasoning.llm import MockLLMProvider
from market_trader.runtime.evaluation import (
    JournalEntry,
    analog_outcomes,
    attribute_performance,
    build_episodic_memory,
    build_trade_journal,
    evaluation_summary_markdown,
    reflect,
)
from market_trader.runtime.learning import log_cycle_predictions
from market_trader.storage import InMemoryBitemporalStore


def _store(symbols: list[str], n_days: int = 120):
    obs = synthetic_price_observations(
        symbols=symbols, start=date(2023, 1, 2), n_days=n_days, seed=5
    )
    store = InMemoryBitemporalStore()
    store.add_many(obs)
    return store, sorted({o.event_time for o in obs})


def _journal(regime: str = "risk_on") -> list[JournalEntry]:
    # 'good' tracks the realised return; 'noise' is constant -> no IC.
    out: list[JournalEntry] = []
    for i, r in enumerate([0.05, 0.03, -0.02, -0.04, 0.01, -0.01]):
        out.append(
            JournalEntry(
                symbol=f"X{i}",
                as_of=datetime(2023, 1, 2 + i, tzinfo=UTC),
                probability=0.5 + r,
                realized_return=r,
                won=r > 0,
                regime=regime,
                features={"good": r * 10, "noise": 1.0},
            )
        )
    return out


def test_build_trade_journal_only_includes_matured_decisions() -> None:
    symbols = [f"S{i}" for i in range(10)]
    store, dates = _store(symbols)
    fs = FeatureStore(store, default_features())

    early = dates[-15]  # leave room for the 5-day forward window
    matrix = fs.compute_matrix(early, symbols)
    log_cycle_predictions(
        store,
        composite_score(matrix, equal_weights(matrix.columns)),
        matrix,
        early,
        model_version="composite",
    )
    journal = build_trade_journal(store, dates[-1], model_version="composite", horizon_days=5)
    assert journal  # matured -> entries exist
    assert journal[0].regime and journal[0].features

    # Logged at the very last bar -> nothing has matured.
    m2 = fs.compute_matrix(dates[-1], symbols)
    log_cycle_predictions(
        store, composite_score(m2, equal_weights(m2.columns)), m2, dates[-1], model_version="fresh"
    )
    assert build_trade_journal(store, dates[-1], model_version="fresh") == []


def test_attribute_performance_finds_the_predictive_signal() -> None:
    report = attribute_performance(_journal())
    assert report.n == 6
    assert report.ic["good"] > 0.9  # ~collinear with the return
    assert "noise" not in report.ic  # constant signal -> skipped, no /0
    assert report.by_regime and report.by_regime[0].regime == "risk_on"
    assert 0.0 <= report.hit_rate <= 1.0


def test_episodic_recall_surfaces_analog_outcomes() -> None:
    journal = _journal()
    assert len(build_episodic_memory(journal)) == len(journal)
    # Query like a strong-'good' winner -> positive analog outcomes.
    dist = analog_outcomes(journal, {"good": 0.5, "noise": 1.0}, k=3)
    assert dist["n"] == 3 and dist["mean"] > 0


def test_reflect_falls_back_without_llm_and_uses_provider_when_given() -> None:
    report = attribute_performance(_journal())
    md = evaluation_summary_markdown(report)
    assert "Self-evaluation" in md and "Signal IC" in md

    assert reflect(report, _journal()) == md  # no LLM -> deterministic summary
    llm = MockLLMProvider(canned="POST-MORTEM")
    assert reflect(report, _journal(), llm) == "POST-MORTEM" and llm.calls


def test_attribute_and_markdown_are_empty_safe() -> None:
    empty = attribute_performance([])
    assert empty.n == 0
    assert "nothing matured" in evaluation_summary_markdown(empty)
