"""Pluggable scorers for the cycle: equal-weight composite vs trained forecaster.

The cycle ranks names by a ``ScoreFn``. The default is the transparent
equal-weight composite. The opt-in ``forecast`` scorer trains the calibrated
ensemble (``forecasting/``) on point-in-time samples from the store and ranks by
P(out-perform). It is **off by default on purpose**: a model only earns its place
by beating the equal-weight baseline out-of-sample, which ``forecaster_vs_baseline_auc``
measures via purged cross-validation. Nothing here changes default behaviour.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from market_trader.backtest.pit import observations_to_price_frame
from market_trader.config import Settings
from market_trader.core.synthetic import PRICE_DATASET
from market_trader.features import FeatureStore, default_features
from market_trader.features.regime import macro_regime
from market_trader.forecasting.dataset import REGIME_FEATURE, build_training_set
from market_trader.forecasting.models import MomentumBaseline, make_stacking
from market_trader.forecasting.pipeline import purged_cv_auc, train_forecaster
from market_trader.portfolio import composite_score, equal_weights, ic_weights
from market_trader.storage.bitemporal import BitemporalStore
from market_trader.universe import Constituent, PointInTimeUniverse

# A scorer maps (feature matrix, decision time) -> a per-symbol score Series.
ScoreFn = Callable[[pd.DataFrame, datetime], pd.Series]


def composite_scorer() -> ScoreFn:
    """The default: an equal-weighted z-score composite of the features."""

    def score(matrix: pd.DataFrame, _at: datetime) -> pd.Series:
        return composite_score(matrix, equal_weights(matrix.columns))

    return score


def ic_weighted_scorer(ic: dict[str, float], *, min_abs_ic: float = 0.0) -> ScoreFn:
    """Composite weighted by each signal's measured IC, dropping the weak ones.

    Sign-aware (a negative-IC signal is inverted, not discarded) and self-pruning
    (|IC| < ``min_abs_ic`` -> zero weight). If no signal clears the bar — e.g. a
    cold start with nothing graded yet — it falls back to the equal-weight
    composite, so enabling this never *worsens* the baseline.
    """

    def score(matrix: pd.DataFrame, _at: datetime) -> pd.Series:
        ics = pd.Series({c: float(ic.get(str(c), 0.0)) for c in matrix.columns}, dtype=float)
        if min_abs_ic > 0:
            ics = ics.where(ics.abs() >= min_abs_ic, 0.0)  # auto-prune decayed signals
        if float(ics.abs().sum()) == 0.0:
            return composite_score(matrix, equal_weights(matrix.columns))
        return composite_score(matrix, ic_weights(ics))

    return score


def _live_universe(symbols: list[str]) -> PointInTimeUniverse:
    # All names treated as members since the distant past — survivorship-correct
    # membership only matters for historical backtests, not live training.
    return PointInTimeUniverse([Constituent(s, s, date(1990, 1, 1), None) for s in symbols])


def _training_schedule(
    store: BitemporalStore, as_of: datetime, *, every: int = 5, max_dates: int = 40
) -> list[datetime]:
    """Sample decision dates from the daily price history in the store."""
    panel = observations_to_price_frame(store.as_of(as_of, dataset=PRICE_DATASET))
    if panel.empty:
        return []
    dates = [d.to_pydatetime() for d in pd.DatetimeIndex(panel.index)]
    return dates[::every][-max_dates:]


def forecast_scorer(
    store: BitemporalStore,
    feature_store: FeatureStore,
    symbols: list[str],
    as_of: datetime,
    *,
    horizon_days: int = 5,
    make_model: Callable[[], object] = make_stacking,
) -> ScoreFn:
    """Train the ensemble on PIT samples and return a P(out-perform) scorer."""
    universe = _live_universe(symbols)
    schedule = _training_schedule(store, as_of)
    model, ts = train_forecaster(
        store,
        feature_store,
        schedule,
        universe=universe,
        make_model=make_model,
        horizon_days=horizon_days,
    )
    feature_names = ts.feature_names

    def score(matrix: pd.DataFrame, at: datetime) -> pd.Series:
        feats = matrix.copy()
        feats[REGIME_FEATURE] = 1.0 if macro_regime(store, at)["risk_on"] else 0.0
        probs = model.predict_proba(feats.reindex(columns=feature_names).to_numpy(dtype=float))
        return pd.Series(probs, index=matrix.index)

    return score


def build_scorer(
    settings: Settings,
    store: BitemporalStore,
    feature_store: FeatureStore,
    symbols: list[str],
    as_of: datetime,
    *,
    ic: dict[str, float] | None = None,
) -> ScoreFn:
    """Pick the scorer from settings — composite (default) or the trained forecaster.

    When ``ic_weighting`` is on and graded ``ic`` is supplied, the composite is
    weighted by measured IC instead of equal weights (the learning loop in action).
    """
    if settings.scorer.strip().lower() == "forecast":
        return forecast_scorer(store, feature_store, symbols, as_of)
    if settings.ic_weighting and ic:
        return ic_weighted_scorer(ic, min_abs_ic=settings.ic_min_abs)
    return composite_scorer()


def forecaster_vs_baseline_auc(
    store: BitemporalStore,
    symbols: list[str],
    as_of: datetime,
    *,
    horizon_days: int = 5,
    forecast_model: Callable[[], object] = make_stacking,
) -> dict[str, float]:
    """Honest gate: purged-CV AUC of the ensemble vs the no-train momentum baseline.

    The forecaster *learns*, so it is scored on purged CV (it can over-fit). The
    baseline does not learn, so its full-sample ranking AUC is already
    leakage-free — and a fair bar the forecaster must clear to earn its place.
    """
    feature_store = FeatureStore(store, default_features())
    ts = build_training_set(
        store,
        feature_store,
        _training_schedule(store, as_of),
        universe=_live_universe(symbols),
        horizon_days=horizon_days,
    )
    y = ts.y.to_numpy()
    baseline = MomentumBaseline(0).predict_proba(ts.X.to_numpy(dtype=float))
    baseline_auc = float(roc_auc_score(y, baseline)) if np.unique(y).size > 1 else float("nan")
    return {
        "n_samples": float(len(ts)),
        "forecast_cv_auc": purged_cv_auc(ts, forecast_model),
        "baseline_auc": baseline_auc,
    }
