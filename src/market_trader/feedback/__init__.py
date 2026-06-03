"""Feedback loop — where accuracy actually improves, through iteration.

Log every prediction with its full input snapshot -> score outcomes -> monitor
drift (live vs backtest) -> prune decayed signals -> retrain. Predictions are
stored as bitemporal observations, so replaying them is point-in-time correct.
"""

from market_trader.feedback.drift import DriftMonitor
from market_trader.feedback.prediction_log import (
    PREDICTION_DATASET,
    PredictionRecord,
    load_predictions,
    log_predictions,
)
from market_trader.feedback.pruning import prune_signals_by_ic
from market_trader.feedback.scoring import score_predictions

__all__ = [
    "PREDICTION_DATASET",
    "DriftMonitor",
    "PredictionRecord",
    "load_predictions",
    "log_predictions",
    "prune_signals_by_ic",
    "score_predictions",
]
