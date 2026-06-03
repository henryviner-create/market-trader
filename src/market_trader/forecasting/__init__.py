"""Forecasting ensemble.

Cross-sectional classifiers that emit **probabilities** (never point predictions)
of a name out-performing over the horizon, trained on point-in-time features with
forward-return labels. Base models (regularised logistic + gradient-boosted trees)
feed a regime-aware stacking meta-learner; calibration is first-class. No deep
learning until a simple baseline is beaten.
"""

from market_trader.forecasting.dataset import REGIME_FEATURE, TrainingSet, build_training_set
from market_trader.forecasting.models import (
    Forecaster,
    MomentumBaseline,
    make_calibrated,
    make_gbt,
    make_logistic,
    make_stacking,
)
from market_trader.forecasting.pipeline import (
    CalibrationReport,
    ForecastStrategy,
    evaluate_predictions,
    purged_cv_auc,
    time_split,
    train_forecaster,
)

__all__ = [
    "REGIME_FEATURE",
    "CalibrationReport",
    "ForecastStrategy",
    "Forecaster",
    "MomentumBaseline",
    "TrainingSet",
    "build_training_set",
    "evaluate_predictions",
    "make_calibrated",
    "make_gbt",
    "make_logistic",
    "make_stacking",
    "purged_cv_auc",
    "time_split",
    "train_forecaster",
]
