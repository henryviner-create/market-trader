"""Base models, the stacking ensemble, and calibration — all via scikit-learn.

Factories return fresh estimators (so they can be refit per CV fold). The thin
:class:`Forecaster` wrapper exposes ``fit`` / ``predict_proba`` returning P(class=1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logistic(c: float = 1.0) -> Pipeline:
    """L2-regularised logistic regression with median imputation + scaling."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000)),  # L2 is the default
        ]
    )


def make_gbt() -> HistGradientBoostingClassifier:
    """Gradient-boosted trees (handles NaNs natively). Shallow + regularised."""
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0
    )


def make_stacking() -> StackingClassifier:
    """Regime-aware stacking: logistic + GBT base learners, logistic meta-learner.

    Regime conditioning enters as a feature column (``REGIME_FEATURE``), so the
    learners can split on regime without bespoke per-regime training.
    """
    return StackingClassifier(
        estimators=[("logit", make_logistic()), ("gbt", make_gbt())],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5,
        passthrough=False,
    )


def make_calibrated(
    estimator: Any, *, method: str = "isotonic", cv: int = 3
) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator, method=method, cv=cv)


class Forecaster:
    """Wraps any sklearn classifier; ``predict_proba`` returns P(class=1)."""

    def __init__(self, estimator: Any, name: str) -> None:
        self.estimator = estimator
        self.name = name

    def fit(self, X: Any, y: Any) -> Forecaster:
        self.estimator.fit(np.asarray(X, dtype=float), np.asarray(y))
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        proba = self.estimator.predict_proba(np.asarray(X, dtype=float))
        return np.asarray(proba[:, 1], dtype=float)


class MomentumBaseline:
    """No-train cross-sectional baseline: rank by one feature column into [0, 1]."""

    name = "momentum_baseline"

    def __init__(self, feature_index: int = 0) -> None:
        self.feature_index = feature_index

    def fit(self, X: Any, y: Any) -> MomentumBaseline:
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        col = np.asarray(X, dtype=float)[:, self.feature_index]
        n = col.size
        if n == 0:
            return np.array([], dtype=float)
        filled = np.nan_to_num(
            col, nan=float(np.nanmin(col)) if np.isfinite(np.nanmin(col)) else 0.0
        )
        ranks = np.argsort(np.argsort(filled))
        return (ranks + 0.5) / n
