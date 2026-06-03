"""Signal tier: features grouped into decorrelated families.

A :class:`Feature` computes a cross-sectional value per symbol *from the
point-in-time store as of a knowledge time* — so the exact same code runs in
backtest and live, eliminating train/serve skew at the source.
"""

from market_trader.features.base import (
    Feature,
    FeatureStore,
    cross_sectional_zscore,
    default_features,
)
from market_trader.features.flow import CongressLeadershipBuys, InsiderNetBuys
from market_trader.features.regime import macro_regime
from market_trader.features.technical import MeanReversion, Momentum, Volatility

__all__ = [
    "CongressLeadershipBuys",
    "Feature",
    "FeatureStore",
    "InsiderNetBuys",
    "MeanReversion",
    "Momentum",
    "Volatility",
    "cross_sectional_zscore",
    "default_features",
    "macro_regime",
]
