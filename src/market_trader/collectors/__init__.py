"""Collection tier.

Collectors split I/O from logic: ``fetch`` does network I/O and returns raw
payloads; ``normalize`` is pure (raw -> canonical :class:`Observation`s with
correct event/knowledge times) and is what tests exercise offline. The
:class:`IngestionGateway` is the normalisation gateway: it assigns deterministic
ids, dedups, runs data-quality checks, and idempotently upserts into the store.
"""

from market_trader.collectors.base import Collector
from market_trader.collectors.congress import CongressTradesCollector
from market_trader.collectors.edgar import Form4Collector
from market_trader.collectors.fred import FredSeriesCollector
from market_trader.collectors.gateway import IngestionGateway, IngestSummary
from market_trader.collectors.gdelt import GdeltNewsCollector
from market_trader.collectors.prices import PriceCollector
from market_trader.collectors.quality import (
    DataQualityError,
    QualityIssue,
    QualityReport,
    validate_observations,
)

__all__ = [
    "Collector",
    "CongressTradesCollector",
    "DataQualityError",
    "Form4Collector",
    "FredSeriesCollector",
    "GdeltNewsCollector",
    "IngestSummary",
    "IngestionGateway",
    "PriceCollector",
    "QualityIssue",
    "QualityReport",
    "validate_observations",
]
