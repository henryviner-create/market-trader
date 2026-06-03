"""Prediction logging.

Every prediction is persisted as a bitemporal observation with its **full input
snapshot** (the feature vector), so it can be replayed exactly and scored once the
outcome is known. Logging is idempotent (deterministic id by symbol/time/model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.storage.bitemporal import BitemporalStore

PREDICTION_DATASET = "forecast.prediction"


@dataclass
class PredictionRecord:
    as_of: datetime
    symbol: str
    probability: float
    horizon_days: int
    model_version: str
    features: dict[str, float] = field(default_factory=dict)


def _to_observation(rec: PredictionRecord) -> Observation:
    return with_deterministic_id(
        Observation(
            source="forecast",
            dataset=PREDICTION_DATASET,
            entity_type="equity",
            entity_id=rec.symbol,
            ref=rec.model_version,
            event_time=rec.as_of,
            knowledge_time=rec.as_of,
            value={
                "probability": rec.probability,
                "horizon_days": rec.horizon_days,
                "model_version": rec.model_version,
            },
            metadata={"features": dict(rec.features)},
        )
    )


def _from_observation(o: Observation) -> PredictionRecord:
    return PredictionRecord(
        as_of=o.event_time,
        symbol=o.entity_id,
        probability=float(o.value["probability"]),
        horizon_days=int(o.value["horizon_days"]),
        model_version=str(o.value["model_version"]),
        features={k: float(v) for k, v in o.metadata.get("features", {}).items()},
    )


def log_predictions(store: BitemporalStore, records: list[PredictionRecord]) -> int:
    observations = [_to_observation(r) for r in records]
    store.upsert_many(observations)
    return len(observations)


def load_predictions(
    store: BitemporalStore, as_of: datetime, *, model_version: str | None = None
) -> list[PredictionRecord]:
    records = [_from_observation(o) for o in store.as_of(as_of, dataset=PREDICTION_DATASET)]
    if model_version is not None:
        records = [r for r in records if r.model_version == model_version]
    return records
