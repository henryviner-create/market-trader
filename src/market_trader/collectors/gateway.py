"""The normalisation / ingestion gateway.

One choke point through which every collected observation passes before storage:
it assigns a deterministic id, dedups within the batch, runs data-quality checks,
and idempotently upserts into the bitemporal store. Re-running a collector is a
no-op; it never duplicates or corrupts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from uuid import UUID

from market_trader.collectors.quality import DataQualityError, QualityReport, validate_observations
from market_trader.core.identity import with_deterministic_id
from market_trader.core.schema import Observation
from market_trader.observability import get_logger
from market_trader.storage.bitemporal import BitemporalStore

_log = get_logger("ingest")


@dataclass(frozen=True)
class IngestSummary:
    received: int
    unique: int
    newly_added: int
    quality_ok: bool
    quality_issues: int


class IngestionGateway:
    def __init__(
        self, store: BitemporalStore, *, validate: bool = True, strict: bool = False
    ) -> None:
        self._store = store
        self._validate = validate
        self._strict = strict

    def ingest(self, observations: Iterable[Observation]) -> IngestSummary:
        received = 0
        by_id: dict[UUID, Observation] = {}
        for o in observations:
            received += 1
            stamped = with_deterministic_id(o)
            by_id[stamped.observation_id] = stamped
        unique = list(by_id.values())

        report: QualityReport | None = None
        if self._validate:
            report = validate_observations(unique)
            if not report.ok:
                if self._strict:
                    raise DataQualityError(report)
                for issue in report.errors:
                    _log.warning(
                        "data_quality_issue",
                        check=issue.check,
                        message=issue.message,
                        entity_id=issue.entity_id,
                    )

        before = self._store.count()
        self._store.upsert_many(unique)
        after = self._store.count()

        summary = IngestSummary(
            received=received,
            unique=len(unique),
            newly_added=after - before,
            quality_ok=report.ok if report is not None else True,
            quality_issues=len(report) if report is not None else 0,
        )
        _log.info("ingested", source=unique[0].source if unique else None, **asdict(summary))
        return summary
