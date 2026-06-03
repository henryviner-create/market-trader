"""Data-quality assertions on ingestion.

Broken feeds feed garbage into every downstream tier, which is worse than missing
data — so observations are checked at the gateway. We keep this in-house and
targeted rather than pulling a heavy schema framework: records are already
pydantic-validated for type/timezone correctness, so the remaining checks are
cross-field sanity (finite/positive values, disclosure lag, in-batch duplicates).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from market_trader.core.schema import Observation

# Value keys we expect to be numeric where present.
NUMERIC_VALUE_KEYS = ("close", "open", "high", "low", "value", "amount_low", "amount_high")
# Of those, the ones that must be strictly positive (prices).
POSITIVE_VALUE_KEYS = ("close", "open", "high", "low")
# Datasets where knowledge_time must be >= event_time (disclosure after the fact).
DISCLOSURE_DATASET_PREFIXES = ("disclosure.", "filing.")


@dataclass(frozen=True)
class QualityIssue:
    level: str  # "error" | "warning"
    check: str
    message: str
    entity_id: str | None = None


@dataclass
class QualityReport:
    issues: list[QualityIssue] = field(default_factory=list)
    n_checked: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.level == "error"]

    def __len__(self) -> int:
        return len(self.issues)


class DataQualityError(Exception):
    def __init__(self, report: QualityReport) -> None:
        self.report = report
        super().__init__(f"data quality failed: {len(report.errors)} error(s)")


def validate_observations(observations: Sequence[Observation]) -> QualityReport:
    issues: list[QualityIssue] = []
    seen: set[tuple[object, ...]] = set()
    obs = list(observations)

    for o in obs:
        for key in NUMERIC_VALUE_KEYS:
            v = o.value.get(key)
            if v is None or not isinstance(v, (int, float)):
                continue
            if math.isnan(v) or math.isinf(v):
                issues.append(
                    QualityIssue("error", "finite", f"{o.dataset}:{key} is NaN/inf", o.entity_id)
                )
            elif key in POSITIVE_VALUE_KEYS and v <= 0:
                issues.append(
                    QualityIssue("error", "positive", f"{o.dataset}:{key}={v} <= 0", o.entity_id)
                )

        if o.dataset.startswith(DISCLOSURE_DATASET_PREFIXES) and o.knowledge_time < o.event_time:
            issues.append(
                QualityIssue("error", "lag", f"{o.dataset}: knowledge precedes event", o.entity_id)
            )

        identity = (o.source, o.dataset, o.entity_id, o.event_time, o.knowledge_time, o.revision)
        if identity in seen:
            issues.append(
                QualityIssue(
                    "warning", "duplicate", f"duplicate identity for {o.entity_id}", o.entity_id
                )
            )
        seen.add(identity)

    return QualityReport(issues=issues, n_checked=len(obs))
