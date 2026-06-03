"""Cadence tiers and which jobs are due.

Tiers: realtime/event (driven elsewhere) and interval tiers (minutely, hourly,
daily, weekly) resolved here. Jobs are idempotent; ``catchup`` decides whether a
missed run is made up (run once now) or skipped — an explicit per-job policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum


class Cadence(StrEnum):
    REALTIME = "realtime"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    EVENT = "event"


_INTERVALS: dict[Cadence, timedelta] = {
    Cadence.MINUTELY: timedelta(minutes=1),
    Cadence.HOURLY: timedelta(hours=1),
    Cadence.DAILY: timedelta(days=1),
    Cadence.WEEKLY: timedelta(weeks=1),
}


@dataclass
class Job:
    name: str
    cadence: Cadence
    fn: Callable[[], None]
    catchup: bool = False  # if True, a missed run is made up; else skipped


@dataclass
class JobRegistry:
    _jobs: dict[str, Job] = field(default_factory=dict)

    def register(self, job: Job) -> None:
        self._jobs[job.name] = job

    def jobs(self) -> list[Job]:
        return list(self._jobs.values())

    def due(self, now: datetime, last_run: dict[str, datetime]) -> list[Job]:
        """Interval-tier jobs whose interval has elapsed since their last run."""
        due_jobs: list[Job] = []
        for job in self._jobs.values():
            interval = _INTERVALS.get(job.cadence)
            if interval is None:  # realtime/event handled by streams, not here
                continue
            last = last_run.get(job.name)
            if last is None or now - last >= interval:
                due_jobs.append(job)
        return due_jobs
