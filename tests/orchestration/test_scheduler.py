"""Cadence-tier scheduling: which jobs are due."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from market_trader.orchestration import Cadence, Job, JobRegistry


def _noop() -> None:
    return None


def test_due_jobs_respect_their_interval() -> None:
    reg = JobRegistry()
    reg.register(Job("daily", Cadence.DAILY, _noop))
    reg.register(Job("hourly", Cadence.HOURLY, _noop))
    reg.register(Job("stream", Cadence.REALTIME, _noop))

    now = datetime(2023, 1, 2, 12, 0, tzinfo=UTC)
    last_run = {"daily": now - timedelta(hours=25), "hourly": now - timedelta(minutes=30)}
    due = {j.name for j in reg.due(now, last_run)}

    assert "daily" in due  # 25h elapsed >= 1 day
    assert "hourly" not in due  # 30m < 1h
    assert "stream" not in due  # realtime is stream-driven, not interval-scheduled


def test_never_run_interval_job_is_due() -> None:
    reg = JobRegistry()
    reg.register(Job("weekly_retrain", Cadence.WEEKLY, _noop))
    due = reg.due(datetime(2023, 1, 1, tzinfo=UTC), {})
    assert [j.name for j in due] == ["weekly_retrain"]
