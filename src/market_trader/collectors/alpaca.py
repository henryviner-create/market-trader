"""Alpaca market-data client (daily bars) — stdlib urllib, offline-testable.

Fetches OHLCV bars from Alpaca's data API and returns records in the shape
:class:`market_trader.collectors.prices.PriceBar` expects, so they flow through the
same normalisation + point-in-time machinery. Keys are env-only (paper).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

DATA_BASE_URL = "https://data.alpaca.markets"

# (url, headers) -> (status_code, json_payload)
DataTransport = Callable[[str, dict[str, str]], tuple[int, dict[str, Any]]]


class AlpacaDataError(RuntimeError):
    pass


def _urllib_get(url: str, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raise AlpacaDataError(
            f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}"
        ) from exc
    except Exception as exc:
        raise AlpacaDataError(f"request failed: {exc}") from exc


class AlpacaDataClient:
    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = DATA_BASE_URL,
        transport: DataTransport | None = None,
    ) -> None:
        if not (key_id and secret_key):
            raise AlpacaDataError("Alpaca data keys are required")
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}
        self._base = base_url.rstrip("/")
        self._get = transport or _urllib_get

    def fetch_daily_bars(
        self,
        symbols: Sequence[str],
        *,
        start: date,
        end: date,
        adjustment: str = "all",
        feed: str = "iex",
    ) -> list[dict[str, Any]]:
        """Daily OHLCV bars per symbol, as PriceBar-shaped records.

        ``feed`` defaults to ``iex`` (the free market-data feed). The paid
        consolidated ``sip`` feed 403s on free plans ("subscription does not
        permit querying recent SIP data"), so it must be opted into explicitly.
        """
        query = urllib.parse.urlencode(
            {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": adjustment,
                "feed": feed,
                "limit": 10000,
            }
        )
        status, payload = self._get(f"{self._base}/v2/stocks/bars?{query}", self._headers)
        if status >= 300:
            raise AlpacaDataError(f"HTTP {status}: {payload}")

        records: list[dict[str, Any]] = []
        for symbol, bars in (payload.get("bars") or {}).items():
            for bar in bars:
                records.append(
                    {
                        "date": str(bar["t"])[:10],
                        "symbol": symbol,
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                    }
                )
        return records

    def fetch_intraday_bars(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        feed: str = "iex",
        adjustment: str = "raw",
    ) -> list[dict[str, Any]]:
        """Intraday OHLCV bars per symbol, full minute ``timestamp`` preserved.

        Unlike :meth:`fetch_daily_bars` (which truncates to a date), the live loop
        needs the bar's exact time, so each record keeps ``timestamp``. Pages are
        followed via ``next_page_token`` so a wide window can't silently truncate.
        """
        base = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "adjustment": adjustment,
            "feed": feed,
            "limit": 10000,
        }
        records: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params = dict(base)
            if page_token:
                params["page_token"] = page_token
            query = urllib.parse.urlencode(params)
            status, payload = self._get(f"{self._base}/v2/stocks/bars?{query}", self._headers)
            if status >= 300:
                raise AlpacaDataError(f"HTTP {status}: {payload}")
            for symbol, bars in (payload.get("bars") or {}).items():
                for bar in bars:
                    records.append(
                        {
                            "timestamp": bar["t"],
                            "symbol": symbol,
                            "open": bar.get("o"),
                            "high": bar.get("h"),
                            "low": bar.get("l"),
                            "close": bar.get("c"),
                            "volume": bar.get("v"),
                        }
                    )
            page_token = payload.get("next_page_token")
            if not page_token:
                return records
