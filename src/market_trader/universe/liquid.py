"""Tradable universes for live selection.

``MEGACAP_WATCHLIST`` is the original 8-name set (kept for tests / a deliberately
narrow run). ``LIQUID_LARGE_CAP`` is a broad, sector-diversified set of liquid US
large caps — the default, so the ranker can surface opportunities across the whole
market instead of recycling the same few megacaps.

This is a *current* tradable list, not a survivorship-correct history: that lives
in :mod:`market_trader.universe.pit_universe` and matters for backtests, not for
deciding what is tradable today. An unknown/dead ticker simply returns no bars and
drops out of the ranking, so the list degrades gracefully.
"""

from __future__ import annotations

MEGACAP_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]

# ~110 liquid US large caps spread across all 11 GICS sectors. Deliberately broad
# so the cross-sectional ranking has real breadth to choose from.
LIQUID_LARGE_CAP = [
    # Information Technology
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "AMD",
    "INTC",
    "CSCO",
    "ACN",
    "IBM",
    "QCOM",
    "TXN",
    "NOW",
    "INTU",
    "AMAT",
    "MU",
    "LRCX",
    "ADI",
    # Communication Services
    "GOOGL",
    "META",
    "NFLX",
    "DIS",
    "CMCSA",
    "T",
    "VZ",
    "TMUS",
    "CHTR",
    # Consumer Discretionary
    "AMZN",
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "LOW",
    "SBUX",
    "BKNG",
    "TJX",
    "GM",
    "F",
    # Consumer Staples
    "PG",
    "KO",
    "PEP",
    "COST",
    "WMT",
    "PM",
    "MO",
    "CL",
    "MDLZ",
    "KMB",
    # Health Care
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "MRK",
    "ABBV",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "CVS",
    "MDT",
    "GILD",
    "ISRG",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "AXP",
    "BLK",
    "SPGI",
    "SCHW",
    "USB",
    "PNC",
    # Industrials
    "CAT",
    "HON",
    "UPS",
    "BA",
    "GE",
    "RTX",
    "UNP",
    "LMT",
    "DE",
    "MMM",
    "EMR",
    "CSX",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "EOG",
    "MPC",
    "PSX",
    # Materials
    "LIN",
    "APD",
    "SHW",
    "FCX",
    "NEM",
    "DOW",
    # Utilities
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    # Real Estate
    "AMT",
    "PLD",
    "EQIX",
    "SPG",
    "O",
]

# Liquid European/UK large caps trading as ADRs on US exchanges (USD, US hours), so
# they slot into the daily cycle exactly like US names. NOTE: direct European-
# exchange shares (LVMH.PA, SAP.DE, ...) are NOT tradable on Alpaca — ADRs are the
# realistic path, and they carry implicit EUR/GBP/CHF currency exposure.
EUROPEAN_ADRS = [
    # Information Technology / Semiconductors
    "ASML",  # Netherlands
    "SAP",  # Germany
    "STM",  # France/Italy
    "ERIC",  # Sweden
    "NOK",  # Finland
    # Health Care
    "NVO",  # Denmark (Novo Nordisk)
    "AZN",  # UK (AstraZeneca)
    "NVS",  # Switzerland (Novartis)
    "GSK",  # UK
    "SNY",  # France (Sanofi)
    # Consumer Staples
    "UL",  # UK (Unilever)
    "DEO",  # UK (Diageo)
    "BTI",  # UK (British American Tobacco)
    "BUD",  # Belgium (AB InBev)
    # Energy
    "SHEL",  # UK (Shell)
    "BP",  # UK
    "TTE",  # France (TotalEnergies)
    "E",  # Italy (Eni)
    "EQNR",  # Norway (Equinor)
    # Financials
    "UBS",  # Switzerland
    "HSBC",  # UK
    "ING",  # Netherlands
    "DB",  # Germany (Deutsche Bank)
    # Materials / Industrials / Comm / Consumer
    "RIO",  # UK (Rio Tinto)
    "STLA",  # Netherlands (Stellantis)
    "SPOT",  # Sweden (Spotify)
    "PHG",  # Netherlands (Philips)
]

# US-listed ETFs for broad European-market exposure (a basket, not single names).
EUROPE_ETFS = ["VGK", "EZU", "IEUR", "FEZ"]

# US liquid large caps + European ADRs + Europe ETFs: broad *global* breadth for
# the cross-sectional ranker while staying entirely Alpaca-tradable (US-listed).
GLOBAL_LIQUID = LIQUID_LARGE_CAP + EUROPEAN_ADRS + EUROPE_ETFS


def resolve_universe(name: str) -> list[str]:
    """Map a universe setting to a symbol list.

    ``"liquid"``/``"broad"`` -> the broad US set (default); ``"global"``/``"world"``
    -> US + European ADRs + Europe ETFs; ``"watchlist"``/``"megacap"`` -> the 8-name
    set; a comma-separated string -> a custom explicit list.
    """
    key = (name or "").strip().lower()
    if "," in name:
        return [s.strip().upper() for s in name.split(",") if s.strip()]
    if key in ("watchlist", "megacap", "mega"):
        return list(MEGACAP_WATCHLIST)
    if key in ("global", "world", "liquid_global"):
        return list(GLOBAL_LIQUID)
    return list(LIQUID_LARGE_CAP)  # "liquid"/"broad"/anything else -> broad default
