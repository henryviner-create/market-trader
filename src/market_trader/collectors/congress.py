"""Congressional trade disclosures.

The signal here is the *disclosure lag*: a trade executed on day T is only
knowable to us on its disclosure date (~45 days later, per the STOCK Act), so we
stamp ``event_time`` = transaction date and ``knowledge_time`` = disclosure date.
A backtest therefore sees the trade only at disclosure, which is what makes the
(already weak, heavily-lagged) signal honestly evaluable.

We also carry the member's *role*: leadership / relevant-committee trades are the
plausibly-informative ones; a backbencher's are mostly noise. The weighting tiers
will use this; here we just record it.

``normalize`` operates on already-parsed records. Live fetching/parsing of raw
House PTRs (PDFs) and the Senate eFD portal is deferred to a parsing adapter;
that is the genuinely hard, low-leverage part and is intentionally out of scope
for this collector's tested core.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close

CONGRESS_DATASET = "disclosure.congress_trade"


class CongressTradeRecord(BaseModel):
    representative: str
    chamber: Literal["house", "senate"]
    ticker: str
    transaction_type: Literal["buy", "sell", "exchange"]
    transaction_date: date
    disclosure_date: date
    amount_low: float | None = None
    amount_high: float | None = None
    party: str | None = None
    role: str | None = None  # "leadership" | "committee:<name>" | "member"
    owner: str | None = None  # self / spouse / child


def is_high_signal_role(role: str | None) -> bool:
    """Leadership or committee membership is plausibly informative; rank others as noise."""
    if not role:
        return False
    r = role.lower()
    return r.startswith("leadership") or r.startswith("committee")


class CongressTradesCollector(Collector):
    source = "congress"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        records = [
            r if isinstance(r, CongressTradeRecord) else CongressTradeRecord.model_validate(r)
            for r in raw
        ]
        out: list[Observation] = []
        for r in records:
            out.append(
                Observation(
                    source=self.source,
                    dataset=CONGRESS_DATASET,
                    entity_type="equity",
                    entity_id=r.ticker.upper(),
                    ref=f"{r.representative}|{r.transaction_type}",
                    event_time=day_close(r.transaction_date),
                    knowledge_time=day_close(r.disclosure_date),
                    value={
                        "transaction_type": r.transaction_type,
                        "amount_low": r.amount_low,
                        "amount_high": r.amount_high,
                        "representative": r.representative,
                        "chamber": r.chamber,
                        "party": r.party,
                    },
                    metadata={
                        "role": r.role,
                        "owner": r.owner,
                        "high_signal_role": is_high_signal_role(r.role),
                        "disclosure_lag_days": (r.disclosure_date - r.transaction_date).days,
                        "parser_version": self.parser_version,
                    },
                )
            )
        return out
