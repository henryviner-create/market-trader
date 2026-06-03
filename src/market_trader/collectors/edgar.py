"""SEC EDGAR insider transactions (Form 4).

Insider Form 4s carry a ~2-day disclosure lag — far more tradable than the
~45-day congressional lag — so they are a higher-value flow signal. ``event_time``
is the transaction date; ``knowledge_time`` is the filing acceptance date. A
purchase (code ``P``) is the informative event; cluster-buys are flagged later.

``normalize`` works on parsed Form 4 records. Live ``fetch`` (EDGAR full-text
search / submissions API + XML parsing) is wired when scheduling collection.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel

from market_trader.collectors.base import Collector
from market_trader.core.schema import Observation
from market_trader.core.time import day_close

FORM4_DATASET = "filing.form4"


class Form4Record(BaseModel):
    issuer_ticker: str
    insider_name: str
    transaction_code: str  # P=purchase, S=sale, A=grant, M=option exercise, ...
    transaction_date: date
    filing_date: date  # acceptance date (~2 days after the transaction)
    insider_title: str | None = None
    shares: float | None = None
    price_per_share: float | None = None


class Form4Collector(Collector):
    source = "edgar"
    parser_version = 1

    def normalize(self, raw: Any) -> list[Observation]:
        records = [r if isinstance(r, Form4Record) else Form4Record.model_validate(r) for r in raw]
        out: list[Observation] = []
        for r in records:
            out.append(
                Observation(
                    source=self.source,
                    dataset=FORM4_DATASET,
                    entity_type="equity",
                    entity_id=r.issuer_ticker.upper(),
                    event_time=day_close(r.transaction_date),
                    knowledge_time=day_close(r.filing_date),
                    value={
                        "transaction_code": r.transaction_code,
                        "is_purchase": r.transaction_code.upper() == "P",
                        "shares": r.shares,
                        "price_per_share": r.price_per_share,
                        "insider_name": r.insider_name,
                        "insider_title": r.insider_title,
                    },
                    metadata={
                        "filing_lag_days": (r.filing_date - r.transaction_date).days,
                        "parser_version": self.parser_version,
                    },
                )
            )
        return out
