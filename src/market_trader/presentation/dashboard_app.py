"""Streamlit dashboard (thin rendering shell).

Run it (after `uv sync --extra dashboard`):
    uv run streamlit run src/market_trader/presentation/dashboard_app.py

Decision-support only — this surfaces information; it never places orders. Not
imported by the package or tests (keeps ``streamlit`` out of the engine/CI).
"""

from __future__ import annotations

import streamlit as st

from market_trader.config import get_settings
from market_trader.core.time import utcnow
from market_trader.presentation.dashboard_data import build_dashboard_data
from market_trader.storage.sqlalchemy_store import SqlAlchemyBitemporalStore


def main() -> None:
    st.set_page_config(page_title="market-trader", layout="wide")
    settings = get_settings()
    store = SqlAlchemyBitemporalStore.from_url(settings.database_url)
    data = build_dashboard_data(store, utcnow())

    st.title("market-trader — decision support")
    st.caption(
        f"As of {data.as_of:%Y-%m-%d %H:%M UTC} · mode={settings.execution_mode} · "
        "decision-support only, not financial advice"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Watchlist")
        st.write(", ".join(data.watchlist) or "—")
        st.subheader("Latest prices")
        st.dataframe([{"symbol": k, "close": v} for k, v in sorted(data.latest_prices.items())])
        st.subheader("Macro")
        st.dataframe([{"series": k, "value": v} for k, v in sorted(data.macro.items())])
    with col2:
        st.subheader("Insider (Form 4)")
        st.dataframe(data.recent_insider)
        st.subheader("Congress")
        st.dataframe(data.recent_congress)
        st.subheader("News")
        st.dataframe(data.recent_news)


if __name__ == "__main__":
    main()
