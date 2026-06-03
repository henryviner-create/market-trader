"""Presentation tier: dashboard, daily briefing, alerts.

Only the *pure* data layer is exported here. The Streamlit app
(``dashboard_app``) imports ``streamlit`` (an optional extra) and is launched
directly, never imported by the package or tests.
"""

from market_trader.presentation.dashboard_data import DashboardData, build_dashboard_data

__all__ = ["DashboardData", "build_dashboard_data"]
