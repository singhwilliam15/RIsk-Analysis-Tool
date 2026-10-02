"""
Risk Analysis Tool — Main Streamlit Web App
Market risk today (eight VaR/ES models up to GARCH(1,1)-t, out-of-sample backtests, portfolio risk
decomposition, stress testing and Excel export), with liquidity, credit, concentration, event and
integrated-stress pillars being added phase by phase (RISK_TOOL_PLAN_V3.md).

This file only wires the UI sections together; each lives in ui/ and the calculations in the top-level modules.
The sidebar, data loading and calculations run once per rerun and are shared by every page.
"""

from ui import market_page, overview_page, trust_page
from ui.analysis import compute
from ui.context import new_context
from ui.data import load_data
from ui.foundations import load_foundations
from ui.pages import MARKET, OVERVIEW, TRUST, render_coming_next, render_navigation
from ui.sidebar import render_sidebar
from ui.styles import setup_page
from ui.trust_layer import compute_trust

setup_page()
page = render_navigation()
ctx = new_context()
render_sidebar(ctx)
load_data(ctx)  # stops the script with an error message if the data cannot be used
compute(ctx)
load_foundations(ctx)
compute_trust(ctx)

if page == OVERVIEW:
    overview_page.render(ctx)
elif page == MARKET:
    market_page.render(ctx)
elif page == TRUST:
    trust_page.render(ctx)
else:
    render_coming_next(page)
