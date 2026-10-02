"""
Risk Analysis Tool — Main Streamlit Web App
Market risk today (eight VaR/ES models up to GARCH(1,1)-t, out-of-sample backtests, portfolio risk
decomposition, stress testing and Excel export), with liquidity, credit, concentration, event and
integrated-stress pillars being added phase by phase (RISK_TOOL_PLAN_V3.md).

This file only wires the UI sections together; each lives in ui/ and the calculations in the top-level modules.
The sidebar, data loading and calculations run once per rerun and are shared by every page.
"""

from ui import concentration_page, credit_page, events_page, liquidity_page, market_page, overview_page, trust_page
from ui.analysis import compute
from ui.concentration_layer import compute_concentration
from ui.context import new_context
from ui.credit_layer import compute_credit
from ui.data import load_data
from ui.events_layer import compute_events
from ui.foundations import load_foundations
from ui.liquidity_layer import compute_liquidity
from ui.pages import (CONCENTRATION, CREDIT, EVENTS, LIQUIDITY, MARKET, OVERVIEW, TRUST, render_coming_next,
                      render_navigation)
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
compute_liquidity(ctx)
compute_credit(ctx)
compute_concentration(ctx)
compute_events(ctx)

if page == OVERVIEW:
    overview_page.render(ctx)
elif page == MARKET:
    market_page.render(ctx)
elif page == LIQUIDITY:
    liquidity_page.render(ctx)
elif page == CREDIT:
    credit_page.render(ctx)
elif page == CONCENTRATION:
    concentration_page.render(ctx)
elif page == EVENTS:
    events_page.render(ctx)
elif page == TRUST:
    trust_page.render(ctx)
else:
    render_coming_next(page)
