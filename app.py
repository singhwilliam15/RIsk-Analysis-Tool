"""
Risk Analysis Tool — Main Streamlit Web App
Five risk pillars (market, liquidity, credit, concentration and factors, event and governance), a trust layer
that gives every headline number a range and an A–D grade, a linked stress engine across the pillars, and a
decision layer (risk-change explanation, risk-reducing trades, limits, CRO dashboard and memo).

This file only wires the UI sections together; each lives in ui/ and the calculations in the top-level modules.
The sidebar, data loading and calculations run once per rerun and are shared by every page.
"""

from ui import (concentration_page, credit_page, decisions_page, events_page, integration_page, liquidity_page,
                market_page, overview_page, trust_page)
from ui.analysis import compute
from ui.concentration_layer import compute_concentration
from ui.context import new_context
from ui.credit_layer import compute_credit
from ui.data import load_data
from ui.events_layer import compute_events
from ui.foundations import load_foundations
from ui.integration_layer import compute_decisions, compute_integration
from ui.liquidity_layer import compute_liquidity
from ui.pages import (CONCENTRATION, CREDIT, DECISIONS, EVENTS, INTEGRATED, LIQUIDITY, MARKET, OVERVIEW, TRUST,
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
compute_integration(ctx)
compute_decisions(ctx)

PAGE_RENDERERS = {
    OVERVIEW: overview_page.render, MARKET: market_page.render, LIQUIDITY: liquidity_page.render,
    CREDIT: credit_page.render, CONCENTRATION: concentration_page.render, EVENTS: events_page.render,
    INTEGRATED: integration_page.render, DECISIONS: decisions_page.render, TRUST: trust_page.render,
}
PAGE_RENDERERS[page](ctx)
