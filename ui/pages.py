"""
Page list and navigation.

Navigation is one keyed radio at the top of the main area rather than st.navigation. Every page shares
one run of the sidebar, data loading and cached calculations, so all pages show the same inputs and
results; and Streamlit's AppTest can switch between radio options, but cannot switch to callable
st.navigation pages, which would leave the pages untested.
"""

import streamlit as st

OVERVIEW = "Overview"
MARKET = "Market"
LIQUIDITY = "Liquidity"
CREDIT = "Credit"
CONCENTRATION = "Concentration & Factors"
EVENTS = "Event & Governance"
INTEGRATED = "Integrated Stress"
DECISIONS = "Decisions"
TRUST = "Trust"
PAGES = (OVERVIEW, MARKET, LIQUIDITY, CREDIT, CONCENTRATION, EVENTS, INTEGRATED, DECISIONS, TRUST)
PAGE_KEY = "page"


def render_navigation() -> str:
    """The page selector; returns the selected page."""
    return st.radio("Page", PAGES, horizontal=True, key=PAGE_KEY, label_visibility="collapsed")
