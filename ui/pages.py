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

# Pages not built yet: the roadmap phase that builds each, and what it will contain
BUILT = (OVERVIEW, MARKET, TRUST)
COMING_NEXT = {
    LIQUIDITY: (2, "Days to liquidate at a participation rate, the SEBI/AMFI-style liquidity stress test, stressed "
                   "volume, spread and market-impact cost, liquidity-adjusted VaR, Amihud illiquidity and circuit-lock risk."),
    CREDIT: (3, "Merton distance to default and model-implied PD, Altman Z and Z'', credit ratios with red flags, "
                "rating actions, and the portfolio's credit-implied expected loss."),
    CONCENTRATION: (4, "Factor betas from Indian and US Fama-French data, factor and sector risk contributions, HHI, "
                       "PCA and the effective number of independent bets, and crisis correlations."),
    EVENTS: (5, "Promoter pledges and margin-call triggers, ASM/GSM surveillance, F&O ban, rating downgrades, "
                "auditor events, an event-risk tier and a jump-risk overlay on ES."),
    INTEGRATED: (6, "One stress scenario hitting every pillar together, against the sum of the separate pillars, "
                    "and a reverse stress test."),
    DECISIONS: (6, "Risk-change explanation, the best risk-reducing trades, limits with traffic lights and a "
                   "one-page CRO memo."),
}


def render_navigation() -> str:
    """The page selector; returns the selected page."""
    return st.radio("Page", PAGES, horizontal=True, key=PAGE_KEY, label_visibility="collapsed")


def render_coming_next(page: str):
    phase, contents = COMING_NEXT[page]
    st.subheader(page)
    st.info(f"**Coming next** (roadmap Phase {phase}). {contents}")
    st.caption("The plan for every page is in RISK_TOOL_PLAN_V3.md; the formulas will be in docs/methodology.md.")
