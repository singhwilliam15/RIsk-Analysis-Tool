"""Sidebar inputs: mode, ticker or holdings, lookback, position size, confidence, horizon."""

import pandas as pd
import streamlit as st
from liquidity import AMFI_EXCLUDE, AMFI_PARTICIPATION, BANGIA_K, DEFAULT_PARTICIPATION, IMPACT_Y
from portfolio import BUY_AND_HOLD, ENTRY_MODES, ENTRY_SHARES, ENTRY_VALUE, ENTRY_WEIGHT, REBALANCE_DAILY
from ui.context import export
from ui.formatting import pct_label

DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "ASIANPAINT.NS", "BRITANNIA.NS"]
DEFAULT_AMOUNTS = {
    ENTRY_WEIGHT: [30.0, 25.0, 20.0, 15.0, 10.0],
    ENTRY_SHARES: [100.0, 150.0, 50.0, 100.0, 20.0],
    ENTRY_VALUE: [300000.0, 250000.0, 200000.0, 150000.0, 100000.0],
}
HOLDINGS_KEYS = {ENTRY_WEIGHT: "holdings", ENTRY_SHARES: "holdings_shares", ENTRY_VALUE: "holdings_value"}


def render_sidebar(ctx):
    """Sidebar inputs: mode, ticker or holdings, lookback, position size, confidence, horizon."""
    st.sidebar.header("⚙️ Risk Parameters & Input")

    # Quick Select Pills
    quick_tickers = {
        "Asian Paints (NSE)": "ASIANPAINT.NS",
        "Britannia (NSE)": "BRITANNIA.NS",
        "Reliance (NSE)": "RELIANCE.NS",
        "HDFC Bank (NSE)": "HDFCBANK.NS",
        "TCS (NSE)": "TCS.NS",
        "Apple (US)": "AAPL",
        "Microsoft (US)": "MSFT",
        "Tesla (US)": "TSLA"
    }

    analysis_mode = st.sidebar.radio("Analysis Mode", ["Single Stock", "Portfolio"], horizontal=True)
    is_portfolio = analysis_mode == "Portfolio"
    entry_mode = ENTRY_WEIGHT  # a single stock is always sized by the investment amount

    if not is_portfolio:
        selected_preset = st.sidebar.selectbox("Quick Preset Ticker", options=["Custom Ticker"] + list(quick_tickers.keys()))

        if selected_preset != "Custom Ticker":
            default_ticker = quick_tickers[selected_preset]
        else:
            default_ticker = "ASIANPAINT.NS"

        ticker_input = st.sidebar.text_input("Enter Yahoo Finance Ticker Symbol", value=default_ticker, help="Examples: ASIANPAINT.NS, BRITANNIA.NS, RELIANCE.NS, AAPL, MSFT").strip().upper()
    else:
        entry_mode = st.sidebar.selectbox("Holdings entered as", ENTRY_MODES,
                                          help="Weight: any units, scaled to 100% of the investment amount. Shares: number of "
                                               "shares held. Value: money held in each stock. Shares and values are converted "
                                               "at the latest close, and the investment amount becomes their total.")
        st.sidebar.caption({
            ENTRY_WEIGHT: "Holdings and weights (any units; they are scaled to 100%).",
            ENTRY_SHARES: "Holdings and the number of shares held.",
            ENTRY_VALUE: "Holdings and the money held in each, in the trading currency.",
        }[entry_mode] + " Add or delete rows in the table. All holdings must trade in the same currency.")
        holdings_input = st.sidebar.data_editor(
            pd.DataFrame({"Ticker": DEFAULT_TICKERS, entry_mode: DEFAULT_AMOUNTS[entry_mode]}),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            key=HOLDINGS_KEYS[entry_mode],  # one table per entry mode, so switching modes keeps each table's edits
        )
        rebalance_mode = st.sidebar.radio("Rebalancing", [REBALANCE_DAILY, BUY_AND_HOLD],
                                          help="Daily: weights reset to the targets every day. Buy-and-hold: shares are bought once and weights drift with prices.")
        st.sidebar.caption("Long-only: weights must be positive. Short positions would need borrow costs, margin and "
                           "a gross/net exposure definition that this tool does not model.")

    period_input = st.sidebar.selectbox("Historical Lookback Window", options=["1y", "2y", "5y", "max"], index=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 Portfolio Settings")

    if is_portfolio and entry_mode != ENTRY_WEIGHT:
        investment_amount = None  # the total value of the holdings, set once prices are loaded
        st.sidebar.caption(f"Investment amount: the total value of the holdings, from the {entry_mode.lower()} entered.")
    else:
        investment_amount = st.sidebar.number_input("Portfolio Investment Amount", min_value=1000.0, max_value=1000000000.0, value=1000000.0, step=50000.0, format="%.2f")

    confidence_level = st.sidebar.select_slider("Confidence Level (1 - α)", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
                                                format_func=pct_label, help="97.5% is the Basel FRTB Expected Shortfall level.")

    holding_period = st.sidebar.selectbox("Holding Period (Days)", options=[1, 5, 10, 21, 30], index=0, help="VaR scaled by sqrt(days)")

    num_sims = st.sidebar.selectbox("Monte Carlo Simulations", options=[1000, 2500, 5000, 10000], index=2)

    with st.sidebar.expander("💧 Liquidity assumptions"):
        st.caption("Assumptions, not data: change them to see how much the liquidity figures depend on them.")
        participation = st.number_input("Participation rate, % of daily volume", 1.0, 100.0,
                                        DEFAULT_PARTICIPATION * 100, 5.0, key="liq_participation",
                                        help="How much of a day's volume you could sell without dominating trading.") / 100
        amfi_participation = st.number_input("AMFI test: % of 3-month volume", 1.0, 100.0, AMFI_PARTICIPATION * 100, 1.0,
                                             key="liq_amfi_participation") / 100
        amfi_exclude = st.number_input("AMFI test: least liquid % excluded", 0.0, 90.0, AMFI_EXCLUDE * 100, 5.0,
                                       key="liq_amfi_exclude") / 100
        bangia_k = st.number_input("Spread volatility multiplier k (Bangia)", 0.0, 10.0, BANGIA_K, 0.5, key="liq_bangia_k")
        impact_y = st.number_input("Market-impact constant Y (square-root law)", 0.0, 5.0, IMPACT_Y, 0.1, key="liq_impact_y")
        user_spread_pct = st.number_input("Known bid-ask spread, % (0 = estimate from high/low)", 0.0, 20.0, 0.0, 0.05,
                                          key="liq_user_spread")
        freeze_override = int(st.number_input("Exit freeze, lower circuits (0 = longest past run, at least 3)", 0, 30, 0, 1,
                                              key="liq_freeze"))
    export(ctx, locals())
