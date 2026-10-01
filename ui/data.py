"""Download and validate prices for the stock or portfolio; set the risk-free rate."""

import streamlit as st
from data_fetcher import suspicious_returns
from portfolio import (
    align_asset_returns,
    alignment_report,
    build_portfolio_frame,
    current_weights,
    normalize_weights,
)
from ui.cache import cached_fetch
from ui.context import export


def load_data(ctx):
    """Download and validate prices for the stock or portfolio; set the risk-free rate."""
    holdings_input = getattr(ctx, 'holdings_input', None)
    is_portfolio = ctx.is_portfolio
    period_input = ctx.period_input
    rebalance_mode = getattr(ctx, 'rebalance_mode', None)
    ticker_input = getattr(ctx, 'ticker_input', None)

    TICKER_TIP = "Tip: For Indian stocks listed on NSE, add '.NS' suffix (e.g. ASIANPAINT.NS, BRITANNIA.NS). For BSE, add '.BO'. US stocks do not require a suffix."

    if not is_portfolio:
        with st.spinner(f"Fetching market data for {ticker_input} from Yahoo Finance..."):
            data_res = cached_fetch(ticker_input, period=period_input)

        if not data_res["success"]:
            st.error(data_res["error"])
            st.info(TICKER_TIP)
            st.stop()

        symbol = data_res["symbol"]
        company_name = data_res["company_name"]
        currency = data_res["currency"]
        current_price = data_res["current_price"]
        df = data_res["df"]
        benchmark_tickers = [symbol]
        data_note = f"{data_res['data_source']}, {data_res['price_basis']} closes"
        suspicious = {symbol: suspicious_returns(df)}
    else:
        try:
            weights = normalize_weights(holdings_input)
        except ValueError as exc:
            st.error(f"Holdings table: {exc}")
            st.stop()

        with st.spinner(f"Fetching market data for {len(weights)} holdings from Yahoo Finance..."):
            fetched = {t: cached_fetch(t, period=period_input) for t in weights.index}

        failed = [res["error"] for res in fetched.values() if not res["success"]]
        if failed:
            st.error("Could not load every holding:\n\n" + "\n\n".join(f"- {e}" for e in failed))
            st.info(TICKER_TIP)
            st.stop()

        currencies = {t: res["currency"] for t, res in fetched.items()}
        if len(set(currencies.values())) > 1:
            st.error("Holdings trade in different currencies (" + ", ".join(f"{t}: {c}" for t, c in currencies.items()) +
                     "). Mixing currencies needs FX conversion, which this tool does not do; use holdings in one currency.")
            st.stop()

        suspicious = {t: suspicious_returns(res["df"]) for t, res in fetched.items()}
        alignment = alignment_report({t: res["df"] for t, res in fetched.items()})
        asset_returns = align_asset_returns({t: res["df"] for t, res in fetched.items()})
        if len(asset_returns) < 60:
            st.error(f"The holdings share only {len(asset_returns)} common trading days; at least 60 are needed.")
            st.stop()

        asset_names = {t: res["company_name"] for t, res in fetched.items()}
        symbol = "PORTFOLIO"
        company_name = f"{len(weights)}-stock portfolio"
        currency = next(iter(currencies.values()))
        current_price = None
        df = build_portfolio_frame(asset_returns, weights, rebalance=rebalance_mode)
        weights_now = current_weights(asset_returns, weights, rebalance_mode)  # targets, or drifted buy-and-hold weights
        benchmark_tickers = list(weights.index)
        sources = sorted({f"{res['data_source']}, {res['price_basis']} closes" for res in fetched.values()})
        data_note = "; ".join(sources)

    returns = df["Returns"].dropna()

    curr_sym = "₹" if currency == "INR" else "$" if currency == "USD" else currency + " "

    risk_free_pct = st.sidebar.number_input(
        f"Risk-free rate, % p.a. ({currency})", min_value=0.0, max_value=20.0,
        value=6.5 if currency == "INR" else 4.0, step=0.25, key=f"risk_free_{currency}",
        help="Editable assumption used for Sharpe and Sortino ratios, not a live market rate. "
             "Defaults: 6.5% for INR, 4.0% for USD."
    )
    export(ctx, locals())
