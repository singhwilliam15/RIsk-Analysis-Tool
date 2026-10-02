"""Data-quality warning, data caption and the five summary cards."""

import numpy as np
import streamlit as st
from data_fetcher import SUSPICIOUS_MOVE


def render_overview(ctx):
    """Data-quality warning, data caption and the five summary cards."""
    cl_label = ctx.cl_label
    company_name = ctx.company_name
    curr_sym = ctx.curr_sym
    current_price = ctx.current_price
    data_note = ctx.data_note
    df = ctx.df
    diversification = getattr(ctx, 'diversification', None)
    is_portfolio = ctx.is_portfolio
    point_name = ctx.point_name
    recommendation = ctx.recommendation
    recommended_model = ctx.recommended_model
    required_days = ctx.required_days
    returns = ctx.returns
    stats = ctx.stats
    suspicious = ctx.suspicious
    symbol = ctx.symbol
    test_days = ctx.test_days
    var_ewma = ctx.var_ewma
    var_hist = ctx.var_hist
    var_selected = ctx.var_selected

    st.subheader(f"📈 Risk Overview for {company_name} ({symbol})")
    flagged_moves = [f"{t} {row.Date:%d %b %Y}: {row.Returns:+.0%}, next day {row.Next_Day_Return:+.0%}"
                     + (" (reversed: likely data error)" if row.Reversed else "")
                     for t, table in suspicious.items() for row in table.itertuples()]
    if flagged_moves:
        st.warning(f"⚠ **Check the data:** {len(flagged_moves)} daily move{'' if len(flagged_moves) == 1 else 's'} larger than "
                   f"{SUSPICIOUS_MOVE:.0%}: " + "; ".join(flagged_moves[:6]) + (" …" if len(flagged_moves) > 6 else "") +
                   ". Moves marked *reversed* undo themselves the next day, which usually means a bad price in the source; "
                   "the others may be real events such as crashes or results days. "
                   "The data are used as downloaded; choose a shorter lookback to exclude these days.")
    st.caption(f"Data: {data_note} · {len(returns)} daily returns · {df['Date'].iloc[0]:%d %b %Y} to {df['Date'].iloc[-1]:%d %b %Y}")

    col1, col2, col3, col4, col5 = st.columns(5)

    first_card = (
        (col1, "Diversification Benefit", f"{curr_sym}{diversification['diversification_benefit']:,.0f}",
         f"{diversification['diversification_ratio']:.0%} lower than standalone", "metric-sub")
        if is_portfolio else
        (col1, "Current Stock Price", f"{curr_sym}{current_price:,.2f}", f"Data Points: {stats['n_obs']} Days", "metric-sub")
    )
    trusted = ctx.trusted

    def money(v):
        return f"{curr_sym}{v:,.0f}"

    def trust_line(metric):
        return f"{metric.range_text(money)} · grade {metric.grade}"

    cards = [
        first_card,
        (col2, "Annualized Volatility", f"{stats['ann_vol']:.2%}", f"EWMA today: {var_ewma['sigma_forecast'] * np.sqrt(252):.2%}", "metric-sub"),
        (col3, f"Historical VaR ({cl_label})", money(var_hist['var_scaled_amount']),
         f"Loss ({var_hist['var_daily_pct']:.2%})<br>{trust_line(trusted['Historical']['VaR'])}", "metric-sub-red"),
        (col4, f"Expected Shortfall ({cl_label})", money(var_hist['cvar_scaled_amount']),
         f"Tail Loss ({var_hist['cvar_daily_pct']:.2%})<br>{trust_line(trusted['Historical']['ES'])}", "metric-sub-red"),
        (
            (col5, "Recommended Model", recommended_model,
             f"Lowest tick loss among passing · VaR {money(var_selected[point_name(recommended_model)]['var_scaled_amount'])}"
             f"<br>{trust_line(trusted[point_name(recommended_model)]['VaR'])}", "metric-sub")
            if recommendation["status"] == "recommended" else
            (col5, "Best VaR Model", recommended_model,
             f"Fails the ES test; ES understated · VaR {money(var_selected[point_name(recommended_model)]['var_scaled_amount'])}"
             f"<br>{trust_line(trusted[point_name(recommended_model)]['VaR'])}", "metric-sub-red")
            if recommendation["status"] == "var_only" else
            (col5, "No Model Passes", recommended_model,
             f"Lowest tick loss shown; use with caution<br>{trust_line(trusted[point_name(recommended_model)]['VaR'])}",
             "metric-sub-red")
            if recommendation["status"] == "none_pass" else
            (col5, "Recommended Model", "Not enough data", f"{test_days} of {required_days} test days needed", "metric-sub-red")
        ),
    ]
    for col, label, value, sub, sub_class in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size: {'1.6rem' if len(value) < 14 else '1.15rem'}">{value}</div>
                <div class="{sub_class}">{sub}</div>
            </div>
            """, unsafe_allow_html=True)
