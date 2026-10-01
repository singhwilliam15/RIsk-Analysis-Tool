"""Market risk page (Pillar 1): the summary cards and the original analysis tabs, unchanged."""

import streamlit as st

from ui import backtest_tab, export_tab, models_tab, portfolio_tab, price_tab, stress_tab
from ui.overview import render_overview


def render(ctx):
    render_overview(ctx)

    tab_names = ["📊 Model Comparison", "📈 Price & Volatility", "⚡ Stress Testing", "🔬 Backtesting", "📥 Excel Report"]
    if ctx.is_portfolio:
        tab_names.insert(1, "🧩 Portfolio Risk")
    tabs = st.tabs(tab_names)
    if ctx.is_portfolio:
        tab1, tab_portfolio, tab2, tab3, tab4, tab5 = tabs
    else:
        tab1, tab2, tab3, tab4, tab5 = tabs

    models_tab.render(ctx, tab1)
    if ctx.is_portfolio:
        ctx.decomposition = portfolio_tab.render(ctx, tab_portfolio)
    price_tab.render(ctx, tab2)
    stress_tab.render(ctx, tab3)
    backtest_tab.render(ctx, tab4)
    export_tab.render(ctx, tab5)
