"""
Value at Risk (VaR) Automated Analysis Tool — Main Streamlit Web App
Eight VaR/ES models (up to GARCH(1,1)-t) for a stock or portfolio, out-of-sample VaR and ES backtests with a tick-loss model
ranking, portfolio risk decomposition, stress testing and Excel export.

This file only wires the UI sections together; each lives in ui/ and the calculations in the top-level modules.
"""

import streamlit as st

from ui import backtest_tab, export_tab, models_tab, portfolio_tab, price_tab, stress_tab
from ui.analysis import compute
from ui.context import new_context
from ui.data import load_data
from ui.overview import render_overview
from ui.sidebar import render_sidebar
from ui.styles import setup_page

setup_page()
ctx = new_context()
render_sidebar(ctx)
load_data(ctx)  # stops the script with an error message if the data cannot be used
compute(ctx)
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
