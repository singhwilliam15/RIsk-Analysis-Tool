"""
Value at Risk (VaR) Automated Analysis Tool — Main Streamlit Web App
Replicates and automates the exact quantitative methodology from VaR_Risk_Management_Tool.xlsx
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_fetcher import fetch_stock_data
from var_calculator import (
    compute_returns,
    calculate_portfolio_statistics,
    calculate_historical_var,
    calculate_parametric_var,
    calculate_monte_carlo_var,
    perform_kupiec_backtest,
    run_stress_testing
)
from excel_exporter import generate_excel_var_report

# Page Configuration
st.set_page_config(
    page_title="VaR Risk Management Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Dark Theme Accent Styling */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1A1F2C 0%, #11151C 100%);
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #68D391;
    }
    .metric-sub-red {
        font-size: 0.8rem;
        color: #FC8181;
    }
    .badge-green {
        background-color: #1C4532;
        color: #68D391;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-yellow {
        background-color: #5B4712;
        color: #F6AD55;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-red {
        background-color: #63171B;
        color: #FC8181;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("⚡ Value at Risk (VaR) Automated Analysis Tool")
st.caption("Automated Risk Management Dashboard matching `VaR_Risk_Management_Tool.xlsx` & `ASIAN-PAINTS-VaR-ANALYSIS.xlsx`")

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
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

selected_preset = st.sidebar.selectbox("Quick Preset Ticker", options=["Custom Ticker"] + list(quick_tickers.keys()))

if selected_preset != "Custom Ticker":
    default_ticker = quick_tickers[selected_preset]
else:
    default_ticker = "ASIANPAINT.NS"

ticker_input = st.sidebar.text_input("Enter Yahoo Finance Ticker Symbol", value=default_ticker, help="Examples: ASIANPAINT.NS, BRITANNIA.NS, RELIANCE.NS, AAPL, MSFT").strip().upper()

period_input = st.sidebar.selectbox("Historical Lookback Window", options=["1y", "2y", "5y", "max"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Portfolio Settings")

investment_amount = st.sidebar.number_input("Portfolio Investment Amount", min_value=1000.0, max_value=1000000000.0, value=1000000.0, step=50000.0, format="%.2f")

confidence_level = st.sidebar.select_slider("Confidence Level (1 - α)", options=[0.90, 0.95, 0.99], value=0.95, format_func=lambda x: f"{int(x*100)}%")

holding_period = st.sidebar.selectbox("Holding Period (Days)", options=[1, 5, 10, 21, 30], index=0, help="VaR scaled by sqrt(days)")

num_sims = st.sidebar.selectbox("Monte Carlo Simulations", options=[1000, 2500, 5000, 10000], index=2)

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity Financial Risk Engine v2.0")

# -------------------------------------------------------------
# FETCH DATA
# -------------------------------------------------------------
with st.spinner(f"Fetching market data for {ticker_input} from Yahoo Finance..."):
    data_res = fetch_stock_data(ticker_input, period=period_input)

if not data_res["success"]:
    st.error(data_res["error"])
    st.info("Tip: For Indian stocks listed on NSE, add '.NS' suffix (e.g. ASIANPAINT.NS, BRITANNIA.NS). For BSE, add '.BO'. US stocks do not require a suffix.")
    st.stop()

symbol = data_res["symbol"]
company_name = data_res["company_name"]
currency = data_res["currency"]
current_price = data_res["current_price"]
df = data_res["df"]
returns = df["Returns"].dropna()

curr_sym = "₹" if currency == "INR" else "$" if currency == "USD" else currency + " "

# -------------------------------------------------------------
# CALCULATIONS
# -------------------------------------------------------------
stats = calculate_portfolio_statistics(returns)
var_hist = calculate_historical_var(returns, investment_amount, confidence_level, holding_period)
var_param = calculate_parametric_var(returns, investment_amount, confidence_level, holding_period)
var_mc = calculate_monte_carlo_var(returns, investment_amount, confidence_level, holding_period, num_simulations=num_sims, seed=42)

# Multi-confidence calculations for summary comparisons
var_hist_90 = calculate_historical_var(returns, investment_amount, 0.90, holding_period)
var_hist_95 = calculate_historical_var(returns, investment_amount, 0.95, holding_period)
var_hist_99 = calculate_historical_var(returns, investment_amount, 0.99, holding_period)

var_param_90 = calculate_parametric_var(returns, investment_amount, 0.90, holding_period)
var_param_95 = calculate_parametric_var(returns, investment_amount, 0.95, holding_period)
var_param_99 = calculate_parametric_var(returns, investment_amount, 0.99, holding_period)

var_mc_90 = calculate_monte_carlo_var(returns, investment_amount, 0.90, holding_period, num_simulations=num_sims, seed=42)
var_mc_95 = calculate_monte_carlo_var(returns, investment_amount, 0.95, holding_period, num_simulations=num_sims, seed=42)
var_mc_99 = calculate_monte_carlo_var(returns, investment_amount, 0.99, holding_period, num_simulations=num_sims, seed=42)

backtest = perform_kupiec_backtest(returns, var_hist["var_daily_pct"], confidence_level)
stress_df = run_stress_testing(investment_amount)

# -------------------------------------------------------------
# EXECUTIVE TOP CARDS
# -------------------------------------------------------------
st.subheader(f"📈 Risk Overview for {company_name} ({symbol})")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Stock Price</div>
        <div class="metric-value">{curr_sym}{current_price:,.2f}</div>
        <div class="metric-sub">Data Points: {stats['n_obs']} Days</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annualized Volatility</div>
        <div class="metric-value">{stats['ann_vol']:.2%}</div>
        <div class="metric-sub">Daily: {stats['daily_vol']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Historical VaR ({int(confidence_level*100)}%)</div>
        <div class="metric-value">{curr_sym}{var_hist['var_scaled_amount']:,.0f}</div>
        <div class="metric-sub-red">Max Loss ({var_hist['var_daily_pct']:.2%})</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Parametric VaR ({int(confidence_level*100)}%)</div>
        <div class="metric-value">{curr_sym}{var_param['var_scaled_amount']:,.0f}</div>
        <div class="metric-sub-red">Normal Model ({var_param['var_daily_pct']:.2%})</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Shortfall (CVaR)</div>
        <div class="metric-value">{curr_sym}{var_hist['cvar_scaled_amount']:,.0f}</div>
        <div class="metric-sub-red">Tail Loss ({var_hist['cvar_daily_pct']:.2%})</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# MAIN DASHBOARD TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Summary & Method Comparison", 
    "📈 Price & Return Visual Analytics", 
    "⚡ Stress Testing & Historical Crises", 
    "🔬 Backtesting & Traffic Light Validation", 
    "📥 Export & Custom Excel Report"
])

# -------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# -------------------------------------------------------------
with tab1:
    st.markdown("### 🏛️ Value at Risk (VaR) Methodology Comparison Table")
    st.caption(f"Portfolio Investment: **{curr_sym}{investment_amount:,.2f}** | Holding Period: **{holding_period} Day(s)**")
    
    summary_data = [
        {
            "Methodology": "Historical VaR (Non-Parametric)",
            "90% Confidence VaR": f"{curr_sym}{var_hist_90['var_scaled_amount']:,.2f} ({var_hist_90['var_daily_pct']:.2%})",
            "95% Confidence VaR": f"{curr_sym}{var_hist_95['var_scaled_amount']:,.2f} ({var_hist_95['var_daily_pct']:.2%})",
            "99% Confidence VaR": f"{curr_sym}{var_hist_99['var_scaled_amount']:,.2f} ({var_hist_99['var_daily_pct']:.2%})",
            "95% Expected Shortfall (CVaR)": f"{curr_sym}{var_hist_95['cvar_scaled_amount']:,.2f} ({var_hist_95['cvar_daily_pct']:.2%})"
        },
        {
            "Methodology": "Parametric VaR (Variance-Covariance)",
            "90% Confidence VaR": f"{curr_sym}{var_param_90['var_scaled_amount']:,.2f} ({var_param_90['var_daily_pct']:.2%})",
            "95% Confidence VaR": f"{curr_sym}{var_param_95['var_scaled_amount']:,.2f} ({var_param_95['var_daily_pct']:.2%})",
            "99% Confidence VaR": f"{curr_sym}{var_param_99['var_scaled_amount']:,.2f} ({var_param_99['var_daily_pct']:.2%})",
            "95% Expected Shortfall (CVaR)": f"{curr_sym}{var_param_95['cvar_scaled_amount']:,.2f} ({var_param_95['cvar_daily_pct']:.2%})"
        },
        {
            "Methodology": "Monte Carlo VaR (5,000 Simulations)",
            "90% Confidence VaR": f"{curr_sym}{var_mc_90['var_scaled_amount']:,.2f} ({var_mc_90['var_daily_pct']:.2%})",
            "95% Confidence VaR": f"{curr_sym}{var_mc_95['var_scaled_amount']:,.2f} ({var_mc_95['var_daily_pct']:.2%})",
            "99% Confidence VaR": f"{curr_sym}{var_mc_99['var_scaled_amount']:,.2f} ({var_mc_99['var_daily_pct']:.2%})",
            "95% Expected Shortfall (CVaR)": f"{curr_sym}{var_mc_95['cvar_scaled_amount']:,.2f} ({var_mc_95['cvar_daily_pct']:.2%})"
        }
    ]
    st.table(pd.DataFrame(summary_data))
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 📊 VaR Method Comparison across Confidence Levels")
        df_comp = pd.DataFrame([
            {"Method": "Historical", "90% VaR": var_hist_90['var_scaled_amount'], "95% VaR": var_hist_95['var_scaled_amount'], "99% VaR": var_hist_99['var_scaled_amount']},
            {"Method": "Parametric", "90% VaR": var_param_90['var_scaled_amount'], "95% VaR": var_param_95['var_scaled_amount'], "99% VaR": var_param_99['var_scaled_amount']},
            {"Method": "Monte Carlo", "90% VaR": var_mc_90['var_scaled_amount'], "95% VaR": var_mc_95['var_scaled_amount'], "99% VaR": var_mc_99['var_scaled_amount']}
        ]).melt(id_vars=["Method"], var_name="Confidence", value_name="VaR_Amount")
        
        fig_comp = px.bar(df_comp, x="Confidence", y="VaR_Amount", color="Method", barmode="group",
                          title=f"Potential Maximum Loss ({curr_sym})",
                          labels={"VaR_Amount": f"Loss Amount ({curr_sym})"},
                          color_discrete_sequence=["#3182CE", "#DD6B20", "#38A169"],
                          template="plotly_dark")
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📉 Return Distribution & VaR Cut-off Thresholds")
        fig_hist = px.histogram(df, x="Returns", nbins=60, title="Empirical Daily Return Distribution",
                                labels={"Returns": "Daily Return %"}, template="plotly_dark",
                                opacity=0.75, color_discrete_sequence=["#63B3ED"])
        
        # Add cutoff vertical lines
        fig_hist.add_vline(x=-var_hist_95['var_daily_pct'], line_dash="dash", line_color="#F6AD55", annotation_text="95% Hist VaR")
        fig_hist.add_vline(x=-var_hist_99['var_daily_pct'], line_dash="solid", line_color="#FC8181", annotation_text="99% Hist VaR")
        fig_hist.add_vline(x=-var_hist_95['cvar_daily_pct'], line_dash="dot", line_color="#E53E3E", annotation_text="95% CVaR (Tail)")
        
        st.plotly_chart(fig_hist, use_container_width=True)

    # Interpretation Card
    st.info(f"""
    📋 **Risk Interpretation & Key Insights**:
    - **Historical VaR ({int(confidence_level*100)}%)**: At {int(confidence_level*100)}% confidence level over a {holding_period}-day holding period, there is a {int((1-confidence_level)*100)}% probability that your investment of **{curr_sym}{investment_amount:,.2f}** in {company_name} will lose more than **{curr_sym}{var_hist['var_scaled_amount']:,.2f}** ({var_hist['var_daily_pct']:.2%}).
    - **Expected Shortfall (CVaR)**: If a tail event occurs beyond the {int(confidence_level*100)}% threshold, the average expected loss is **{curr_sym}{var_hist['cvar_scaled_amount']:,.2f}** ({var_hist['cvar_daily_pct']:.2%}).
    - **Sharpe Ratio**: **{stats['sharpe_ratio']:.2f}** | **Sortino Ratio**: **{stats['sortino_ratio']:.2f}** | **Max Drawdown**: **{stats['max_drawdown']:.2%}**.
    """)

# -------------------------------------------------------------
# TAB 2: PRICE & RETURN VISUAL ANALYTICS
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Stock Price & Volatility Analytics")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        fig_price = px.line(df, x="Date", y="Close", title=f"{company_name} ({symbol}) Closing Price History",
                            labels={"Close": f"Price ({curr_sym})"}, template="plotly_dark", color_discrete_sequence=["#4FD1C5"])
        st.plotly_chart(fig_price, use_container_width=True)
        
    with col_p2:
        fig_returns = px.line(df, x="Date", y="Returns", title="Daily Percentage Returns",
                              labels={"Returns": "Daily Return"}, template="plotly_dark", color_discrete_sequence=["#63B3ED"])
        fig_returns.add_hline(y=0, line_dash="dash", line_color="#A0AEC0")
        st.plotly_chart(fig_returns, use_container_width=True)
        
    st.markdown("#### 🌊 30-Day Rolling Annualized Volatility")
    fig_vol = px.area(df, x="Date", y="Rolling_30d_Vol", title="30-Day Rolling Annualized Volatility Time Series",
                      labels={"Rolling_30d_Vol": "Annualized Volatility (%)"}, template="plotly_dark", color_discrete_sequence=["#ED8936"])
    st.plotly_chart(fig_vol, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: STRESS TESTING
# -------------------------------------------------------------
with tab3:
    st.markdown("### ⚡ Stress Testing & Historical Scenario Analysis")
    st.caption("Simulates the impact of severe historical market crises on your current portfolio valuation.")
    
    col_st1, col_st2 = st.columns([1.2, 1])
    
    with col_st1:
        # Custom shock slider
        custom_shock_pct = st.slider("Custom Market Shock Scenario (%)", min_value=-60.0, max_value=0.0, value=-20.0, step=1.0)
        custom_impact = investment_amount * (custom_shock_pct / 100.0)
        custom_post = investment_amount + custom_impact
        
        df_display_stress = stress_df.copy()
        df_display_stress["Shock %"] = df_display_stress["Shock"].map(lambda x: f"{x:.1%}")
        df_display_stress["Portfolio Impact"] = df_display_stress["Portfolio_Impact"].map(lambda x: f"{curr_sym}{x:,.2f}")
        df_display_stress["Post-Shock Value"] = df_display_stress["Post_Shock_Value"].map(lambda x: f"{curr_sym}{x:,.2f}")
        
        st.dataframe(
            df_display_stress[["Scenario", "Shock %", "Portfolio Impact", "Post-Shock Value", "Recovery_Days", "Risk_Level"]],
            use_container_width=True,
            hide_index=True
        )
        
        st.warning(f"💡 **Custom Scenario Result**: A **{custom_shock_pct:.1f}%** drop causes a loss of **{curr_sym}{abs(custom_impact):,.2f}**, reducing portfolio to **{curr_sym}{custom_post:,.2f}**.")

    with col_st2:
        fig_stress = px.bar(stress_df, x="Scenario", y="Portfolio_Impact", color="Risk_Level",
                            title=f"Scenario Portfolio Loss ({curr_sym})",
                            labels={"Portfolio_Impact": f"Loss Amount ({curr_sym})"},
                            color_discrete_map={"HIGH": "#E53E3E", "MEDIUM": "#DD6B20", "LOW": "#38A169"},
                            template="plotly_dark")
        fig_stress.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_stress, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: BACKTESTING & BASEL TRAFFIC LIGHT
# -------------------------------------------------------------
with tab4:
    st.markdown("### 🔬 VaR Model Backtesting — Kupiec & Basel Traffic Light System")
    st.caption("Validates VaR model statistical accuracy by comparing predicted exceedances vs actual historical breaches.")
    
    col_b1, col_b2 = st.columns([1, 1.2])
    
    with col_b1:
        st.markdown("#### 📋 Backtest Diagnostics Table")
        
        traffic_badge = backtest["traffic_light"]
        if "GREEN" in traffic_badge:
            badge_html = f'<span class="badge-green">{traffic_badge}</span>'
        elif "YELLOW" in traffic_badge:
            badge_html = f'<span class="badge-yellow">{traffic_badge}</span>'
        else:
            badge_html = f'<span class="badge-red">{traffic_badge}</span>'
            
        st.markdown(f"### Status: {badge_html}", unsafe_allow_html=True)
        st.write(f"**Interpretation**: {backtest['status_desc']}")
        st.write(f"**Kupiec LR Test Result**: **{backtest['test_result']}**")
        
        bt_summary = pd.DataFrame([
            {"Diagnostic Metric": "Total Historical Observations (T)", "Value": f"{backtest['total_observations']} days"},
            {"Diagnostic Metric": "Model Confidence Level", "Value": f"{backtest['confidence_level']:.1%}"},
            {"Diagnostic Metric": "Expected VaR Failures (α × T)", "Value": f"{backtest['expected_failures']:.1f}"},
            {"Diagnostic Metric": "Actual Historical Failures", "Value": f"{backtest['actual_breaches']}"},
            {"Diagnostic Metric": "Empirical Breach Rate", "Value": f"{backtest['breach_rate']:.2%}"},
            {"Diagnostic Metric": "Kupiec LR Statistic", "Value": f"{backtest['lr_stat']:.4f}"},
            {"Diagnostic Metric": "Chi-Square Critical Value (95%)", "Value": f"{backtest['chi_sq_critical']:.3f}"},
            {"Diagnostic Metric": "Scaled 250-Day Breaches", "Value": f"{backtest['scaled_breaches_250']}"}
        ])
        st.table(bt_summary)

    with col_b2:
        st.markdown("#### 🔴 Historical VaR Exception Timeline")
        df_breaches = df.copy()
        cutoff_val = -var_hist['var_daily_pct']
        df_breaches['Is_Breach'] = df_breaches['Returns'] < cutoff_val
        
        fig_breach = px.scatter(df_breaches, x="Date", y="Returns", color="Is_Breach",
                                title="Daily Returns vs VaR Cutoff Threshold",
                                color_discrete_map={True: "#FC8181", False: "#4FD1C5"},
                                labels={"Is_Breach": "VaR Breach"}, template="plotly_dark")
        fig_breach.add_hline(y=cutoff_val, line_dash="dash", line_color="#FC8181", annotation_text=f"VaR Cutoff ({cutoff_val:.2%})")
        st.plotly_chart(fig_breach, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: EXPORT & REPORTS
# -------------------------------------------------------------
with tab5:
    st.markdown("### 📥 Download Custom Excel Risk Management Report")
    st.write("Generate a complete, fully formatted `.xlsx` workbook containing all raw price data, calculated VaR statistics, backtesting results, and stress tests structured exactly like `VaR_Risk_Management_Tool.xlsx`.")
    
    excel_bytes = generate_excel_var_report(
        symbol=symbol,
        company_name=company_name,
        currency=currency,
        investment=investment_amount,
        confidence_level=confidence_level,
        holding_period=holding_period,
        df_data=df,
        stats=stats,
        var_hist=var_hist,
        var_param=var_param,
        var_mc=var_mc,
        backtest=backtest,
        stress_df=stress_df
    )
    
    st.download_button(
        label=f"📥 Download Complete Excel Report ({symbol}_VaR_Report.xlsx)",
        data=excel_bytes,
        file_name=f"{symbol}_VaR_Risk_Report_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )
    
    st.markdown("---")
    st.markdown("#### 📄 Raw Market Data Export (CSV)")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📄 Download Raw Prices & Returns ({symbol}_data.csv)",
        data=csv_bytes,
        file_name=f"{symbol}_market_data.csv",
        mime="text/csv"
    )
