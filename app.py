"""
Value at Risk (VaR) Automated Analysis Tool — Main Streamlit Web App
Eight VaR/ES models (up to GARCH(1,1)-t) for a stock or portfolio, out-of-sample VaR and ES backtests with a tick-loss model
ranking, portfolio risk decomposition, stress testing and Excel export.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from scipy.stats import norm, t as student_t

from data_fetcher import fetch_stock_data, suspicious_returns, SUSPICIOUS_MOVE
from var_calculator import (
    calculate_portfolio_statistics,
    calculate_all_var,
    ewma_volatility,
    fit_models,
    rolling_forecasts,
    MONTE_CARLO_BACKTEST_NAME,
    perform_kupiec_backtest,
    backtest_all_methods,
    recommend_model,
    min_backtest_days,
    BACKTEST_WINDOW,
    LOW_POWER,
    estimate_beta,
    historical_worst_losses,
)
from excel_exporter import generate_excel_var_report
from portfolio import (
    normalize_weights,
    align_asset_returns,
    build_portfolio_frame,
    diversification_summary,
    alignment_report,
    current_weights,
    risk_decomposition,
    what_if,
    what_if_weights,
    REBALANCE_DAILY,
    BUY_AND_HOLD,
    DECOMPOSITION_BASES,
    HISTORICAL_ES,
    PARAMETRIC_VAR,
)
from stress import (
    MARKETS,
    PROXY,
    REPLAY,
    NO_MARKET_DATA,
    downside_beta,
    load_scenarios,
    price_series,
    run_scenarios,
    volatility_shock,
)

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
st.caption("Eight VaR / Expected Shortfall models, from historical simulation to GARCH(1,1)-t, for a stock or portfolio, with out-of-sample VaR and ES backtests, risk decomposition, stress testing and Excel export")

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

analysis_mode = st.sidebar.radio("Analysis Mode", ["Single Stock", "Portfolio"], horizontal=True)
is_portfolio = analysis_mode == "Portfolio"

if not is_portfolio:
    selected_preset = st.sidebar.selectbox("Quick Preset Ticker", options=["Custom Ticker"] + list(quick_tickers.keys()))

    if selected_preset != "Custom Ticker":
        default_ticker = quick_tickers[selected_preset]
    else:
        default_ticker = "ASIANPAINT.NS"

    ticker_input = st.sidebar.text_input("Enter Yahoo Finance Ticker Symbol", value=default_ticker, help="Examples: ASIANPAINT.NS, BRITANNIA.NS, RELIANCE.NS, AAPL, MSFT").strip().upper()
else:
    st.sidebar.caption("Holdings and weights (any units; they are scaled to 100%). Add or delete rows in the table. All holdings must trade in the same currency.")
    holdings_input = st.sidebar.data_editor(
        pd.DataFrame({
            "Ticker": ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "ASIANPAINT.NS", "BRITANNIA.NS"],
            "Weight": [30.0, 25.0, 20.0, 15.0, 10.0],
        }),
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key="holdings",
    )
    rebalance_mode = st.sidebar.radio("Rebalancing", [REBALANCE_DAILY, BUY_AND_HOLD],
                                      help="Daily: weights reset to the targets every day. Buy-and-hold: shares are bought once and weights drift with prices.")
    st.sidebar.caption("Long-only: weights must be positive. Short positions would need borrow costs, margin and "
                       "a gross/net exposure definition that this tool does not model.")

period_input = st.sidebar.selectbox("Historical Lookback Window", options=["1y", "2y", "5y", "max"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("💼 Portfolio Settings")

investment_amount = st.sidebar.number_input("Portfolio Investment Amount", min_value=1000.0, max_value=1000000000.0, value=1000000.0, step=50000.0, format="%.2f")

def pct_label(cl: float) -> str:
    return f"{cl * 100:g}%"

confidence_level = st.sidebar.select_slider("Confidence Level (1 - α)", options=[0.90, 0.95, 0.975, 0.99], value=0.95,
                                            format_func=pct_label, help="97.5% is the Basel FRTB Expected Shortfall level.")

holding_period = st.sidebar.selectbox("Holding Period (Days)", options=[1, 5, 10, 21, 30], index=0, help="VaR scaled by sqrt(days)")

num_sims = st.sidebar.selectbox("Monte Carlo Simulations", options=[1000, 2500, 5000, 10000], index=2)


# -------------------------------------------------------------
# FETCH DATA
# -------------------------------------------------------------
cached_fetch = st.cache_data(ttl=3600, show_spinner=False)(fetch_stock_data)

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

# -------------------------------------------------------------
# CALCULATIONS
# -------------------------------------------------------------
stats = calculate_portfolio_statistics(returns, risk_free_rate=risk_free_pct / 100)

confidence_levels = (0.90, 0.95, 0.975, 0.99)
model_fits = fit_models(returns)  # GARCH(1,1)-t and Student-t MLE, fitted once and reused at every confidence level
var_by_level = {
    cl: calculate_all_var(returns, investment_amount, cl, holding_period, num_simulations=num_sims, seed=42, fits=model_fits)
    for cl in confidence_levels
}
var_selected = var_by_level[confidence_level]
method_names = list(var_selected)
var_hist = var_selected["Historical"]
var_ewma = var_selected["EWMA (RiskMetrics)"]
cl_label = pct_label(confidence_level)
var_garch = var_selected["GARCH(1,1)-t"]
var_cf = var_selected["Cornish-Fisher"]
if not var_garch.get("fallback"):
    gp = var_garch["garch_params"]
    garch_text = (f"Fitted α = {gp.alpha:.3f}, β = {gp.beta:.3f} (persistence {gp.persistence:.3f}), ν = {gp.nu:.1f}; "
                  f"today's volatility {var_garch['sigma_forecast'] * np.sqrt(252):.1%} a year against a long-run {var_garch['long_run_vol'] * np.sqrt(252):.1%}.")
else:
    garch_text = "The fit did not converge on this sample, so EWMA is shown instead."

# Benchmark: normal beta and downside (crisis) beta over the lookback window
is_indian = all(t.endswith((".NS", ".BO")) for t in benchmark_tickers)
market_key = "NIFTY50" if is_indian else "SP500"
benchmark_symbol, benchmark_name = MARKETS[market_key]
bench_res = cached_fetch(benchmark_symbol, period=period_input)
beta = beta_down = float("nan")
stock_by_date = df.set_index(df["Date"].dt.normalize())["Returns"]
if bench_res["success"]:
    bench_df = bench_res["df"]
    bench_by_date = bench_df.set_index(bench_df["Date"].dt.normalize())["Returns"]
    beta = estimate_beta(stock_by_date, bench_by_date)
    beta_down = downside_beta(stock_by_date, bench_by_date)
beta_used = beta if np.isfinite(beta) else 1.0
proxy_beta = beta_down if np.isfinite(beta_down) else beta_used
worst_df = historical_worst_losses(returns, investment_amount)

# Crisis scenarios: replayed over the full price histories (independent of the lookback window)
market_max = cached_fetch(benchmark_symbol, period="max")
if not is_portfolio:
    position_max = cached_fetch(symbol, period="max")
    position_history = price_series(position_max["df"]).pct_change().dropna() if position_max["success"] else stock_by_date.dropna()
else:
    max_frames = {t: cached_fetch(t, period="max") for t in weights.index}
    if all(res["success"] for res in max_frames.values()):
        long_returns = align_asset_returns({t: res["df"] for t, res in max_frames.items()})
        long_frame = build_portfolio_frame(long_returns, weights, rebalance=rebalance_mode)
        position_history = long_frame.set_index("Date")["Returns"]
    else:
        position_history = stock_by_date.dropna()
if market_max["success"]:
    stress_table = run_scenarios(position_history, price_series(market_max["df"]), load_scenarios(market=market_key),
                                 proxy_beta, investment_amount)
else:
    stress_table = None
vol_shock_table = volatility_shock(returns, investment_amount, confidence_level, holding_period)

# Out-of-sample backtest of every model: each day's VaR comes only from the preceding 250 days.
# The window stays fixed; too few remaining test days are flagged rather than hidden by shrinking it.
backtest_window = BACKTEST_WINDOW
test_days = max(len(returns) - backtest_window, 0)
required_days = min_backtest_days(confidence_level)
low_power = test_days < required_days
if test_days > 0:
    rolling = rolling_forecasts(returns, confidence_level, window=backtest_window, num_simulations=num_sims)
    forecasts = rolling["var"]
    backtest_table = backtest_all_methods(returns, forecasts, confidence_level, rolling["es"], rolling["sigma"])
    # At one day, Monte Carlo is GARCH-t plus simulation noise, so it is scored but not recommended
    recommendation = recommend_model(backtest_table, exclude=[MONTE_CARLO_BACKTEST_NAME])
    passing_models = backtest_table.loc[backtest_table["Verdict"] == "PASS", "Method"].tolist()
else:
    rolling, forecasts, backtest_table, passing_models = None, None, None, []
    recommendation = {"model": None, "status": "low_power"}
recommended_model = recommendation["model"]


def point_name(backtest_name: str) -> str:
    """Map a backtest column name to the matching model in the point-estimate table."""
    return method_names[-1] if backtest_name == MONTE_CARLO_BACKTEST_NAME else backtest_name

if recommendation["status"] == "recommended":
    backtest_summary = (f"{', '.join(passing_models)} passed all three tests at {cl_label}; "
                        f"**{recommended_model}** is recommended (lowest tick loss among the passing models).")
elif recommendation["status"] == "none_pass":
    backtest_summary = (f"no model passed all three tests at {cl_label}. **{recommended_model}** has the lowest tick loss, "
                        "but its breaches are still statistically off, so treat every VaR figure here with caution.")
else:
    backtest_summary = (f"only {test_days} out-of-sample day{"" if test_days == 1 else "s"} available; at least {required_days} are needed for a meaningful "
                        f"test at {cl_label}. Choose a 5y or max lookback.")

# Portfolio decomposition
if is_portfolio:
    diversification = diversification_summary(asset_returns, weights_now, investment_amount, confidence_level, holding_period)
    correlation = asset_returns[weights.index].corr()

# Distribution diagnostics
skewness, excess_kurtosis = stats["skewness"], stats["kurtosis"]
jarque_bera = stats["n_obs"] / 6 * (skewness ** 2 + excess_kurtosis ** 2 / 4)
jarque_bera_p = float(np.exp(-jarque_bera / 2))  # chi-square(2) survival function

# -------------------------------------------------------------
# EXECUTIVE TOP CARDS
# -------------------------------------------------------------
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
cards = [
    first_card,
    (col2, "Annualized Volatility", f"{stats['ann_vol']:.2%}", f"EWMA today: {var_ewma['sigma_forecast'] * np.sqrt(252):.2%}", "metric-sub"),
    (col3, f"Historical VaR ({cl_label})", f"{curr_sym}{var_hist['var_scaled_amount']:,.0f}", f"Loss ({var_hist['var_daily_pct']:.2%})", "metric-sub-red"),
    (col4, f"Expected Shortfall ({cl_label})", f"{curr_sym}{var_hist['cvar_scaled_amount']:,.0f}", f"Tail Loss ({var_hist['cvar_daily_pct']:.2%})", "metric-sub-red"),
    (
        (col5, "Recommended Model", recommended_model,
         f"Lowest tick loss among passing · VaR {curr_sym}{var_selected[point_name(recommended_model)]['var_scaled_amount']:,.0f}", "metric-sub")
        if recommendation["status"] == "recommended" else
        (col5, "No Model Passes", recommended_model, "Lowest tick loss shown; use with caution", "metric-sub-red")
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

# -------------------------------------------------------------
# MAIN DASHBOARD TABS
# -------------------------------------------------------------
tab_names = ["📊 Model Comparison", "📈 Price & Volatility", "⚡ Stress Testing", "🔬 Backtesting", "📥 Excel Report"]
if is_portfolio:
    tab_names.insert(1, "🧩 Portfolio Risk")
tabs = st.tabs(tab_names)
if is_portfolio:
    tab1, tab_portfolio, tab2, tab3, tab4, tab5 = tabs
else:
    tab1, tab2, tab3, tab4, tab5 = tabs

MODEL_COLORS = ["#3182CE", "#DD6B20", "#38A169", "#D53F8C", "#805AD5", "#4FD1C5", "#F56565", "#ECC94B"]

# -------------------------------------------------------------
# TAB 1: MODEL COMPARISON
# -------------------------------------------------------------
with tab1:
    st.markdown("### 🏛️ VaR and Expected Shortfall by Model")
    st.caption(f"Position: **{curr_sym}{investment_amount:,.2f}** | Holding Period: **{holding_period} Day(s)** | ES shown at **{cl_label}**")

    summary_rows = []
    for method in method_names:
        flag = " ⚠" if (method == "Cornish-Fisher" and not var_cf["cf_valid"]) or var_selected[method].get("fallback") else ""
        row = {"Model": method + flag}
        for cl in confidence_levels:
            res = var_by_level[cl][method]
            row[f"{pct_label(cl)} VaR"] = f"{curr_sym}{res['var_scaled_amount']:,.0f} ({res['var_scaled_pct']:.2%})"
        row[f"{cl_label} ES"] = f"{curr_sym}{var_selected[method]['cvar_scaled_amount']:,.0f} ({var_selected[method]['cvar_scaled_pct']:.2%})"
        row["Multi-day rule"] = var_selected[method]["scaling_rule"] if holding_period > 1 else "1 day"
        summary_rows.append(row)
    st.table(pd.DataFrame(summary_rows).set_index("Model"))

    if holding_period > 1:
        st.caption(
            f"**{holding_period}-day check (Historical):** √t scaling gives **{curr_sym}{var_hist['var_scaled_amount']:,.0f}** VaR / "
            f"**{curr_sym}{var_hist['cvar_scaled_amount']:,.0f}** ES; the actual overlapping {holding_period}-day returns give "
            f"**{curr_sym}{var_hist['overlapping_var_amount']:,.0f}** / **{curr_sym}{var_hist['overlapping_cvar_amount']:,.0f}** "
            f"({var_hist['overlapping_observations']} overlapping windows, so the observations are not independent)."
        )
    if not var_cf["cf_valid"]:
        st.warning(f"⚠ **Cornish-Fisher is outside its valid region; treat with caution.** With skewness {var_cf['skewness']:.2f} and "
                   f"excess kurtosis {var_cf['excess_kurtosis']:.2f}, the expansion is not monotonic in z, so its quantiles are unreliable.")
    if var_garch.get("fallback"):
        st.warning(f"⚠ GARCH(1,1)-t did not converge on this sample ({var_garch['fallback_reason']}); its row shows EWMA, "
                   "and Monte Carlo samples a fitted normal instead.")
    mc_result = var_selected[method_names[-1]]
    if mc_result["few_tail_draws"]:
        st.warning(f"Monte Carlo: {num_sims:,} simulations at {cl_label} leave only {mc_result['tail_draws']} draws in the tail, "
                   "so its VaR and especially its ES are noisy. Use 10,000 simulations or more.")

    with st.expander("How each model works"):
        st.markdown(f"""
- **Historical:** the empirical {pct_label(1 - confidence_level)} percentile of past daily returns. No distribution assumed.
- **Parametric (Normal):** `z·σ − μ`. Fast, but a normal distribution has thin tails.
- **Student-t:** fatter-tailed distribution fitted by {var_selected['Student-t']['fit_method']} (ν = {var_selected['Student-t']['degrees_of_freedom']:.1f}).
- **Cornish-Fisher:** adjusts the normal quantile for skewness ({skewness:.2f}) and excess kurtosis ({excess_kurtosis:.2f}); {"inside" if var_cf["cf_valid"] else "**outside**"} the region where the expansion is valid.
- **EWMA (RiskMetrics):** normal quantile on an exponentially weighted volatility forecast (λ = 0.94), so it reacts to recent market stress.
- **FHS (EWMA-filtered):** divides past returns by their EWMA volatility, takes the empirical quantile of these standardized returns and rescales it by tomorrow's volatility: real tail shape, current volatility.
- **GARCH(1,1)-t:** volatility clustering and fat tails together. {garch_text}
- **Monte Carlo:** {num_sims:,} simulated paths of the fitted GARCH(1,1)-t process; for multi-day horizons volatility evolves along each path instead of using √t.

**Multi-day horizons:** Normal, Student-t, Cornish-Fisher and EWMA use `z·σ·√t − μ·t`; Historical and FHS scale the 1-day quantile by √t; GARCH sums its expected daily variances; Monte Carlo simulates full paths. √t assumes independent returns, so compare it with the overlapping-window check above.
""")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 📊 VaR by Model and Confidence Level")
        df_comp = pd.DataFrame([
            {"Model": m, "Confidence": pct_label(cl), "VaR_Amount": var_by_level[cl][m]["var_scaled_amount"]}
            for m in method_names for cl in confidence_levels
        ])
        fig_comp = px.bar(df_comp, x="Confidence", y="VaR_Amount", color="Model", barmode="group",
                          title=f"Potential Loss ({curr_sym})",
                          labels={"VaR_Amount": f"Loss Amount ({curr_sym})"},
                          color_discrete_sequence=MODEL_COLORS, template="plotly_dark")
        st.plotly_chart(fig_comp, width="stretch")

    with col_chart2:
        st.markdown("#### 📉 Return Distribution vs Normal and Student-t")
        mu, sigma = returns.mean(), returns.std(ddof=1)
        t_fit = var_selected["Student-t"]
        nu, t_loc, t_scale = t_fit["degrees_of_freedom"], t_fit["loc"], t_fit["scale"]
        x_grid = np.linspace(returns.min(), returns.max(), 400)

        fig_hist = go.Figure()
        fig_hist.add_histogram(x=returns, nbinsx=80, histnorm="probability density", name="Actual returns",
                               marker_color="#63B3ED", opacity=0.6)
        fig_hist.add_scatter(x=x_grid, y=norm.pdf(x_grid, mu, sigma), mode="lines", name="Normal fit",
                             line=dict(color="#F6AD55", width=2))
        fig_hist.add_scatter(x=x_grid, y=student_t.pdf((x_grid - t_loc) / t_scale, nu) / t_scale, mode="lines",
                             name=f"Student-t fit (ν={nu:.1f})", line=dict(color="#68D391", width=2, dash="dash"))
        fig_hist.add_vline(x=-var_hist["var_daily_pct"], line_dash="dot", line_color="#FC8181",
                           annotation_text=f"{cl_label} Hist VaR")
        fig_hist.update_layout(template="plotly_dark", title="Daily Return Density",
                               xaxis_title="Daily Return", yaxis_title="Density", xaxis_tickformat=".1%")
        st.plotly_chart(fig_hist, width="stretch")
        st.caption(f"Skewness {skewness:.2f} · Excess kurtosis {excess_kurtosis:.2f} · Jarque-Bera {jarque_bera:,.1f} "
                   f"(p = {jarque_bera_p:.4f}){' → returns are not normal' if jarque_bera_p < 0.05 else ''}")

    var_values = [var_selected[m]["var_scaled_amount"] for m in method_names]
    st.info(f"""
    📋 **Risk Interpretation**:
    - **Historical VaR ({cl_label})**: over a {holding_period}-day horizon there is a {pct_label(1 - confidence_level)} chance that this {curr_sym}{investment_amount:,.0f} position in {company_name} loses more than **{curr_sym}{var_hist['var_scaled_amount']:,.0f}** ({var_hist['var_daily_pct']:.2%} per day).
    - **Expected Shortfall**: when losses do exceed VaR, the average loss is **{curr_sym}{var_hist['cvar_scaled_amount']:,.0f}**.
    - **Model risk**: the {len(method_names)} models range from **{curr_sym}{min(var_values):,.0f}** to **{curr_sym}{max(var_values):,.0f}**, a spread of {max(var_values) / min(var_values) - 1:.0%}.
    - **Backtest**: {backtest_summary}
    - **CAGR** {stats['cagr']:.2%} · **Sharpe** {stats['sharpe_ratio']:.2f} · **Sortino** {stats['sortino_ratio']:.2f} (risk-free {risk_free_pct:.2f}%, assumption) · **Max Drawdown** {stats['max_drawdown']:.2%}
    """)

# -------------------------------------------------------------
# PORTFOLIO TAB: RISK DECOMPOSITION
# -------------------------------------------------------------
if is_portfolio:
    with tab_portfolio:
        st.markdown(f"### 🧩 Where the Portfolio's Risk Comes From ({cl_label}, {holding_period}-day)")
        sample_note = (f"**{alignment['limiting_ticker']}** has the shortest history and limits the sample to {alignment['common_days']} common "
                       f"trading days ({alignment['days_dropped']} days dropped from the longest history)." if alignment["days_dropped"] > 0 else
                       f"All holdings share the full sample of {alignment['common_days']} trading days.")
        st.caption(f"{len(weights)} holdings · {rebalance_mode.lower()} · {sample_note}")
        if rebalance_mode == BUY_AND_HOLD:
            st.caption("Buy-and-hold: the decomposition applies today's drifted weights to the return history, so its total can differ "
                       "from the headline Historical VaR, which follows the actual buy-and-hold path.")
        with st.expander("History available for each holding"):
            spans = alignment["spans"].copy()
            for col in ("First Date", "Last Date"):
                spans[col] = spans[col].dt.strftime("%d %b %Y")
            st.dataframe(spans, width="stretch", hide_index=True)

        basis = st.radio("Decomposition basis", DECOMPOSITION_BASES, index=0, horizontal=True, key="decomposition_basis",
                         help="Historical ES: exact tail-conditional (Euler) allocation. Historical VaR: Euler allocation estimated from the days "
                              "closest to the VaR quantile. Parametric VaR: normal-distribution Euler allocation.")
        decomposition = risk_decomposition(asset_returns, weights_now, investment_amount, confidence_level, holding_period, basis)
        dtable = decomposition["table"]
        dtable.insert(1, "Company", dtable["Ticker"].map(asset_names))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Portfolio {basis}", f"{curr_sym}{decomposition['total']:,.0f}", help="The components below add up exactly to this total.")
        m2.metric("Sum of Standalone", f"{curr_sym}{decomposition['standalone_sum']:,.0f}")
        m3.metric("Diversification Benefit", f"{curr_sym}{decomposition['diversification_benefit']:,.0f}",
                  f"-{decomposition['diversification_benefit'] / decomposition['standalone_sum']:.0%} risk", delta_color="inverse")
        m4.metric("Average Pairwise Correlation", f"{diversification['average_correlation']:.2f}")

        comp_display = dtable.copy()
        comp_display["Weight"] = comp_display["Weight"].map(lambda x: f"{x:.1%}")
        comp_display["Annualized Volatility"] = comp_display["Annualized Volatility"].map(lambda x: f"{x:.1%}")
        for col in ("Standalone", "Component", "Incremental"):
            comp_display[col] = comp_display[col].map(lambda x: f"{curr_sym}{x:,.0f}")
        comp_display["Contribution %"] = comp_display["Contribution %"].map(lambda x: f"{x:.1%}")
        comp_display["Risk / Weight"] = comp_display["Risk / Weight"].map(lambda x: f"{x:.2f}×")
        comp_display.loc[len(comp_display)] = ["TOTAL", "", "100%", "", f"{curr_sym}{decomposition['standalone_sum']:,.0f}",
                                               f"{curr_sym}{decomposition['total']:,.0f}", "100%", "", ""]
        st.dataframe(comp_display.rename(columns={"Standalone": f"Standalone {basis}", "Component": f"Component {basis}",
                                                  "Incremental": f"Incremental {basis}"}), width="stretch", hide_index=True)
        st.caption(f"**Standalone**: each holding's {basis} on its own. **Component**: its share of the portfolio {basis} (Euler allocation); "
                   "the components add up exactly to the total. **Incremental**: portfolio risk with the holding minus without it "
                   "(other positions unchanged). **Risk / Weight** above 1× means the holding adds more risk than capital.")

        col_pr1, col_pr2 = st.columns(2)
        with col_pr1:
            df_share = dtable.melt(id_vars="Ticker", value_vars=["Weight", "Contribution %"], var_name="Measure", value_name="Share")
            df_share["Measure"] = df_share["Measure"].replace({"Weight": "Capital weight", "Contribution %": "Risk contribution"})
            fig_share = px.bar(df_share, x="Ticker", y="Share", color="Measure", barmode="group",
                               title=f"Capital Weight vs Share of {basis}", template="plotly_dark",
                               color_discrete_sequence=["#63B3ED", "#FC8181"])
            fig_share.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_share, width="stretch")
        with col_pr2:
            fig_corr = px.imshow(correlation, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                 title="Correlation of Daily Returns", template="plotly_dark")
            st.plotly_chart(fig_corr, width="stretch")

        top = dtable.loc[dtable["Contribution %"].idxmax()]
        overweight_risk = dtable.loc[dtable["Risk / Weight"] > 1.1, "Ticker"].tolist()
        pairs = correlation.where(~np.eye(len(correlation), dtype=bool)).stack()
        most_correlated = pairs.idxmax()
        st.info(f"""
        📋 **Portfolio Insights** ({basis}):
        - **{top['Ticker']}** is the largest risk contributor: **{top['Contribution %']:.0%}** of portfolio {basis} from a **{top['Weight']:.0%}** weight.
        - {('Holdings adding more risk than their capital weight: **' + ', '.join(overweight_risk) + '**.') if overweight_risk else 'No holding adds much more risk than its capital weight.'}
        - The most correlated pair is **{most_correlated[0]} / {most_correlated[1]}** ({pairs.max():.2f}); together they diversify the least.
        - Diversification cuts {basis} by **{curr_sym}{decomposition['diversification_benefit']:,.0f}** compared with adding up each position's risk separately.
        """)

        st.markdown("#### 🔧 What-if: change a weight or add a holding")
        st.caption("The chosen holding is set to the new weight; the others keep their relative sizes and are scaled to fill the rest.")
        add_label = "➕ Add a new ticker"
        wi1, wi2, wi3 = st.columns(3)
        choice = wi1.selectbox("Holding", list(weights.index) + [add_label], key="what_if_holding")
        new_ticker = wi2.text_input("New ticker", key="what_if_ticker", disabled=choice != add_label,
                                    placeholder="e.g. INFY.NS").strip().upper()
        target = new_ticker if choice == add_label else choice
        current_pct = float(weights_now.get(target, 0.0) * 100)
        new_pct = wi3.number_input("New weight (%)", min_value=0.0, max_value=100.0, value=round(current_pct, 1), step=1.0,
                                   key=f"what_if_weight_{target}", disabled=not target)

        if target:
            what_if_returns, what_if_error = asset_returns, None
            if target not in asset_returns.columns:
                extra = cached_fetch(target, period=period_input)
                if not extra["success"]:
                    what_if_error = extra["error"]
                elif extra["currency"] != currency:
                    what_if_error = f"{target} trades in {extra['currency']}, the portfolio in {currency}; mixing currencies is not supported."
                else:
                    frames = {t: res["df"] for t, res in fetched.items()}
                    frames[target] = extra["df"]
                    what_if_returns = align_asset_returns(frames)
                    if len(what_if_returns) < len(asset_returns):
                        st.caption(f"{target} has a shorter history: both columns below use the {len(what_if_returns)} days all holdings share.")
            if what_if_error:
                st.warning(f"What-if: {what_if_error}")
            else:
                try:
                    wi_table = what_if(what_if_returns, weights_now, {target: new_pct / 100}, investment_amount, confidence_level, holding_period)
                    wi_weights = what_if_weights(weights_now, {target: new_pct / 100})
                    wi_display = wi_table.copy()
                    for col in ("Current", "What-if", "Change"):
                        wi_display[col] = wi_display[col].map(lambda x: f"{'-' if x < 0 else ''}{curr_sym}{abs(x):,.0f}")
                    wi_display["Change %"] = wi_table["Change %"].map(lambda x: f"{x:+.1%}")
                    st.dataframe(wi_display, width="stretch", hide_index=True)
                    st.caption("What-if weights: " + ", ".join(f"{t} {w:.1%}" for t, w in wi_weights.items()))
                except ValueError as exc:
                    st.warning(f"What-if: {exc}")

# -------------------------------------------------------------
# TAB 2: PRICE & VOLATILITY
# -------------------------------------------------------------
with tab2:
    st.markdown("### 📈 Price & Volatility Analytics")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        fig_price = px.line(df, x="Date", y="Close",
                            title="Portfolio Value Index (start = 100)" if is_portfolio else f"{company_name} ({symbol}) Closing Price History",
                            labels={"Close": "Index Level" if is_portfolio else f"Price ({curr_sym})"},
                            template="plotly_dark", color_discrete_sequence=["#4FD1C5"])
        st.plotly_chart(fig_price, width="stretch")

    with col_p2:
        fig_returns = px.line(df, x="Date", y="Returns", title="Daily Percentage Returns",
                              labels={"Returns": "Daily Return"}, template="plotly_dark", color_discrete_sequence=["#63B3ED"])
        fig_returns.add_hline(y=0, line_dash="dash", line_color="#A0AEC0")
        st.plotly_chart(fig_returns, width="stretch")

    st.markdown("#### 🌊 Annualized Volatility: 30-Day Rolling vs EWMA")
    ewma_sigma, _ = ewma_volatility(returns)
    fig_vol = px.area(df, x="Date", y="Rolling_30d_Vol", title="Volatility clustering: calm and stressed periods",
                      labels={"Rolling_30d_Vol": "Annualized Volatility"}, template="plotly_dark", color_discrete_sequence=["#ED8936"])
    fig_vol.add_scatter(x=df.loc[ewma_sigma.index, "Date"], y=ewma_sigma * np.sqrt(252), mode="lines",
                        name="EWMA (λ = 0.94)", line=dict(color="#68D391", width=1.5))
    fig_vol.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_vol, width="stretch")

# -------------------------------------------------------------
# TAB 3: STRESS TESTING
# -------------------------------------------------------------
with tab3:
    st.markdown("### ⚡ Stress Testing")
    position_word = "portfolio" if is_portfolio else "stock"
    sb1, sb2, sb3 = st.columns(3)
    sb1.metric(f"Beta vs {benchmark_name}", f"{beta:.2f}" if np.isfinite(beta) else "n/a",
               help="OLS beta on all days in the lookback window.")
    sb2.metric("Downside beta", f"{beta_down:.2f}" if np.isfinite(beta_down) else "n/a",
               help=f"Beta measured only on the {benchmark_name}'s worst 10% of days in the lookback window.")
    sb3.metric("Beta used for proxies", f"{proxy_beta:.2f}")
    if not np.isfinite(beta):
        st.warning(f"Could not download {benchmark_name} data to estimate beta, so β = 1 is assumed.")

    st.markdown("#### 🏛️ Historical crises")
    if stress_table is None:
        st.warning(f"Could not download {benchmark_name} history, so the crisis scenarios cannot be measured.")
    else:
        st.caption(f"Each crisis is the {benchmark_name}'s largest peak-to-trough fall inside the window in `stress_scenarios.csv`, measured from "
                   f"downloaded prices. **{REPLAY}** uses the {position_word}'s actual return between the same two dates. "
                   f"**{PROXY}** is used only when the {position_word} has no prices for that period: market fall × downside beta "
                   f"({proxy_beta:.2f}). Market recovery counts trading days from the trough until the index regained its peak.")
        covered = stress_table[stress_table["Method"] != NO_MARKET_DATA].copy()
        shown = pd.DataFrame({
            "Scenario": covered["Scenario"],
            "Peak → Trough": covered["Peak"].dt.strftime("%d %b %Y") + " → " + covered["Trough"].dt.strftime("%d %b %Y"),
            f"{benchmark_name} Fall": covered["Market Drawdown"].map(lambda x: f"{x:.1%}"),
            "Market Recovery": covered["Market Recovery (days)"].map(lambda d: "not yet" if pd.isna(d) else f"{int(d)} days"),
            "Method": covered["Method"],
            f"{position_word.title()} Return": covered["Position Return"].map(lambda x: f"{x:+.1%}"),
            "P&L": covered["P&L"].map(lambda x: f"{'-' if x < 0 else '+'}{curr_sym}{abs(x):,.0f}"),
            "Value After": covered["Post-Shock Value"].map(lambda x: f"{curr_sym}{x:,.0f}"),
        })
        st.dataframe(shown, width="stretch", hide_index=True)
        if is_portfolio and (covered["Method"] == PROXY).any():
            limited_by = f" (limited by {alignment['limiting_ticker']})" if alignment["days_dropped"] > 0 else ""
            st.caption(f"β-proxy rows appear because the portfolio replay needs prices for every holding, and the holdings only share "
                       f"prices from {position_history.index[0]:%d %b %Y}{limited_by}.")
        missing = stress_table.loc[stress_table["Method"] == NO_MARKET_DATA, "Scenario"].tolist()
        if missing:
            st.caption(f"Not shown, because the {benchmark_name} history does not cover them: " + ", ".join(missing) + ".")

        if len(covered):
            chart = covered.melt(id_vars="Scenario", value_vars=["Market Drawdown", "Position Return"], var_name="Series", value_name="Return")
            chart["Series"] = chart["Series"].replace({"Market Drawdown": benchmark_name, "Position Return": position_word.title()})
            fig_stress = px.bar(chart, x="Scenario", y="Return", color="Series", barmode="group",
                                title=f"Crisis Peak-to-Trough: {benchmark_name} vs {position_word.title()}", template="plotly_dark",
                                color_discrete_sequence=["#A0AEC0", "#FC8181"])
            fig_stress.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
            st.plotly_chart(fig_stress, width="stretch")

    st.markdown("#### 🎚️ Custom market move")
    custom_shock_pct = st.slider(f"{benchmark_name} move (%)", min_value=-60.0, max_value=40.0, value=-20.0, step=1.0)
    move_beta = proxy_beta if custom_shock_pct < 0 else beta_used
    custom_move = max(custom_shock_pct / 100.0 * move_beta, -1.0)
    custom_pnl = investment_amount * custom_move
    target_name = "the portfolio" if is_portfolio else symbol
    st.warning(f"💡 A **{custom_shock_pct:+.0f}%** {benchmark_name} move implies **{custom_move:+.1%}** for {target_name} "
               f"({'downside' if custom_shock_pct < 0 else 'normal'} β = {move_beta:.2f}): P&L of **{'-' if custom_pnl < 0 else '+'}{curr_sym}{abs(custom_pnl):,.0f}**, "
               f"leaving **{curr_sym}{investment_amount + custom_pnl:,.0f}**. Falls use the downside beta, rises the normal beta.")

    st.markdown(f"#### 🌪️ Volatility shock ({cl_label}, {holding_period}-day)")
    st.caption("VaR and ES re-run with every return's distance from the mean multiplied by k: r′ = μ + k·(r − μ).")
    vs_display = vol_shock_table.copy()
    for col in ("Historical VaR", "Historical ES", "Normal VaR", "Normal ES"):
        vs_display[col] = vs_display[col].map(lambda x: f"{curr_sym}{x:,.0f}")
    st.dataframe(vs_display, width="stretch", hide_index=True)

    st.markdown("#### 📉 Worst Actual Losses in the Sample")
    st.caption(f"The {position_word}'s own worst compounded return over each horizon. Compare these with VaR: a VaR well below them understates real tail risk.")
    worst_display = worst_df.copy()
    worst_display["Worst Return"] = worst_display["Worst Return"].map(lambda x: f"{x:.2%}")
    worst_display["Loss"] = worst_display["Loss"].map(lambda x: f"{curr_sym}{x:,.0f}")
    worst_display["Window End"] = df.loc[worst_df["Window End"], "Date"].dt.strftime("%d %b %Y").to_numpy()
    st.dataframe(worst_display, width="stretch", hide_index=True)

# -------------------------------------------------------------
# TAB 4: BACKTESTING
# -------------------------------------------------------------
with tab4:
    st.markdown(f"### 🔬 Out-of-Sample Backtest of Every Model ({cl_label} VaR)")
    st.caption(f"Each day's VaR is estimated from the previous {backtest_window} trading days only, then compared with that day's actual return. "
               "**Kupiec** tests whether the breach count is right; **Christoffersen independence** tests whether breaches cluster together; "
               "**conditional coverage** combines both. A model passes if every p-value is at least 0.05. "
               "Passing models are ranked by **tick loss** (average quantile loss; lower is better). A higher p-value is not evidence of a better model. "
               "**ES test** (McNeil-Frey): on breach days, (loss − ES)/σ should average zero; a low p-value means ES is too small. "
               "**Monte Carlo** is scored but never recommended: its 1-day forecast is the GARCH-t model plus simulation noise. Its value is in multi-day paths.")
    if rolling is not None and rolling["garch_refits"]:
        st.caption(f"GARCH(1,1)-t and Monte Carlo were refitted every 20 trading days ({rolling['garch_refits']} fit{'' if rolling['garch_refits'] == 1 else 's'}, on up to 1,000 past days each); "
                   f"{rolling['garch_fallbacks']} fit{'' if rolling['garch_fallbacks'] == 1 else 's'} did not converge and used EWMA for that block.")

if backtest_table is None:
    with tab4:
        st.warning(f"Not enough data to backtest: the first {backtest_window} returns are needed to estimate the model, "
                   f"which leaves {test_days} test days. Choose a 5y or max lookback.")
else:
    with tab4:
        if low_power:
            st.warning(f"Not enough data for a meaningful backtest at {cl_label}: {test_days} out-of-sample day{"" if test_days == 1 else "s"}, "
                       f"{required_days} needed (only {test_days * (1 - confidence_level):.1f} breaches expected). "
                       f"Verdicts are shown as {LOW_POWER}; choose a 5y or max lookback.")
        elif recommendation["status"] == "none_pass":
            st.warning(f"No model passes all three tests. **{recommended_model}** has the lowest tick loss, but its breach "
                       "pattern is still statistically off.")
        else:
            st.success(f"Recommended model: **{recommended_model}** (lowest tick loss among the models that pass all three tests).")

        bt_display = backtest_table.copy()
        bt_display["Expected Breaches"] = bt_display["Expected Breaches"].map(lambda x: f"{x:.1f}")
        bt_display["Breach Rate"] = bt_display["Breach Rate"].map(lambda x: f"{x:.2%}")
        bt_display["Avg VaR"] = bt_display["Avg VaR"].map(lambda x: f"{x:.2%}")
        bt_display["Tick Loss (bp)"] = bt_display["Tick Loss"].map(lambda x: f"{x * 1e4:.3f}")
        for col in ("Kupiec p-value", "Independence p-value", "Conditional Coverage p-value", "ES p-value"):
            bt_display[col] = bt_display[col].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "n/a")
        verdict_colors = {"PASS": "color: #68D391; font-weight: 600", "FAIL": "color: #FC8181; font-weight: 600",
                          LOW_POWER: "color: #718096; font-style: italic", "TOO FEW BREACHES": "color: #718096; font-style: italic"}
        st.dataframe(
            bt_display[["Method", "Actual Breaches", "Expected Breaches", "Verdict", "Tick Loss (bp)", "ES Test", "ES p-value",
                        "Kupiec p-value", "Independence p-value", "Conditional Coverage p-value", "Breach Rate",
                        "Back-to-Back Breaches", "Traffic Light", "Avg VaR"]].style.map(lambda v: verdict_colors.get(v, ""), subset=["Verdict", "ES Test"]),
            width="stretch", hide_index=True
        )

        st.markdown("---")
        model_options = list(forecasts.columns)
        default_model = recommended_model if recommended_model in model_options else model_options[0]
        selected_model = st.selectbox("Inspect a model", options=model_options, index=model_options.index(default_model))
        detail = perform_kupiec_backtest(returns, forecasts[selected_model], confidence_level)

        col_b1, col_b2 = st.columns([1, 1.6])

        with col_b1:
            traffic_badge = detail["traffic_light"]
            badge_class = "badge-green" if "GREEN" in traffic_badge else "badge-yellow" if "YELLOW" in traffic_badge else "badge-red"
            st.markdown(f"#### Basel status: <span class=\"{badge_class}\">{traffic_badge}</span>", unsafe_allow_html=True)
            st.write(detail["status_desc"])
            st.table(pd.DataFrame([
                {"Metric": "Test Days (T)", "Value": f"{detail['total_observations']}"},
                {"Metric": "Expected Breaches (α × T)", "Value": f"{detail['expected_failures']:.1f}"},
                {"Metric": "Actual Breaches", "Value": f"{detail['actual_breaches']}"},
                {"Metric": "Kupiec LR (crit. 3.841)", "Value": f"{detail['lr_stat']:.3f}"},
                {"Metric": "Kupiec p-value", "Value": f"{detail['p_value']:.3f}"},
                {"Metric": "Binomial P(X ≤ breaches)", "Value": f"{detail['cumulative_prob']:.2%}"},
            ]).set_index("Metric"))

        with col_b2:
            df_breaches = df.loc[forecasts.index, ["Date", "Returns"]].copy()
            df_breaches["VaR_Threshold"] = -forecasts[selected_model]
            df_breaches = df_breaches.dropna()
            df_breaches["Is_Breach"] = df_breaches["Returns"] < df_breaches["VaR_Threshold"]

            fig_breach = px.scatter(df_breaches, x="Date", y="Returns", color="Is_Breach",
                                    title=f"Daily Returns vs Rolling {selected_model} VaR",
                                    color_discrete_map={True: "#FC8181", False: "#4FD1C5"},
                                    labels={"Is_Breach": "VaR Breach"}, template="plotly_dark")
            fig_breach.add_scatter(x=df_breaches["Date"], y=df_breaches["VaR_Threshold"], mode="lines",
                                   line=dict(color="#F6AD55", dash="dash"), name="Rolling VaR")
            fig_breach.update_layout(yaxis_tickformat=".1%")
            st.plotly_chart(fig_breach, width="stretch")

# -------------------------------------------------------------
# TAB 5: EXPORT & REPORTS
# -------------------------------------------------------------
with tab5:
    st.markdown("### 📥 Download Excel Risk Report")
    st.write("A formatted `.xlsx` workbook with every model's VaR and ES, the out-of-sample backtest, β-adjusted stress tests, worst historical losses and the raw price data.")

    excel_bytes = generate_excel_var_report(
        symbol=symbol,
        company_name=company_name,
        currency=currency,
        investment=investment_amount,
        confidence_level=confidence_level,
        holding_period=holding_period,
        df_data=df,
        var_by_level={m: {cl: var_by_level[cl][m] for cl in confidence_levels} for m in method_names},
        backtest_table=backtest_table,
        backtest_window=backtest_window,
        stress_table=stress_table,
        vol_shock_table=vol_shock_table,
        beta_down=beta_down,
        worst_df=worst_df,
        benchmark_name=benchmark_name,
        beta=beta,
        recommendation=recommendation,
        data_note=data_note,
        risk_free_rate=risk_free_pct / 100,
        portfolio={"decomposition": decomposition, "diversification": diversification, "correlation": correlation,
                   "alignment": alignment} if is_portfolio else None
    )

    st.download_button(
        label=f"📥 Download Excel Report ({symbol})",
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
