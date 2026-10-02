"""Backtesting tab: VaR and ES tests for every model and the recommended model."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from var_calculator import LOW_POWER, perform_kupiec_backtest


def render(ctx, tab4):
    """Backtesting tab: VaR and ES tests for every model and the recommended model."""
    backtest_table = ctx.backtest_table
    backtest_window = ctx.backtest_window
    cl_label = ctx.cl_label
    confidence_level = ctx.confidence_level
    df = ctx.df
    forecasts = ctx.forecasts
    low_power = ctx.low_power
    recommendation = ctx.recommendation
    recommended_model = ctx.recommended_model
    required_days = ctx.required_days
    returns = ctx.returns
    rolling = ctx.rolling
    test_days = ctx.test_days

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
                st.warning(f"Not enough data for a meaningful backtest at {cl_label}: {test_days} out-of-sample day{'' if test_days == 1 else 's'}, "
                           f"{required_days} needed (only {test_days * (1 - confidence_level):.1f} breaches expected). "
                           f"Verdicts are shown as {LOW_POWER}; choose a 5y or max lookback.")
            elif recommendation["status"] == "none_pass":
                st.warning(f"No model passes all three tests. **{recommended_model}** has the lowest tick loss, but its breach "
                           "pattern is still statistically off.")
            elif recommendation["status"] == "var_only":
                st.warning(f"Every model that passes the three VaR tests fails the ES test. **{recommended_model}** has the "
                           "lowest tick loss among them and is shown as the headline, but it is not recommended: its ES "
                           "understates the losses beyond VaR.")
            else:
                st.success(f"Recommended model: **{recommended_model}** (lowest tick loss among the models that pass the "
                           "three VaR tests and the ES test).")

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
