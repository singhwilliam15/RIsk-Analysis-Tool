"""Model Comparison tab: VaR/ES by model, charts and interpretation."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm, t as student_t
from ui.formatting import MODEL_COLORS, pct_label


def render(ctx, tab1):
    """Model Comparison tab: VaR/ES by model, charts and interpretation."""
    backtest_summary = ctx.backtest_summary
    cl_label = ctx.cl_label
    company_name = ctx.company_name
    confidence_level = ctx.confidence_level
    confidence_levels = ctx.confidence_levels
    curr_sym = ctx.curr_sym
    excess_kurtosis = ctx.excess_kurtosis
    garch_text = ctx.garch_text
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    jarque_bera = ctx.jarque_bera
    jarque_bera_p = ctx.jarque_bera_p
    method_names = ctx.method_names
    num_sims = ctx.num_sims
    returns = ctx.returns
    risk_free_pct = ctx.risk_free_pct
    skewness = ctx.skewness
    stats = ctx.stats
    var_by_level = ctx.var_by_level
    var_cf = ctx.var_cf
    var_garch = ctx.var_garch
    var_hist = ctx.var_hist
    var_selected = ctx.var_selected

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
