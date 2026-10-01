"""Stress Testing tab: measured crises, custom move, volatility shock, worst losses."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from stress import NO_MARKET_DATA, PROXY, REPLAY


def render(ctx, tab3):
    """Stress Testing tab: measured crises, custom move, volatility shock, worst losses."""
    alignment = getattr(ctx, 'alignment', None)
    benchmark_name = ctx.benchmark_name
    beta = ctx.beta
    beta_down = ctx.beta_down
    beta_used = ctx.beta_used
    cl_label = ctx.cl_label
    curr_sym = ctx.curr_sym
    df = ctx.df
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    is_portfolio = ctx.is_portfolio
    position_history = ctx.position_history
    proxy_beta = ctx.proxy_beta
    stress_table = ctx.stress_table
    symbol = ctx.symbol
    vol_shock_table = ctx.vol_shock_table
    worst_df = ctx.worst_df

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
