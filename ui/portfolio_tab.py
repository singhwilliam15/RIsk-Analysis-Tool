"""Portfolio Risk tab: risk decomposition, correlations and the what-if panel."""

import numpy as np
import plotly.express as px
import streamlit as st
from portfolio import (
    align_asset_returns,
    BUY_AND_HOLD,
    DECOMPOSITION_BASES,
    risk_decomposition,
    what_if,
    what_if_weights,
)
from ui.cache import cached_fetch


def render(ctx, tab_portfolio):
    """Portfolio Risk tab: risk decomposition, correlations and the what-if panel."""
    alignment = getattr(ctx, 'alignment', None)
    asset_names = getattr(ctx, 'asset_names', None)
    asset_returns = getattr(ctx, 'asset_returns', None)
    cl_label = ctx.cl_label
    confidence_level = ctx.confidence_level
    correlation = getattr(ctx, 'correlation', None)
    curr_sym = ctx.curr_sym
    currency = ctx.currency
    diversification = getattr(ctx, 'diversification', None)
    fetched = getattr(ctx, 'fetched', None)
    holding_period = ctx.holding_period
    investment_amount = ctx.investment_amount
    period_input = ctx.period_input
    rebalance_mode = getattr(ctx, 'rebalance_mode', None)
    weights = getattr(ctx, 'weights', None)
    weights_now = getattr(ctx, 'weights_now', None)

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
    return decomposition
