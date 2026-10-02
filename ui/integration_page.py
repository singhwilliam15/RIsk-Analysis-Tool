"""
Integrated Stress page (Phase 6): one scenario through every pillar, against the siloed sum, and the reverse
stress test. Layout: headline → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.integration_layer import REVERSE_LOSS_KEY, REVERSE_LOSSES

METHODOLOGY = "docs/methodology.md, section 13"
PARTS = (("Market Loss", "#3182CE"), ("+ Liquidity", "#DD6B20"), ("+ Credit", "#805AD5"), ("+ Events", "#E53E3E"))


def render(ctx):
    curr, inv = ctx.curr_sym, ctx.investment_amount

    def money(v):
        return f"{curr}{v:,.0f}" if v is not None and np.isfinite(v) else "not available"

    st.subheader(f"🔗 Integrated stress: {ctx.company_name}")
    linked = ctx.linked
    if linked.empty:
        st.info("No crisis window could be measured (no long benchmark history), so the linked engine has no scenario.")
    else:
        worst = linked.loc[linked["Linked Total"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Worst linked loss ({worst['Scenario']})", money(worst["Linked Total"]), f"{-worst['Linked Total'] / inv:.1%}",
                  delta_color="off")
        c2.metric("Siloed sum of the separate pillars", money(worst["Siloed Sum"]))
        c3.metric("Interaction effect", money(worst["Interaction"]), f"{worst['Interaction'] / inv:+.1%} of value", delta_color="off")
        st.caption("The interaction effect is what reading each pillar on its own page misses: in a crisis the price fall, "
                   "thinner volume, higher volatility, weaker credit and forced pledge selling arrive together and feed "
                   "each other. A negative interaction means the standalone pillar figures were more conservative than "
                   "the linked effects in that scenario.")

        fig = go.Figure()
        for col, color in PARTS:
            fig.add_bar(name=col.replace("+ ", ""), x=linked["Scenario"], y=linked[col], marker_color=color)
        fig.add_scatter(name="Siloed sum", x=linked["Scenario"], y=linked["Siloed Sum"], mode="markers",
                        marker=dict(symbol="diamond", size=10, color="#F7FAFC"))
        fig.update_layout(barmode="stack", template="plotly_dark", height=420, margin=dict(t=30),
                          yaxis_title=f"Loss ({curr})", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        shown = linked.drop(columns=["detail"]).copy()
        for col in ("Market Loss", "+ Liquidity", "+ Credit", "+ Events", "Linked Total", "Siloed Sum", "Interaction"):
            shown[col] = shown[col].map(money)
        shown["Market Move"] = shown["Market Move"].map(lambda v: f"{v:.1%}")
        st.dataframe(shown, hide_index=True, width="stretch")
        s = ctx.linked_standalone
        st.caption(f"Siloed sum = scenario market loss + liquidity cost today ({money(s['liquidity'])}) + credit expected "
                   f"loss today ({money(s['credit'])}) + event ES gap today ({money(s['events'])}). Linked: volume falls to "
                   "that crisis's measured level (never above normal), volatility to its crisis level; Merton is re-solved "
                   "on the shocked equity; a pledge margin call adds forced-selling impact; a banded stock that fell by its "
                   "band is assumed frozen for its exit-freeze scenario. Custom shocks use downside beta and the median "
                   "crisis-to-normal volatility ratio.")
        with st.expander("Per-holding detail, worst scenario"):
            detail = worst["detail"].copy()
            for col in ("Market Loss", "Liquidity", "Credit", "Events", "Total"):
                detail[col] = detail[col].map(money)
            detail["Return"] = detail["Return"].map(lambda v: f"{v:.1%}")
            st.dataframe(detail, hide_index=True, width="stretch")
        crisis = ctx.conc.get("crisis") if isinstance(ctx.conc, dict) else None
        if crisis is not None:
            t = crisis["table"].set_index("Sample")
            if "Crisis windows" in t.index and np.isfinite(t.loc["Crisis windows", "ES"]):
                st.caption(f"Crisis correlation (Pillar 4): after the shock, 1-day ES under crisis correlations is "
                           f"{money(t.loc['Crisis windows', 'ES'])} against {money(t.loc['Full sample', 'ES'])} normally "
                           "(on the full common history); this shapes the next day's tail rather than the scenario loss.")

    _reverse(ctx, money)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Second-round effects beyond one step**: forced selling that triggers further circuits, rating downgrades and fund redemptions over weeks.
- **New kinds of crisis**: the linked engine replays the past windows' mix of price, volume and volatility moves.
- **Tier escalation**: a holding whose stressed DD falls into the High tier would carry a larger jump risk afterwards; that is not added here.
- **Counterparties, funding and FX** (Phase 8, optional) are not modelled.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _reverse(ctx, money):
    st.markdown("### ↩️ Reverse stress test")
    st.radio("Loss over one month", REVERSE_LOSSES, index=1, key=REVERSE_LOSS_KEY, horizontal=True, format_func=lambda v: f"{v:.0%}")
    rv = ctx.reverse
    full = rv["full"]
    rows = [{"Ticker": t, "Most plausible 1-month move (full sample)": f"{x:+.1%}"} for t, x in zip(ctx.positions["Ticker"], full["shock"])]
    if "crisis" in rv:
        for row, x in zip(rows, rv["crisis"]["shock"]):
            row["Under crisis correlation"] = f"{x:+.1%}"
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    p = full["plausibility"]
    st.caption(f"x* = −L·Σw / (wᵀΣw) minimises xᵀΣ⁻¹x subject to a {rv['loss']:.0%} loss (Σ = 21 × daily covariance). "
               f"Mahalanobis distance {np.sqrt(full['d2']):.2f}: a move at least this far out has probability "
               f"{p['normal']:.2%} under a multivariate normal and {p['student_t']:.2%} under a multivariate Student-t "
               f"(ν = {ctx.var_selected['Student-t']['degrees_of_freedom']:.1f}).")
    lk = rv["linked"]
    st.markdown(f"Run through the linked engine, this scenario loses **{money(lk['Linked Total'])}**: market "
                f"{money(lk['Market Loss'])} + liquidity {money(lk['+ Liquidity'])} + credit {money(lk['+ Credit'])} + "
                f"events {money(lk['+ Events'])}.")
    macro = rv.get("macro")
    if macro:
        shock = ", ".join(f"**{k} {v:+.1%}**" for k, v in macro["shock"].items())
        st.markdown(f"**Macro space**: the most plausible way to lose {rv['loss']:.0%} in a month is {shock}.")
        a = macro["analogue"]
        if a:
            st.markdown(f"Nearest historical analogue: **{a['start']:%d %b %Y} to {a['end']:%d %b %Y}** (macro moves "
                        + ", ".join(f"{k} {v:+.1%}" for k, v in a["moves"].items()) +
                        f"; the portfolio actually returned {a['portfolio_return']:+.1%}).")
        st.caption(f"Portfolio daily returns regressed on the macro factors over the last {min(504, macro['days'])} common "
                   f"days (R² {macro['r2']:.2f}); the shock is y* = −L·Σb/(bᵀΣb) in factor space, and the analogue is the "
                   "past 21-day window closest to it in Mahalanobis distance. USD/INR up means the rupee weakened. "
                   + (f"Not available from Yahoo: {', '.join(rv['missing_macro'])}." if rv["missing_macro"] else ""))
    else:
        st.info("Macro reverse stress needs at least two macro series (Yahoo Finance); they could not be loaded.")
