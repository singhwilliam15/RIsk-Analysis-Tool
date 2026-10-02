"""
Integrated Stress page (Phase 6): one scenario through every pillar with price feedback, its Shapley split and the cross-pillar
interaction, jump to default, and the reverse
stress test. Layout: headline → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.integration_layer import REVERSE_LOSS_KEY, REVERSE_LOSSES

METHODOLOGY = "docs/methodology.md, section 13"
PARTS = (("Market", "#3182CE"), ("Liquidity", "#DD6B20"), ("Credit", "#805AD5"), ("Events", "#E53E3E"))
MONEY_COLUMNS = ("Market Loss", "Market", "Liquidity", "Credit", "Events", "Linked Total", "Interaction")


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
        c2.metric("Interaction (cross effect)", money(worst["Interaction"]), f"{worst['Interaction'] / inv:+.2%} of value",
                  delta_color="off")
        c3.metric("Feedback rounds", f"{int(worst['Rounds'])}" + ("" if worst["Converged"] else " (not settled)"))
        st.caption("Interaction = the linked loss − (the market loss alone + each link switched on alone with the market). "
                   "It is what only appears when the pillars act together: forced pledge selling moves the price, which "
                   "triggers a circuit or more margin calls. The bars split the linked loss across market, liquidity, "
                   "credit and events by exact Shapley values, so the parts add up to the total.")
        if len(ctx.linked_active) < 2:
            st.info("Only the liquidity link can act on these holdings: no pledge data are loaded (Events page), and "
                    "credit adds no loss for an equity holder. With one active link the interaction is zero by "
                    "construction; feedback from your own exit still moves the price (see the detail).")

        fig = go.Figure()
        for col, color in PARTS:
            fig.add_bar(name=col, x=linked["Scenario"], y=linked[col], marker_color=color)
        fig.add_scatter(name="Market loss alone", x=linked["Scenario"], y=linked["Market Loss"], mode="markers",
                        marker=dict(symbol="diamond", size=10, color="#F7FAFC"))
        fig.update_layout(barmode="relative", template="plotly_dark", height=420, margin=dict(t=30),
                          yaxis_title=f"Loss ({curr})", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        shown = linked.drop(columns=["detail"]).copy()
        for col in MONEY_COLUMNS:
            shown[col] = shown[col].map(money)
        shown["Market Move"] = shown["Market Move"].map(lambda v: f"{v:.1%}")
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption("Each round: lenders sell pledged shares to restore cover once the price is below the margin-call "
                   "trigger; that selling and your own exit lower the price by the permanent share "
                   f"({ctx.permanent_share:.0%}, an assumption) of square-root impact at the crisis's volume (never above "
                   "normal) and volatility; a banded stock whose fall reaches its band is locked for its exit-freeze "
                   "scenario (in a replay, only for the days beyond the lower circuits already in the replayed path). "
                   "The rounds stop when the price moves less than 0.1% (at most 20). Your loss is the fall to the "
                   "final price plus the temporary part of impact and the spread on your exit. Market Loss is the plain "
                   "scenario loss with no links. Custom shocks use downside beta and the median crisis-to-normal "
                   "volatility ratio.")
        with st.expander("Per-holding detail, worst scenario"):
            detail = worst["detail"].copy()
            for col in ("Market Loss", "Market", "Liquidity", "Credit", "Events", "Total", "Interaction",
                        "JTD Loss (10% recovery)", "JTD Loss (0% recovery)"):
                detail[col] = detail[col].map(money)
            for col in ("Return", "Feedback Fall"):
                detail[col] = detail[col].map(lambda v: f"{v:.2%}")
            detail["Stressed PD"] = detail["Stressed PD"].map(lambda v: f"{v:.3%}" if np.isfinite(v) else "-")
            st.dataframe(detail, hide_index=True, width="stretch")
            st.caption("Feedback Fall: how far the price fell beyond the market shock because of selling and circuits. "
                       "Credit is a signal for equity holders: Merton re-solved at the final equity value (Stressed DD; "
                       "the PD is risk-neutral and model-implied). It adds no loss, because the equity price already "
                       "carries the default risk.")
        _jump_to_default(linked, money)
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
- **Contagion between holdings and other investors**: the feedback runs within each stock. Fund redemptions, rating downgrades and other holders' selling over weeks are not modelled.
- **Credit as a loss**: for equity holders it is a signal and a jump-to-default scenario; bonds and loans (with a credit loss) come later.
- **New kinds of crisis**: the linked engine replays the past windows' mix of price, volume and volatility moves.
- **Tier escalation**: a holding whose stressed DD falls into the High tier would carry a larger jump risk afterwards; that is not added here.
- **Counterparties, funding and FX** (Phase 8, optional) are not modelled.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _jump_to_default(linked, money):
    """Holdings whose stressed distance to default falls below the threshold, in any scenario."""
    rows = []
    for r in linked.to_dict("records"):
        d = r["detail"]
        for h in d[d["JTD Loss (0% recovery)"].notna()].to_dict("records"):
            rows.append({"Scenario": r["Scenario"], "Ticker": h["Ticker"], "Stressed DD": f"{h['Stressed DD']:.2f}",
                         "Stressed PD (risk-neutral)": f"{h['Stressed PD']:.2%}",
                         "Loss if it defaults": f"{money(h['JTD Loss (10% recovery)'])} to {money(h['JTD Loss (0% recovery)'])}"})
    st.markdown("#### Jump to default")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("A separate scenario, not added to the linked loss: the whole position falls to a 10% or 0% recovery "
                   "(assumptions). The PD next to it is Merton's risk-neutral, model-implied probability.")
    else:
        st.caption("No holding's stressed distance to default falls below the threshold (Credit assumptions) in any "
                   "scenario. Banks and NBFCs are not modelled by Merton.")


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
    prob, share = full["probability"], full["plausibility"]
    nu = ctx.var_selected["Student-t"]["degrees_of_freedom"]

    def pct(p):
        """Small probabilities keep two significant figures (0.011%, not 0.01%)."""
        if not np.isfinite(p):
            return "not available"
        return f"{p:.2%}" if p >= 0.01 else f"{p * 100:.2g}%"

    def chance(key):
        p, years = prob[key], prob[f"{key}_years"]
        if not np.isfinite(p):
            return "not available"
        return f"{pct(p)} a month (about once every {years:,.0f} years)" if years >= 1.5 else f"{pct(p)} a month"

    st.markdown(f"**Chance of losing at least {rv['loss']:.0%} in a month:** {chance('normal')} under a normal; "
                f"{chance('student_t')} under a Student-t (ν = {nu:.1f}).")

    crisis_note = ""
    if "crisis" in rv:
        cp = rv["crisis"]["probability"]
        crisis_note = f" With crisis correlations the chance is {pct(cp['normal'])} (normal) and {pct(cp['student_t'])} (Student-t)."
    st.caption(f"P(portfolio return ≤ −{rv['loss']:.0%}) = Φ(−L/σ_p) under a normal, and the univariate t tail at the "
               f"same distance (scaled for the t's variance) under a Student-t; σ_p = {full['portfolio_sigma']:.2%} a month."
               + crisis_note +
               f" **Assumption:** ν is fitted to daily returns; one-month sums are closer to normal, so the Student-t "
               f"figure is an upper-end estimate. x* = −L·Σw / (wᵀΣw) minimises xᵀΣ⁻¹x subject to the loss "
               f"(Σ = 21 × daily covariance). Its Mahalanobis distance is {np.sqrt(full['d2']):.2f}; the share of "
               f"outcomes at least this extreme in any direction (gains included, so not the chance of the loss) is "
               f"{pct(share['normal'])} under a normal and {pct(share['student_t'])} under a Student-t.")
    lk = rv["linked"]
    st.markdown(f"Run through the linked engine, this scenario loses **{money(lk['Linked Total'])}** (Shapley split: market "
                f"{money(lk['Market'])}, liquidity {money(lk['Liquidity'])}, credit {money(lk['Credit'])}, "
                f"events {money(lk['Events'])}; interaction {money(lk['Interaction'])}).")
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
