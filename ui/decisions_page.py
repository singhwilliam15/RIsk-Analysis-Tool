"""
Decisions page (Phase 6): what changed, what to do, limits, and the CRO memo.
Layout: headline → detail → what this misses → methodology.
"""

import copy
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import decisions as Dz
import memo
from ui.integration_layer import CHANGE_MONTHS, CHANGE_MONTHS_KEY, LIMITS_KEY, SNAPSHOT_KEY

METHODOLOGY = "docs/methodology.md, section 14"
LIGHTS = {"green": "🟢", "amber": "🟠", "red": "🔴"}
PLAYER_LABELS = {"positions": "Positions", "volatility": "Volatility", "correlation": "Correlation", "window": "Data window",
                 "model": "Model"}


def render(ctx):
    curr = ctx.curr_sym

    def money(v):
        return f"{curr}{v:,.0f}" if v is not None and np.isfinite(v) else "not available"

    st.subheader(f"🧭 Decisions: {ctx.company_name}")
    _limits(ctx)
    _change(ctx, money)
    _trades(ctx, money)
    _memo(ctx)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Trades are measured on history**: the ES after a trade uses the same past returns; it ignores costs, taxes and market impact of the trade itself.
- **The hedge is measurement only**: no basis risk, margin, roll cost or lot sizes.
- **Shapley attribution needs a model of the past**: "window" collects everything about the sample's shape that volatility and correlation do not; with an uploaded snapshot it also contains the holdings' different histories.
- **Limits are examples**, not recommendations: set them for your own mandate.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _limits(ctx):
    st.markdown("### 🚦 Limits")
    t = ctx.limits_table
    shown = pd.DataFrame({
        "Limit": t["Limit"], "Status": t["Status"].map(lambda s: f"{LIGHTS.get(s, '⚪')} {s}"),
        "Value": [v if isinstance(v, str) else (f"{v:.2%}" if u == "fraction" else f"{v:,.2f}") if np.isfinite(v) else "not available"
                  for v, u in zip(t["Value"], t["Unit"])],
        "Limit value": [v if isinstance(v, str) else (f"{v:.0%}" if u == "fraction" else f"{v:g}") for v, u in zip(t["Limit Value"], t["Unit"])],
        "Utilisation": t["Utilisation"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "-")})
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Green below 80% of the limit, amber 80–100%, red above. The minimum trust grade compares the worst headline "
               "grade (A = 1 … D = 4) with the limit's grade.")
    with st.expander("Edit limits (this session), or download them as JSON"):
        limits = ctx.limits
        rows = pd.DataFrame([{"Key": k, "Limit": v["label"], "Value": str(v["value"]), "Unit": v["unit"]} for k, v in limits["limits"].items()])
        edited = st.data_editor(rows, hide_index=True, width="stretch", disabled=["Key", "Limit", "Unit"], key="limits_editor")
        if st.button("Apply limits", key="apply_limits"):
            new = copy.deepcopy(limits)
            for r in edited.to_dict("records"):
                new["limits"][r["Key"]]["value"] = r["Value"].strip().upper() if r["Unit"] == "grade" else float(r["Value"])
            st.session_state[LIMITS_KEY] = new
            st.rerun()
        st.download_button("Download limits.json", json.dumps(limits, indent=2).encode(), file_name="limits.json",
                           mime="application/json")


def _change(ctx, money):
    st.markdown("### 🔄 What changed in the risk")
    c1, c2 = st.columns([1, 2])
    c1.selectbox("Compare today with", CHANGE_MONTHS, index=2, key=CHANGE_MONTHS_KEY, format_func=lambda m: f"{m} month(s) ago")
    upload = c2.file_uploader("…or a saved snapshot (JSON)", type="json", key="snapshot_upload")
    file_id = getattr(upload, "file_id", None)
    if file_id != st.session_state.get("snapshot_upload_id"):
        st.session_state["snapshot_upload_id"] = file_id
        st.session_state[SNAPSHOT_KEY] = Dz.snapshot_from_json(upload.getvalue().decode()) if upload else None
        st.rerun()
    st.download_button("Save today's snapshot (JSON)", Dz.snapshot_to_json(ctx.risk_snapshot_now).encode(),
                       file_name=f"risk_snapshot_{ctx.prices_as_of:%Y%m%d}.json", mime="application/json")
    change = ctx.risk_change
    if change is None:
        st.info("Not enough history before that date to compare.")
        return
    contrib = change["contributions"]
    labels = ["ES then"] + [PLAYER_LABELS[k] for k in contrib] + ["ES now"]
    values = [change["start"]] + list(contrib.values()) + [change["end"]]
    fig = go.Figure(go.Waterfall(x=labels, y=values, measure=["absolute"] + ["relative"] * len(contrib) + ["total"],
                                 text=[money(v) for v in values], textposition="outside",
                                 connector={"line": {"color": "#718096"}}))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(t=30), yaxis_title=f"1-day ES ({ctx.curr_sym})")
    st.plotly_chart(fig, width="stretch")
    old = ctx.risk_change_old
    st.caption(f"{ctx.cl_label} 1-day ES on {money(ctx.investment_amount)}: snapshot of {old['as_of']:%d %b %Y} ({old['model']}) "
               f"vs {ctx.prices_as_of:%d %b %Y} ({ctx.risk_snapshot_now['model']}). Exact Shapley split over all 32 orderings "
               "of positions, volatility, correlation, data window and model; the parts add up to the change."
               + (f" Not in both snapshots: {', '.join(change['dropped'])}." if change["dropped"] else ""))


def _trades(ctx, money):
    st.markdown("### 🛠️ Best risk-reducing trades")
    comp = ctx.es_components
    total = comp.sum()
    weights = ctx.positions.set_index("Ticker")["Weight"]
    st.dataframe(pd.DataFrame({"Ticker": comp.index, "Weight": weights.loc[comp.index].map("{:.0%}".format).to_numpy(),
                               "ES contribution": comp.map(money).to_numpy(),
                               "Share of ES": (comp / total).map("{:.0%}".format).to_numpy(),
                               "Marginal ES (per 1% of value)": (comp / weights.loc[comp.index] / 100).map(money).to_numpy()}),
                 hide_index=True, width="stretch")
    st.caption("Historical ES split by holding (Euler, tail-conditional); the parts add up to the portfolio's historical ES.")
    trades = ctx.trades
    if len(trades):
        inv = ctx.investment_amount
        st.dataframe(pd.DataFrame({
            "Trade": trades["Trade"], "ES change": (trades["ES Change"] * inv).map(lambda v: f"{ctx.curr_sym}{v:+,.0f}"),
            "ES change %": (trades["ES Change"] / trades["ES Before"]).map("{:+.1%}".format),
            "Days to sell 50%": trades["Days to Liquidate 50%"].map(lambda v: f"{v:,.2f}"),
            "Weighted PD": trades["Weighted PD"].map(lambda v: f"{v:.3%}" if np.isfinite(v) else "not available"),
            "Value in Elevated/High tier": trades["Value in Elevated/High"].map("{:.0%}".format)}), hide_index=True, width="stretch")
        st.caption("Every sale of 5% of the portfolio into cash and every 5% switch between holdings is tried. Shown: the "
                   "best sale into cash and the two best switches (a switch keeps the money invested, so it shows where "
                   "diversification helps), each with its effect on the other pillars.")
    h = ctx.hedge
    st.markdown(f"**Index hedge (measurement only)**: shorting index futures worth **{h['ratio']:.0%}** of the value "
                f"minimises historical ES, from {money(h['es_before'] * ctx.investment_amount)} to "
                f"{money(h['es_after'] * ctx.investment_amount)} ({h['days']} days; no basis, margin or roll costs).")


def _memo(ctx):
    st.markdown("### 📝 One-page CRO memo")
    content = ctx.memo_content
    md = memo.markdown(content)
    c1, c2 = st.columns(2)
    c1.download_button("Download memo (PDF)", memo.pdf(content), file_name=f"CRO_memo_{ctx.prices_as_of:%Y%m%d}.pdf",
                       mime="application/pdf", type="primary")
    c2.download_button("Download memo (Markdown)", md.encode(), file_name=f"CRO_memo_{ctx.prices_as_of:%Y%m%d}.md",
                       mime="text/markdown")
    with st.expander("Preview"):
        st.markdown(md)
