"""
Trust page: how far the market-risk numbers can be relied on.
Layout shared by every pillar page: headline metrics (with grade) → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trust import BACKTEST_POINTS, CI_LEVEL, GRADE_BANDS, GRADE_RULES
from ui.formatting import MODEL_COLORS

METHODOLOGY = "docs/methodology.md, section 8"
LOOKBACK_INFO = ("Lookback", "Days", "Available", "From", "Note")


def render(ctx):
    curr_sym, cl_label = ctx.curr_sym, ctx.cl_label

    def money(v):
        return f"{curr_sym}{v:,.0f}" if np.isfinite(v) else "not available"

    trusted, risk = ctx.trusted, ctx.trust["model_risk"]
    headline = ctx.headline_model
    st.subheader(f"🛡️ Trust: {ctx.company_name}")
    status_note = {"recommended": " (recommended by the backtest)",
                   "none_pass": " (lowest tick loss, but no model passes every backtest)",
                   "low_power": " (too few test days to recommend a model, so Historical is shown)"}
    es_failed = (ctx.recommendation["status"] == "recommended"
                 and ctx.trust["grades"].set_index("Model").loc[headline, "Backtest"] == "fail")
    st.caption(f"{cl_label}, {ctx.holding_period}-day horizon, on {money(ctx.investment_amount)}. Headline model: "
               f"**{headline}**{status_note[ctx.recommendation['status']]}."
               + (" The recommendation uses the three VaR tests only; this model fails the ES backtest, which the "
                  "grade counts as a failed backtest." if es_failed else ""))

    # Headline metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"VaR ({headline})", money(trusted[headline]["VaR"].value),
                help=trusted[headline]["VaR"].format(money))
    col1.caption(trusted[headline]["VaR"].range_text(money))
    col2.metric(f"ES ({headline})", money(trusted[headline]["ES"].value), help=trusted[headline]["ES"].format(money))
    col2.caption(trusted[headline]["ES"].range_text(money))
    col3.metric("Trust grade", trusted[headline]["ES"].grade)
    col3.caption("; ".join(trusted[headline]["ES"].reasons) or "no deductions")
    col4.metric("Model-risk add-on", money(risk["add_on"]) if np.isfinite(risk["add_on"]) else "not defined")
    col4.caption(f"Highest passing ES − recommended ES. ES across {risk['basis']}: {money(risk['low'])} to {money(risk['high'])}.")

    _model_table(ctx, money)
    _lookback(ctx, money)
    _ghost(ctx, money)
    _grade_rules()

    st.markdown("### 🕳️ What this metric misses")
    st.markdown(f"""
- **Only estimation error.** The {CI_LEVEL:.0%} ranges show how much the number could move with a different sample from the same process. They do not cover a change of regime, a crisis unlike anything in the window, or a model that is wrong in kind; the model-dispersion and lookback panels give a feel for those.
- **The block length is itself an estimate** (Politis-White: {ctx.trust_ranges['block_length']:.1f} days, {ctx.trust_ranges['block_method']}). A longer block would widen the ranges when volatility clusters strongly.
- **GARCH and Monte Carlo ranges cover parameter uncertainty only**, from the asymptotic covariance; they assume that covariance is accurate, which is weak when α is near zero.
- **Percentile bootstrap ranges for Historical ES are too narrow** at small tail sizes: on simulated normal data (500 days, 95%) they contained the true ES about 84% of the time instead of 90%.
- **Grades are rule-based**, not statistical tests. The rules and thresholds are assumptions, listed below and in {METHODOLOGY}.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _model_table(ctx, money):
    st.markdown("### 📏 Every model: value, 90% range and grade")
    grades = ctx.trust["grades"].set_index("Model")
    rows = []
    for row in ctx.ranges_table.to_dict("records"):
        model = row["Model"]
        rows.append({"Model": model, "VaR": money(row["VaR"]), "VaR 90% range": f"{money(row['VaR Low'])} – {money(row['VaR High'])}",
                     "ES": money(row["ES"]), "ES 90% range": f"{money(row['ES Low'])} – {money(row['ES High'])}",
                     "Backtest": grades.loc[model, "Backtest"], "Grade": grades.loc[model, "Grade"],
                     "Range method": row["Range Method"], "Reasons": grades.loc[model, "Reasons"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Ranges are the 5th–95th percentiles of {ctx.trust_ranges['models']['Historical']['resamples']} bootstrap "
               "resamples (Student-t: refitted on 200; GARCH: parameter draws). Multi-day ranges follow each model's own scaling rule.")


def _lookback(ctx, money):
    st.markdown("### 🔭 Lookback sensitivity: ES on different windows")
    table = ctx.lookback
    available = table[table["Available"]]
    models = [c for c in available.columns if c not in LOOKBACK_INFO]
    if available.empty:
        st.info("The history is shorter than one year, so lookback sensitivity is not available.")
        return
    fig = go.Figure()
    for i, model in enumerate(models):
        fig.add_scatter(x=available["Lookback"], y=available[model], mode="lines+markers", name=model,
                        line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)]))
    fig.update_layout(template="plotly_dark", yaxis_title=f"ES ({ctx.curr_sym})", xaxis_title="Lookback window",
                      height=380, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    shown = table.assign(**{m: table[m].map(lambda v: money(v) if pd.notna(v) else "not available") for m in models})
    shown["Days"] = shown["Days"].astype(int)
    st.dataframe(shown.drop(columns=["Available", "From"], errors="ignore"), hide_index=True, width="stretch")
    st.caption("The last 252, 504 and 1,260 trading days and the full history, taken from the long price history used for "
               "the crisis replay; a window longer than the history is not shown rather than shortened. A window with "
               "reversing spikes (likely bad prices) is flagged: its figures are distorted, but the data are not altered.")


def _ghost(ctx, money):
    st.markdown("### 👻 Ghost effect in Historical VaR")
    ghost = ctx.ghost
    st.markdown(f"Historical VaR uses the last **{ghost['window_days']}** returns. "
                f"**{len(ghost['leaving'])}** tail loss(es) among the oldest {ghost['horizon']} days leave the window within "
                f"{ghost['horizon']} trading days. Without those days, Historical VaR would be **{money(ghost['var_after'])}** "
                f"instead of **{money(ghost['var_now'])}**, if the coming days add nothing to the tail.")
    if len(ghost["leaving"]):
        st.dataframe(ghost["leaving"].assign(Date=ghost["leaving"]["Date"].dt.strftime("%d %b %Y"),
                                             Return=ghost["leaving"]["Return"].map("{:.2%}".format)),
                     hide_index=True, width="stretch")
    path, events = ghost["rolling_var"], ghost["events"]
    if len(path):
        fig = go.Figure()
        fig.add_scatter(x=path.index, y=path.to_numpy(), mode="lines", name="Rolling Historical VaR", line=dict(color="#63B3ED"))
        exits = events[events["Cause"] == "exit"]
        if len(exits):
            fig.add_scatter(x=exits["Date"], y=exits["VaR After"], mode="markers", name="Jump when an old loss left",
                            marker=dict(color="#FC8181", size=8))
        fig.update_layout(template="plotly_dark", height=340, margin=dict(t=30), yaxis_title=f"VaR ({ctx.curr_sym})")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Over the full history, {ghost['exits']} day-to-day VaR moves above {ghost['jump']:.0%} (an assumption) "
                   "happened only because a large loss dropped out of the window: the ghost effect. "
                   f"{int((events['Cause'] == 'entry').sum())} were new losses arriving.")
        if len(exits):
            st.dataframe(exits.tail(10).assign(**{
                "Date": exits.tail(10)["Date"].dt.strftime("%d %b %Y"),
                "Date Leaving": exits.tail(10)["Date Leaving"].dt.strftime("%d %b %Y"),
                "VaR Before": exits.tail(10)["VaR Before"].map(money), "VaR After": exits.tail(10)["VaR After"].map(money),
                "Change": exits.tail(10)["Change"].map("{:+.0%}".format),
                "Return Leaving": exits.tail(10)["Return Leaving"].map("{:.2%}".format),
                "Return Entering": exits.tail(10)["Return Entering"].map("{:.2%}".format)}),
                hide_index=True, width="stretch")


def _grade_rules():
    with st.expander("How grades are set"):
        r = GRADE_RULES
        st.markdown(f"""
| Rule | 0 points | 1 point | 2 points |
| --- | --- | --- | --- |
| ES 90% range width ÷ ES | ≤ {r['range_width'][0]:.0%} | ≤ {r['range_width'][1]:.0%} | more |
| ES spread across passing models ÷ recommended ES | ≤ {r['dispersion'][0]:.0%} | ≤ {r['dispersion'][1]:.0%} | more |
| Backtest | pass | low power / not tested | fail (VaR tests or ES test) |
| Lowest data-quality score | ≥ {r['data_quality'][0]} | ≥ {r['data_quality'][1]} | lower |
| Daily returns | ≥ {r['sample'][0]} | ≥ {r['sample'][1]} | fewer |
| Share of inputs that are assumptions | ≤ {r['assumptions'][0]:.0%} | ≤ {r['assumptions'][1]:.0%} | more |

Total ≤ {GRADE_BANDS[0][0]}: **A** · ≤ {GRADE_BANDS[1][0]}: **B** · ≤ {GRADE_BANDS[2][0]}: **C** · more: **D**. A model that fails its
backtest is graded at best C; a data-quality score below 50 gives D. A missing input costs one point.
Backtest points: {", ".join(f"{k} {v}" for k, v in BACKTEST_POINTS.items())}.
""")
