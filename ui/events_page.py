"""
Event & Governance page (Pillar 5). Layout shared by every pillar page:
headline metrics (with grade) → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import events as E
from disclosures import as_of_stamp

METHODOLOGY = "docs/methodology.md, section 12"
TIER_ICONS = {E.LOW: "🟢 Low", E.ELEVATED: "🟠 Elevated", E.HIGH: "🔴 High"}
SIGNAL_LABELS = {"pledge": "Promoter pledge", "surveillance": "ASM / GSM", "fo_ban": "F&O ban", "rating": "Rating actions",
                 "auditor": "Auditor events", "merton": "Credit (Merton DD trend; RBI PCA for banks)", "circuit": "Circuit history", "group": "Group tag"}


def _signal_text(name, s):
    if s is None:
        return "not available"
    if name == "pledge":
        change = f", {s['change_4q_pp']:+.1f} pp in 4 quarters" if np.isfinite(s["change_4q_pp"]) else ""
        return f"{s['pledged_pct_of_promoter']:.1f}% of promoter holding ({s['quarter_end']:%b %Y}){change}"
    if name == "surveillance":
        return f"{s['measure']} {s['stage_text']} since {s['since']:%d %b %Y}" if s["measure"] else "not under surveillance"
    if name == "fo_ban":
        return f"banned on {s['last_ban']:%d %b %Y}" if s["last_ban"] is not None else "no ban in window"
    if name == "rating":
        parts = [f"{s['downgrades']} downgrade(s) in 12 months"] + (["negative watch"] if s["negative_watch"] else [])
        return f"{s['latest'] or 'no rating on file'}; " + ", ".join(parts)
    if name == "auditor":
        return ", ".join(s["events"]) if s["events"] else "no events in 24 months"
    if name == "merton" and "pca_band" in s:
        return f"RBI PCA band {s['pca_band']}" + (f" ({s['pca_detail']})" if s["pca_detail"] else "") + (
            f"; {len(s['warnings'])} early warning(s)" if s["warnings"] else "")
    if name == "merton":
        trend = f" (6 months ago {s['dd_6m_ago']:.2f})" if np.isfinite(s["dd_6m_ago"]) else ""
        return f"DD {s['dd']:.2f}{trend}"
    if name == "circuit":
        return f"{s['lower_circuit_days']} lower-circuit days, longest run {s['longest_run']}"
    if name == "group":
        return f"{s['group']} ({s['same_group_holdings']} holding(s))"
    return str(s)


def render(ctx):
    curr = ctx.curr_sym

    def money(v):
        return f"{curr}{v:,.0f}" if v is not None and np.isfinite(v) else "not available"

    m = ctx.event_metrics
    st.subheader(f"⚠️ Event & governance: {ctx.company_name}")
    stamp = as_of_stamp(ctx.disclosures["files"])
    if stamp:
        st.caption(stamp + ". Official NSE files (see Overview → Disclosure data for each file and its source).")
    if ctx.event_missing:
        st.warning(f"Disclosure data not loaded: {', '.join(ctx.event_missing)}. Those signals show 'not available' and "
                   "never count as 'no risk'; each tier says how many signals it rests on. Load the official files "
                   "(Overview → Disclosure data, data/disclosures/README.md).")
    cols = st.columns(4)
    cards = [(m["event_es"], money), (m["gap"], money), (m["flagged"], lambda v: f"{v:.0%}"), (m["high"], lambda v: f"{v:.0f}")]
    for col, (metric, fmt) in zip(cols, cards):
        col.metric(metric.name, fmt(metric.value))
        col.caption(f"{metric.range_text(fmt)} · grade **{metric.grade}**")
        col.caption("; ".join(metric.reasons) or "no deductions")

    _panel(ctx)
    _margin(ctx, money)
    _overlay(ctx, money)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Fraud or misstatement not yet in any disclosure.** The signals only see what has been filed or rated.
- **Regulatory action without warning** (SEBI orders, sector bans, tax demands) and litigation.
- **Promoter debt outside pledges**: non-disposal undertakings and loans at holding companies are only partly in the pledge data.
- **Jump probabilities and sizes are assumptions**, calibrated on price-based proxies for the tiers (methodology §12.4), not on the disclosure tiers themselves; the table above shows how much they matter.
- **Rules are coarse.** A tier is a flag for attention, not a probability; two holdings in the same tier can differ a lot.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _panel(ctx):
    st.markdown("### 🚨 Early-warning panel")
    rows = []
    for t, signals in ctx.event_signals.items():
        tier = ctx.event_tiers[t]
        row = {"Ticker": t, "Tier": TIER_ICONS[tier["tier"]], "Basis": tier["basis"]}
        row.update({SIGNAL_LABELS[k]: _signal_text(k, signals[k]) for k in E.SIGNALS})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for t, tier in ctx.event_tiers.items():
        if tier["fired"]:
            st.markdown(f"**{t}** — " + "; ".join(f"{level}: {text}" for level, text in tier["fired"]))
    st.caption(f"As of {ctx.prices_as_of:%d %b %Y}: only disclosures public by then are used (pledges: disclosure date, or "
               "quarter end + 21 days). Rules and thresholds: config/event_rules.json (assumptions).")


def _margin(ctx, money):
    st.markdown("### 🔗 Promoter-pledge margin calls")
    rows = []
    for t, mc in ctx.event_margin.items():
        rows.append({"Ticker": t, "Price fall that triggers a margin call": f"{mc['trigger_fall']:.0%}",
                     "Trigger price": f"{ctx.curr_sym}{mc['trigger_price']:,.2f}",
                     "Loan (assumed)": money(mc["loan"]),
                     "Selling if all pledged shares are sold": f"{mc['days_all']:,.1f} days of volume" if np.isfinite(mc["days_all"]) else "not available",
                     "Selling to restore cover": f"{mc['days_restore']:,.1f} days of volume" if np.isfinite(mc["days_restore"]) else "not available"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Assumptions: the loan was sized at today's price at {ctx.initial_cover:g}× cover, and lenders can invoke the "
               f"pledge when cover falls to {ctx.trigger_cover:g}×, i.e. after a {1 - ctx.trigger_cover / ctx.initial_cover:.0%} "
               "fall. Selling is in days of 60-day average volume. Pledged quantities come from the pledge file; without it, "
               "only the trigger is shown.")


def _overlay(ctx, money):
    st.markdown("### ⚡ Jump overlay on Expected Shortfall")
    o, inv = ctx.event_overlay, ctx.investment_amount
    fig = go.Figure(go.Bar(x=["Standard ES", "Event-adjusted ES"], y=[o["es"] * inv, o["event_es"] * inv],
                           marker_color=["#63B3ED", "#FC8181"], text=[money(o["es"] * inv), money(o["event_es"] * inv)],
                           textposition="outside"))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30), yaxis_title=f"1-day ES ({ctx.curr_sym})")
    st.plotly_chart(fig, width="stretch")
    rows = [{"Ticker": t, "Tier": ctx.event_tiers[t]["tier"], "Jump probability (a day)": f"{p:.2%}",
             "Chance of a jump in a year": f"{E.annual_probability(p):.0%}", "Jump size": f"{j:.0%}",
             "Share of the ES gap": money(o["by_holding"].get(t, 0.0) * inv)} for t, (p, j) in ctx.event_jumps.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Base: the {ctx.event_base['model']} one-day distribution of the {'portfolio' if ctx.is_portfolio else 'stock'} "
               "return. Each holding can jump by its tier's J with its tier's probability; every combination of jumps is "
               "included exactly (closed-form Student-t tail, no simulation). Shares of the gap: each holding's jump alone, "
               "scaled to add up to the total. 1-day figures at the selected confidence level.")
    _sensitivity(ctx, money)


def _sensitivity(ctx, money):
    sens = ctx.event_sensitivity
    table = sens["table"]
    shown = pd.DataFrame({f"J = {j:.0%}": [money(v) for v in table[j]] for j in table.columns})
    shown.insert(0, "p a day (a year)", [f"{p:.2%} ({E.annual_probability(p):.0%})" for p in table.index])
    who = ", ".join(sens["targets"])
    st.markdown("**How much the jump assumptions matter**: event-adjusted ES if "
                + (f"{who} (no holding is Elevated or High, so this is hypothetical: the largest holding)" if sens["hypothetical"]
                   else f"the Elevated/High holding(s) {who}") + " had jump probability p and size J")
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Every other holding has no jump. A daily p compounds: 0.5% a day is a 72% chance of at least one jump "
               "in a year. The defaults and how they were calibrated: docs/methodology.md §12.4.")
