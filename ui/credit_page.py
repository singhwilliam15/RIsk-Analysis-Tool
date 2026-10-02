"""
Credit page (Pillar 3). Layout shared by every pillar page:
headline metrics (with grade) → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import credit as C
from ui.credit_layer import EQUITY_VOL_CHOICES
from ui.formatting import MODEL_COLORS

METHODOLOGY = "docs/methodology.md, section 10"
BANK_METRICS = ("GNPA %", "NNPA %", "Capital adequacy (CAR) %", "CASA %", "Net interest margin %")


def _pct(v, digits=2) -> str:
    return f"{v:.{digits}%}" if v is not None and np.isfinite(v) else "not available"


def _pd(v) -> str:
    """PDs of strong firms are astronomically small; below 0.001% the exact figure means nothing."""
    if v is None or not np.isfinite(v):
        return "not available"
    return "below 0.001%" if v < 1e-5 else f"{v:.3%}"


def _num(v, digits=2) -> str:
    return f"{v:,.{digits}f}" if v is not None and np.isfinite(v) else "not available"


def render(ctx):
    curr = ctx.curr_sym

    def money(v):
        if v is None or not np.isfinite(v):
            return "not available"
        return f"under {curr}1" if 0 <= v < 0.5 else f"{curr}{v:,.0f}"

    m, view = ctx.credit_metrics, ctx.credit_view
    st.subheader(f"🏦 Credit: {ctx.company_name}")
    st.caption(f"As of {ctx.prices_as_of:%d %b %Y}, using only statements public by then. Merton equity volatility: "
               f"**{ctx.equity_vol_choice}** (change it under Credit assumptions). The page leads with the **distance to "
               "default** and where it sits among a reference universe, and with the **agencies' published default "
               "rates**. Merton's PD is a model-implied, risk-neutral probability: for strong firms it is astronomically "
               "small and its exact value means little.")

    universe = view["dd_universe"]
    weakest = m["weakest_dd"]
    t = view["table"]
    pct = t.loc[t["DD"] == weakest.value, "DD Percentile"]
    pct = pct.iloc[0] if len(pct) else np.nan
    cols = st.columns(4)
    cols[0].metric(weakest.name, _num(weakest.value),
                   f"percentile {pct:.0%} of {universe['size']} reference stocks" if np.isfinite(pct) else None, delta_color="off")
    cols[0].caption(f"{weakest.range_text(lambda v: _num(v))} · grade **{weakest.grade}**")
    cols[0].caption("; ".join(weakest.reasons) or "no deductions")
    cols[1].metric("Agency 1-year default rate (value-weighted)", _pct(view["agency_weighted_pd"], 2))
    cols[1].caption(f"Real-world, historical: each holding's latest long-term rating mapped to its agency's published "
                    f"average default rate. Covers {view['agency_coverage']:.0%} of the value (rated holdings).")
    for col, (metric, fmt, label) in zip(cols[2:], [(m["weighted_pd"], _pd, "Merton PD (risk-neutral, model-implied)"),
                                                     (m["weakest_altman"], lambda v: _num(v), m["weakest_altman"].name)]):
        col.metric(label, fmt(metric.value))
        col.caption(f"{metric.range_text(fmt)} · grade **{metric.grade}**")
        col.caption("; ".join(metric.reasons) or "no deductions")
    st.caption(f"{view['coverage']:.0%} of the portfolio value is modelled by Merton. Credit-implied expected loss (Σ Merton "
               f"PD × value, LGD 100%): {money(m['expected_loss'].value)}. Reference universe for the DD percentile: "
               + (f"{universe['size']} NSE stocks (the presets, Jaiprakash Power and the five most-pledged mid/small caps; "
                  f"data/credit/dd_universe.csv, built {universe['built_on']})." if universe["size"] else
                  "not built yet (scripts/build_dd_universe.py)."))

    _portfolio_table(ctx, money)
    for ticker, res in ctx.credit_results.items():
        with st.expander(f"{ticker}: details", expanded=len(ctx.credit_results) == 1):
            if res["financial"]:
                _financial_panel(ticker, res, ctx)
            else:
                _merton(res, ctx, money)
                _altman(res)
                _ratios(res, ctx)
            _rating(res)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Off-balance-sheet debt**: guarantees, leases not on the balance sheet, supplier finance and contingent liabilities.
- **Group-company exposure**: debt of subsidiaries or sister companies that the listed company may have to support.
- **Promoter-level leverage**: loans against pledged shares sit with the promoter, not the company, but can force selling (Pillar 5).
- **Risk-neutral PD** uses the risk-free rate as the asset drift, so it overstates real-world default frequencies; and Merton assumes one debt maturity at T and lognormal assets.
- **Stale accounts**: annual statements can be up to 14 months old by the next filing; quarterly deterioration is missed.
- **Altman Z** was fitted on US firms (1968: manufacturers; Z'': emerging-market bonds); its zones are not calibrated to Indian default rates.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _portfolio_table(ctx, money):
    st.markdown("### 📋 By holding")
    t = ctx.credit_view["table"]
    st.dataframe(pd.DataFrame({
        "Ticker": t["Ticker"], "Value": t["Value"].map(money),
        "DD": t["DD"].map(_num), "DD percentile": t["DD Percentile"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "-"),
        "Agency 1-yr default rate": t["Agency PD"].map(lambda v: _pct(v, 2)),
        "Merton PD (risk-neutral)": t["PD"].map(_pd),
        "Altman": [f"{m}: {_num(s)} ({z})" if m != "not applicable" else "not applicable"
                   for m, s, z in zip(t["Altman Model"], t["Altman Score"], t["Altman Zone"])],
        "Red flags": t["Red Flags"], "Rating": t["Rating"], "Direction": t["Rating Direction"],
        "Balance sheet": t["Balance Sheet"].map(lambda d: f"FY ending {d:%b %Y}" if pd.notna(d) else "not available")}),
        hide_index=True, width="stretch")


def _merton(res, ctx, money):
    st.markdown("**Merton distance to default**")
    for note in res["notes"]:
        st.warning(note)
    if not res["merton"]:
        return
    rows = []
    for name in EQUITY_VOL_CHOICES:
        mres = res["merton"].get(name, {})
        rows.append({"Equity volatility": name, "σ_E": _pct(res["vols"].get(name), 1), "Asset value V": money(mres.get("V")),
                     "Asset volatility σ_V": _pct(mres.get("sigma_v"), 1), "DD": _num(mres.get("DD")),
                     "PD": _pd(mres.get("PD")), "Solved": "yes" if mres.get("converged") else "no"})
    kmv = res["kmv"]
    rows.append({"Equity volatility": "Iterative KMV (cross-check)", "σ_E": "from the asset path",
                 "Asset value V": money(kmv.get("V")), "Asset volatility σ_V": _pct(kmv.get("sigma_v"), 1),
                 "DD": _num(kmv.get("DD")), "PD": _pd(kmv.get("PD")),
                 "Solved": f"{kmv.get('iterations', 0)} iterations" if kmv.get("converged") else "no"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Equity value {money(res['market_cap'])} (latest close × {res['shares']:,.0f} shares); default point "
               f"{money(res['default_point'])} = short-term + {ctx.ltd_weight:g} × long-term debt from the FY ending "
               f"{res['period_end']:%b %Y}; T = {ctx.merton_horizon:g} year(s); r = {ctx.risk_free_pct:.2f}%.")
    rolling = res["rolling"]
    if len(rolling):
        fig = go.Figure(go.Scatter(x=rolling["Date"], y=rolling["DD"], mode="lines+markers", line=dict(color=MODEL_COLORS[0])))
        for change in rolling.loc[rolling["Balance Sheet"].ne(rolling["Balance Sheet"].shift()), "Date"].iloc[1:]:
            fig.add_vline(x=change, line_dash="dot", line_color="#718096")
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30), yaxis_title="Distance to default",
                          title="Month-end DD (dotted lines: a new balance sheet becomes public)")
        st.plotly_chart(fig, width="stretch")


def _altman(res):
    st.markdown("**Altman Z-scores**")
    rows = []
    for label, (score, zone), formula in (("Z (Altman 1968, manufacturers)", res["z"], "1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5"),
                                          ("Z'' (non-manufacturers, emerging markets)", res["z2"], "6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4")):
        primary = (res["primary"] == "Z") == label.startswith("Z (")
        rows.append({"Model": label + (" — primary" if primary else ""), "Score": _num(score), "Zone": zone, "Formula": formula})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    r = res.get("altman_ratios", {})
    st.caption("X1 working capital / TA " + _num(r.get("X1"), 3) + " · X2 retained earnings / TA " + _num(r.get("X2"), 3) +
               " · X3 EBIT / TA " + _num(r.get("X3"), 3) + " · X4 market equity / TL " + _num(r.get("X4_market"), 3) +
               " (Z) or book equity / TL " + _num(r.get("X4_book"), 3) + " (Z'') · X5 sales / TA " + _num(r.get("X5"), 3) +
               ". Zones: Z > 2.99 safe, 1.81–2.99 grey, < 1.81 distress; Z'' > 2.60 safe, 1.10–2.60 grey, < 1.10 distress.")
    if res["altman_missing"]:
        st.info("Not computed from partial inputs. Missing: " + ", ".join(res["altman_missing"]) +
                ". Upload the figures with the CSV template on the Overview page.")


def _ratios(res, ctx):
    st.markdown("**Credit ratios**")
    ratios = res["ratios"]
    if ratios.empty:
        st.info("No statements available.")
        return
    shown = ratios[list(C.RATIO_LABELS)].rename(columns=C.RATIO_LABELS).T
    shown.columns = [f"FY {c:%b %Y}" for c in shown.columns]
    st.dataframe(shown.map(lambda v: _num(v)), width="stretch")
    t = ctx.credit_config["red_flags"]
    if res["flags"]:
        st.warning("Red flags (latest year): " + "; ".join(res["flags"]))
    else:
        st.success("No red flags on the latest year.")
    st.caption(f"Thresholds (assumptions, config/credit_thresholds.json): debt/equity > {t['debt_to_equity_max']:g}, net debt/EBITDA > "
               f"{t['net_debt_to_ebitda_max']:g}, interest cover < {t['interest_cover_min']:g}, current ratio < {t['current_ratio_min']:g}, "
               f"quick ratio < {t['quick_ratio_min']:g}, CFO/EBITDA < {t['cfo_to_ebitda_min']:g}, negative CFO for "
               f"{t['negative_cfo_years']}+ years.")


def _rating(res):
    r = res["rating"]
    if r["rating"]:
        st.markdown(f"**Rating:** {r['rating']} ({r['agency']}), last action **{r['action']}** on {r['date']:%d %b %Y} "
                    f"(direction: {r['direction']}).")
        a = res.get("agency_pd")
        if a:
            st.caption(f"Historical 1-year default rate for the {a['category']} category: **{a['pd']:.2%}** "
                       + (f"({a['study']}, page {a['page']})" if a.get("page") else "(in default)")
                       + (f". {r['agency']}'s own study is not on file, so {a['agency_used']}'s is used; the agencies' "
                          "scales are comparable but not identical." if a["fallback"] else ".")
                       + " A real-world frequency, unlike Merton's risk-neutral PD.")
    else:
        st.caption(f"Rating: {r['direction']}. Load rating actions from the agencies' rationales "
                   "(Overview → Disclosure data → Credit rating actions).")


def _financial_panel(ticker, res, ctx):
    for note in res["notes"]:
        st.info(note)
    st.markdown("**Bank / NBFC / insurer metrics, from the annual report**")
    key = f"bank_metrics_{ticker}"
    table = st.data_editor(pd.DataFrame({"Metric": BANK_METRICS, "Value": [None] * len(BANK_METRICS),
                                         "Source (report and page)": [""] * len(BANK_METRICS)}),
                           hide_index=True, width="stretch", key=key, disabled=["Metric"])
    car = pd.to_numeric(table.loc[table["Metric"] == "Capital adequacy (CAR) %", "Value"], errors="coerce").iloc[0]
    minimum = ctx.credit_config["capital_adequacy_min_pct"]
    if np.isfinite(car):
        (st.warning if car < minimum else st.success)(
            f"Capital adequacy {car:.2f}% vs the RBI minimum of {minimum:g}% (9% CRAR + 2.5% conservation buffer).")
    st.caption("Values are typed in from the annual report and are not stored beyond this session; nothing is filled in.")
