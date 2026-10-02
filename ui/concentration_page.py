"""
Concentration & Factors page (Pillar 4). Layout shared by every pillar page:
headline metrics (with grade) → detail → what this misses → methodology.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.formatting import MODEL_COLORS

METHODOLOGY = "docs/methodology.md, section 11"
FACTOR_LABELS = {"MKT_RF": "Market", "SMB": "Size (SMB)", "HML": "Value (HML)", "MOM": "Momentum", "MKT": "Benchmark",
                 "Specific": "Specific (stock-only)"}


def _num(v, digits=2):
    return f"{v:,.{digits}f}" if v is not None and np.isfinite(v) else "not available"


def _pct(v, digits=0):
    return f"{v:.{digits}%}" if v is not None and np.isfinite(v) else "not available"


def render(ctx):
    res, m, meta = ctx.conc, ctx.conc_metrics, ctx.conc_factor_meta
    curr = ctx.curr_sym
    st.subheader(f"🧭 Concentration & factors: {ctx.company_name}")
    if meta:
        st.caption(f"Factor data: {meta['source']}, {meta['first_date']} to **{meta['end_date']}** (downloaded "
                   f"{meta['downloaded_on']}). Cite: {meta['citation']}")
    elif ctx.conc_region is None:
        st.warning("The holdings mix Indian and US stocks; one factor set cannot cover both without FX conversion.")
    else:
        st.warning("No factor data: run `python scripts/refresh_factor_data.py --iima-release YYYY-MM` (see the script for "
                   "manual steps). The single-index model on the benchmark is used instead.")

    shown = [("enb", lambda v: _num(v, 2)), ("first_pc", _pct), ("factor_share", _pct), ("kept", _pct)]
    cols = st.columns(max(1, sum(k in m for k, _ in shown)))
    for col, (key, fmt) in zip(cols, [(k, f) for k, f in shown if k in m]):
        metric = m[key]
        col.metric(metric.name, fmt(metric.value))
        col.caption(f"{metric.range_text(fmt)} · grade **{metric.grade}**")
        col.caption("; ".join(metric.reasons) or "no deductions")

    _factor_table(ctx, res)
    _factor_risk(ctx, res, curr)
    if len(ctx.conc_weights) > 1:
        _concentration(ctx, res)
        _crisis(ctx, res, curr)
    else:
        st.info("A single stock is 100% concentrated by definition (HHI = 1, one bet); the concentration and "
                "crisis-correlation sections apply to portfolios.")
    _rolling(res)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Crowding and common ownership.** Stocks held by the same funds can fall together when those funds sell, beyond what factor loadings show.
- **Group-company links.** Several holdings from one business group (e.g. Tata, Adani, Bajaj) share governance and funding risk that no return factor captures; Pillar 5 adds a manual group flag.
- **Factor regimes change.** Betas and factor premia move over time (see the rolling betas); a one-year estimate can miss a shift.
- **Indian factor data lag.** The IIMA library is released with a delay, so the most recent months are not in the regression.
- **Normal ES for crisis correlation.** It isolates the effect of correlation; it is not a forecast of crisis losses (see Market → Stress Testing).
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _factor_table(ctx, res):
    st.markdown(f"### 🧬 Factor exposures ({res['model']})")
    rows = []
    for t, fit in res["fits"].items():
        if fit is None:
            rows.append({"Ticker": t, "Days": 0, "Note": "too few days overlap"})
            continue
        row = {"Ticker": t, "Days": fit["nobs"], "From": f"{fit['start']:%d %b %Y}", "To": f"{fit['end']:%d %b %Y}"}
        for name, beta in fit["betas"].items():
            row[FACTOR_LABELS.get(name, name)] = f"{beta:+.2f} (t {fit['t_nw'][name]:+.1f})"
        row["Alpha, annual"] = f"{fit['alpha_annual']:+.1%} (t {fit['t_nw']['alpha']:+.1f})"
        row["R²"] = f"{fit['r2']:.2f}"
        row["Factor share of variance"] = _pct(fit["factor_var"] / (fit["factor_var"] + fit["resid_var"]))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    lags = next((f["lags"] for f in res["fits"].values() if f), 0)
    st.caption("Excess daily return regressed on the factors over the days both exist. t-statistics use Newey-West "
               f"standard errors ({lags} lags). |t| above about 2 means the exposure is unlikely to be zero.")


def _factor_risk(ctx, res, curr):
    fr = res["factor_risk"]
    if fr is None:
        return
    st.markdown("### 🧮 Where the portfolio's risk comes from")
    parts = fr["shares"].rename(index=FACTOR_LABELS)
    fig = go.Figure(go.Bar(x=parts.index, y=parts.to_numpy(), marker_color=MODEL_COLORS[:len(parts)],
                           text=[f"{v:.0%}" for v in parts.to_numpy()], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(t=30), yaxis_tickformat=".0%",
                      yaxis_title="Share of variance")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({"Source": parts.index, "Share of variance": parts.map(lambda v: f"{v:.1%}"),
                               f"Parametric VaR part ({ctx.cl_label}, 1-day)": fr["var_parts"].map(lambda v: f"{curr}{v:,.0f}").to_numpy(),
                               "Portfolio beta": [f"{fr['betas'].get(k, np.nan):+.2f}" if k in fr["betas"] else "-"
                                                  for k in fr["shares"].index]}),
                 hide_index=True, width="stretch")
    st.caption(f"Euler split: factor k contributes b_k·(Σ_f b)_k and specific risk Σ wᵢ²·σ²(εᵢ); the parts add up exactly "
               f"to the model variance. A negative share means that exposure offsets the others (a hedge). Total parametric "
               f"VaR {curr}{fr['var_total']:,.0f} (zero mean).")


def _concentration(ctx, res):
    st.markdown("### 🎯 Concentration")
    c1, c2, c3 = st.columns(3)
    c1.metric("HHI", _num(res["hhi"], 3))
    c2.metric("Effective number of holdings", _num(1 / res["hhi"], 1))
    c3.metric("Holdings", f"{len(ctx.conc_weights)}")
    sectors = res["sectors"]
    st.dataframe(sectors.assign(Weight=sectors["Weight"].map("{:.0%}".format),
                                **{"Risk Share": sectors["Risk Share"].map("{:.0%}".format)}),
                 hide_index=True, width="stretch")
    holding = res["holding_risk"]
    st.caption("Risk share = each holding's Euler share of portfolio variance, summed by sector: "
               + ", ".join(f"{t} {v:.0%} of risk vs {ctx.conc_weights[t]:.0%} of value" for t, v in holding.items()) + ".")
    pca = res["pca"]
    fig = go.Figure(go.Bar(x=[f"PC{i + 1}" for i in range(len(pca["portfolio_shares"]))], y=pca["portfolio_shares"],
                           marker_color="#63B3ED"))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(t=30), yaxis_tickformat=".0%",
                      yaxis_title="Share of portfolio variance", title="Portfolio variance by principal component")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"The first component explains {pca['first_share']:.0%} of the holdings' total variance and "
               f"{pca['first_portfolio_share']:.0%} of the portfolio's. Effective number of bets (Meucci, 2009) = "
               f"exp(entropy of the portfolio's variance across components) = {pca['enb']:.2f}, out of a possible "
               f"{len(ctx.conc_weights)}.")


def _crisis(ctx, res, curr):
    st.markdown("### 🌪️ Correlation in a crisis")
    crisis = res["crisis"]
    if crisis is None:
        st.info("Not every holding has prices back to the crisis windows, so crisis correlation is not available.")
        return
    t = crisis["table"]
    st.dataframe(pd.DataFrame({
        "Sample": t["Sample"], "Days": t["Days"], "Average correlation": t["Average Correlation"].map(lambda v: _num(v, 2)),
        f"Portfolio ES ({ctx.cl_label}, 1-day)": t["ES"].map(lambda v: f"{curr}{v:,.0f}" if np.isfinite(v) else "not enough days"),
        "Diversification benefit kept": t["Benefit Kept"].map(lambda v: _pct(v))}), hide_index=True, width="stretch")
    st.caption(f"Normal ES with each sample's correlations and the full-sample volatilities, so only correlation changes. "
               f"Sum of standalone ES: {curr}{crisis['standalone_es']:,.0f}. Benefit kept = (standalone − portfolio ES) in the "
               "crisis sample ÷ the same in the full sample, on the common long history of all holdings. Choosing days by "
               "how far the market fell narrows the market's range within the sample, which lowers measured correlation "
               "(Boyer, Gibson and Loretan, 1999); the crisis windows avoid that bias, and the headline takes the lower "
               "of the two.")


def _rolling(res):
    rolling = {t: r for t, r in res.get("rolling", {}).items() if len(r)}
    if not rolling:
        return
    st.markdown("### 📈 One-year market beta over time")
    fig = go.Figure()
    for i, (t, r) in enumerate(rolling.items()):
        fig.add_scatter(x=r.index, y=r["MKT_RF"], mode="lines", name=t, line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)]))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30), yaxis_title="Market beta")
    st.plotly_chart(fig, width="stretch")
    st.caption("Each point uses the 252 trading days ending on that date, re-estimated every 21 days.")
