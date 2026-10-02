"""
Liquidity page (Pillar 2). Layout shared by every pillar page:
headline metrics (with grade) → detail → what this misses → methodology.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import liquidity as L
from portfolio import ENTRY_SHARES, build_positions, clean_holdings
from ui.cache import cached_market_data
from ui.formatting import MODEL_COLORS

METHODOLOGY = "docs/methodology.md, section 9"
VALIDATION_KEY = "amfi_validation"
VALIDATION_TEMPLATE = pd.DataFrame({"Ticker": ["", ""], "Quantity": ["", ""]})


def _days(v) -> str:
    if not np.isfinite(v):
        return "never (no volume)"
    if v < 0.01:
        return "under 0.01 days"
    return f"{v:,.2f} days" if v < 10 else f"{v:,.1f} days"


def render(ctx):
    curr = ctx.curr_sym

    def money(v):
        return f"{curr}{v:,.0f}" if v is not None and np.isfinite(v) else "not available"

    m = ctx.liquidity_metrics
    st.subheader(f"💧 Liquidity: {ctx.company_name}")
    st.caption(f"Participation rate {ctx.participation:.0%} of daily volume (assumption) · volume: "
               + "; ".join(" + ".join(v) for v in ctx.volume_sources.values()))
    st.caption("90% ranges: days to liquidate and the share sellable are recomputed with each day's average volume over "
               "the past year, so today's figure can sit outside its range when recent volume is unusual; the LVaR range "
               "adds the VaR range to bootstrapped spread costs and impact over the past year's volume.")

    cols = st.columns(4)
    cards = [(m["amfi50"], _days, "Days to liquidate 50% (AMFI)"),
             (m["sell5"], lambda v: f"{v:.0%}", "Sellable in 5 days"),
             (m["lvar"], money, "Liquidity-adjusted VaR"),
             (m["circuit"], money, "Circuit-lock loss")]
    for col, (metric, fmt, label) in zip(cols, cards):
        col.metric(label, fmt(metric.value) if np.isfinite(metric.value) else "not available")
        col.caption(f"{metric.range_text(fmt)} · grade **{metric.grade}**")
        col.caption("; ".join(metric.reasons) or "no deductions")

    _capacity(ctx, money)
    _amfi(ctx)
    _stressed(ctx)
    _costs(ctx, money)
    _circuit(ctx, money)
    _amihud(ctx)
    _validation(ctx)

    st.markdown("### 🕳️ What this metric misses")
    st.markdown("""
- **Block and bulk deals.** A large holder can often sell a block off-market at a negotiated discount, faster than the daily-volume figures suggest; the discount itself is not modelled.
- **Free float.** Volume measures trading, not how many shares are actually available; a stock with a small free float can freeze faster than its volume suggests.
- **The liabilities side.** Fund redemptions force selling at the worst time; the AMFI test assumes pro-rata selling, while real funds often sell their most liquid holdings first and leave the rest more concentrated.
- **Intraday liquidity and order-book depth.** Daily volume and high-low spreads say nothing about depth at a given moment; impact is a square-root-law approximation with an assumed constant.
- **Circuit bands change.** Exchanges revise bands; an inferred band is a guess from history, and F&O stocks have no fixed band but can still halt under market-wide circuit breakers.
- **Volume on Yahoo** may miss BSE for large caps (see Overview) and includes no off-exchange trades.
""")
    st.caption(f"Methodology: {METHODOLOGY}.")


def _capacity(ctx, money):
    st.markdown("### 🚚 Trading capacity")
    h = ctx.liquidity_holdings
    table = pd.DataFrame({
        "Ticker": h["Ticker"], "Value": h["Value"].map(money), "ADV 60d (shares)": h["ADV 60d"].map("{:,.0f}".format),
        "Position, % of a day's volume": h["Position / ADV"].map(lambda v: f"{v:.3%}" if np.isfinite(v) else "no volume"),
        "Traded value 60d": h["Traded Value 60d"].map(money), "Days to liquidate": h["Days to Liquidate"].map(_days),
        "Stressed days": h["Stressed Days"].map(_days),
        "Amihud (bp per ₹1 cr)": h["Amihud"].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "not available")})
    st.dataframe(table, hide_index=True, width="stretch")
    s = ctx.liquidity_sellable
    st.markdown(f"Share of the portfolio sellable within **1 day: {s[1]:.0%} · 5 days: {s[5]:.0%} · 10 days: {s[10]:.0%}**, "
                f"each holding sold at {ctx.participation:.0%} of its 60-day average volume. "
                f"Slowest holding: **{ctx.liquidity_slowest['Ticker']}** ({_days(ctx.liquidity_slowest['Days to Liquidate'])}).")


def _amfi(ctx):
    st.markdown("### 🏛️ SEBI/AMFI-style liquidity stress test")
    rows = []
    for fraction in L.AMFI_FRACTIONS:
        with_ex, without = ctx.liquidity_amfi[(fraction, True)], ctx.liquidity_amfi[(fraction, False)]
        rows.append({"Portfolio sold (pro rata)": f"{fraction:.0%}",
                     f"Days, least liquid {ctx.amfi_exclude:.0%} excluded": _days(with_ex["days"]),
                     "Slowest remaining holding": with_ex["binding"] or "-",
                     "Days, nothing excluded": _days(without["days"]), "Slowest holding": without["binding"] or "-"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Each holding is sold pro rata at {ctx.amfi_participation:.0%} of its 3-month (63-day) average volume "
               "(NSE + BSE where available). Holdings are ranked from least to most liquid and removed until "
               f"{ctx.amfi_exclude:.0%} of the portfolio value is excluded; the boundary holding is excluded only in part. "
               "This is our reading of the AMFI method, not an official calculation.")
    with st.expander("Which holdings were excluded (50% case)"):
        table = ctx.liquidity_amfi[(0.50, True)]["table"]
        st.dataframe(table.assign(**{"Kept Share": table["Kept Share"].map("{:.0%}".format),
                                     "Days": table["Days"].map(_days), "Full Days": table["Full Days"].map(_days)}),
                     hide_index=True, width="stretch")


def _stressed(ctx):
    st.markdown("### 🌪️ Volume in past crises")
    parts = []
    for t, res in ctx.liquidity_stressed.items():
        table = res["table"].assign(Ticker=t)
        parts.append(table)
    table = pd.concat(parts, ignore_index=True)
    shown = table.assign(Start=table["Start"].dt.strftime("%d %b %Y"), End=table["End"].dt.strftime("%d %b %Y"),
                         Ratio=table["Ratio"].map(lambda v: f"{v:.2f}×" if np.isfinite(v) else "not enough data"))
    st.dataframe(shown[["Ticker", "Scenario", "Start", "End", "Window Days", "Ratio"]], hide_index=True, width="stretch")
    def factor_text(t, r):
        if not np.isfinite(r["measured"]):
            return f"{t} not available"
        return f"{t} {r['measured']:.2f}× measured, {r['factor']:.2f}× applied"

    st.caption("Ratio = average volume in the crisis window ÷ average volume in the 120 trading days before it, on the "
               "primary listing's full history. Stress factor = median ratio: "
               + "; ".join(factor_text(t, r) for t, r in ctx.liquidity_stressed.items()) +
               ". Large caps often trade more in a sell-off, but that volume is other sellers' too, so the factor applied "
               "is capped at 1 (an assumption): a crisis is never taken to make selling easier. Without enough history, "
               "today's volume is used unchanged.")


def _costs(ctx, money):
    st.markdown("### 💸 Spread, impact and liquidity-adjusted VaR")
    w = ctx.liquidity_waterfall
    fig = go.Figure(go.Waterfall(x=w["Step"], y=w["Amount"], measure=["absolute", "relative", "relative", "relative"],
                                 text=[money(v) for v in w["Amount"]], textposition="outside",
                                 connector={"line": {"color": "#718096"}}))
    fig.add_bar(x=["= Liquidity-adjusted VaR"], y=[w["Cumulative"].iloc[-1]], marker_color="#FC8181",
                text=[money(w["Cumulative"].iloc[-1])], textposition="outside", showlegend=False)
    fig.update_layout(template="plotly_dark", height=380, margin=dict(t=30), yaxis_title=f"Loss ({ctx.curr_sym})",
                      showlegend=False)
    st.plotly_chart(fig, width="stretch")
    h = ctx.liquidity_holdings
    st.dataframe(pd.DataFrame({
        "Ticker": h["Ticker"], "Spread": h["Spread"].map(lambda v: f"{v:.2%}" if np.isfinite(v) else "not available"),
        "Spread source": h["Spread Source"], "Spread cost": h["Spread Cost"].map(money),
        "Impact cost": h["Impact Cost"].map(money)}), hide_index=True, width="stretch")
    st.caption(f"VaR: {ctx.headline_model} at {ctx.cl_label}, {ctx.holding_period}-day. Spread cost (Bangia et al., 1999) = "
               f"½ × value × (mean monthly spread + {ctx.bangia_k:g} × its standard deviation), over the last 12 months. "
               f"Impact = {ctx.impact_y:g} × daily σ × √(shares ÷ stressed daily volume). Circuit-lock add-on = loss beyond "
               "VaR if every banded holding is frozen for its exit-freeze scenario: a stress add-on, not a probability-based "
               "figure. k, Y and the participation rate are assumptions you can change in the sidebar.")


def _circuit(ctx, money):
    st.markdown("### 🔒 Circuit-lock risk")
    h = ctx.liquidity_holdings
    st.dataframe(pd.DataFrame({
        "Ticker": h["Ticker"], "Band": h["Band"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "no fixed band"),
        "Band source": h["Band Source"], "Lower-circuit days": h["Lower-Circuit Days"], "Longest run": h["Longest Run"],
        "Exit freeze (days)": h["Freeze Days"], "Loss if frozen": h["Circuit Loss"].map(money)}),
        hide_index=True, width="stretch")
    st.caption("Lower-circuit day: closed at the low after falling by the band (±0.1 percentage points), in the lookback "
               "window. Exit freeze: N consecutive lower circuits (default: the longest past run, at least 3) during which "
               "the holding cannot be sold; loss = 1 − (1 − band)^N. An inferred band is a guess from history; upload the "
               "official NSE price-band file to replace it (Overview → Disclosure data).")


def _amihud(ctx):
    st.markdown("### 📉 Amihud illiquidity")
    fig = go.Figure()
    for i, (t, series) in enumerate(ctx.liquidity_amihud.items()):
        series = series.dropna()
        fig.add_scatter(x=series.index, y=series.to_numpy(), mode="lines", name=t,
                        line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)]))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(t=30), yaxis_title="bp of price move per ₹1 crore traded")
    st.plotly_chart(fig, width="stretch")
    st.caption("60-day rolling average of |daily return| ÷ daily traded value (Amihud, 2002), in basis points of price "
               "move per ₹1 crore (10 million currency units) traded. Higher means each rupee of trading moves the price more.")


def _validation(ctx):
    with st.expander("✅ Validate against a fund's published liquidity stress test"):
        st.markdown("Upload a fund's monthly portfolio as **Ticker, Quantity** (Yahoo tickers, e.g. `RELIANCE.NS`) and "
                    "enter the days to liquidate 50% and 25% that the AMC published for the same month. The tool downloads "
                    "the prices and volume, runs the same test and reports the gap.")
        st.download_button("Download template", VALIDATION_TEMPLATE.to_csv(index=False).encode(),
                           file_name="fund_portfolio_template.csv", mime="text/csv")
        upload = st.file_uploader("Fund portfolio (CSV)", type="csv", key="amfi_upload")
        c1, c2 = st.columns(2)
        published50 = c1.number_input("Published days, 50%", 0.0, 1000.0, 0.0, 0.1, key="amfi_pub50")
        published25 = c2.number_input("Published days, 25%", 0.0, 1000.0, 0.0, 0.1, key="amfi_pub25")
        if upload is not None and st.button("Run the validation", key="amfi_run"):
            st.session_state[VALIDATION_KEY] = run_validation(pd.read_csv(io.BytesIO(upload.getvalue()), dtype=str), ctx)
        result = st.session_state.get(VALIDATION_KEY)
        if result is None:
            return
        if result.get("error"):
            st.error(result["error"])
            return
        rows = []
        for fraction, published in ((0.50, published50), (0.25, published25)):
            ours = result["days"][fraction]
            rows.append({"Sold": f"{fraction:.0%}", "This tool": _days(ours),
                         "Published": f"{published:,.1f} days" if published else "not entered",
                         "Gap": f"{ours - published:+,.1f} days" if published and np.isfinite(ours) else "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if result["missing"]:
            st.warning("Could not load: " + ", ".join(result["missing"]) + ". These holdings were left out, which shortens "
                       "the computed days if they are illiquid.")
        st.caption("Likely causes of a gap: a different 3-month window (the AMC uses the month-end it reports), NSE-only vs "
                   "NSE + BSE volume (Yahoo often lacks BSE history for large caps), how the least liquid 20% is cut "
                   "(whole holdings vs our partial cut), holdings Yahoo cannot price (unlisted, debt, cash), and rounding.")


def run_validation(table: pd.DataFrame, ctx) -> dict:
    """Run the AMFI test on an uploaded fund portfolio with today's 3-month volume."""
    try:
        quantities = clean_holdings(table.rename(columns=lambda c: str(c).strip().title()), "Quantity")
    except (ValueError, KeyError) as exc:
        return {"error": f"Fund portfolio: {exc}"}
    fetched = {t: cached_market_data(t, period="1y") for t in quantities.index}
    ok = {t: r for t, r in fetched.items() if r["success"] and len(r["df"]) > 20}
    if len(ok) < 2:
        return {"error": "Fewer than two holdings could be loaded from Yahoo Finance."}
    positions = build_positions(quantities[list(ok)], ENTRY_SHARES, {t: float(r["df"]["Close"].iloc[-1]) for t, r in ok.items()})
    adv = {t: L.average_volume(r["df"], L.AMFI_ADV_DAYS)[0] for t, r in ok.items()}
    days = {f: L.amfi_stress(positions, adv, f, ctx.amfi_participation, ctx.amfi_exclude)["days"] for f in L.AMFI_FRACTIONS}
    return {"days": days, "missing": [t for t in quantities.index if t not in ok], "error": None}
