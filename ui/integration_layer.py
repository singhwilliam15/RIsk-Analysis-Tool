"""
Integration and decisions shared by every page (Phase 6): linked stress, reverse stress, risk-change explanation,
risk-reducing trades, hedge, limits, the CRO dashboard rows and the memo content.
"""

import numpy as np
import pandas as pd
import streamlit as st

import decisions as Dz
import integration as I
import liquidity as L
from factor_data import region_for
from portfolio import align_asset_returns
from stress import load_scenarios, price_series
from ui.cache import cached_fetch
from ui.context import export

MACRO = {
    "india": {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Nifty IT": "^CNXIT", "USD/INR": "INR=X", "Brent crude": "BZ=F"},
    "us": {"S&P 500": "^GSPC", "Nasdaq 100": "^NDX", "US dollar index": "DX-Y.NYB", "Brent crude": "BZ=F"},
}
REVERSE_LOSS_KEY, CHANGE_MONTHS_KEY, SNAPSHOT_KEY, LIMITS_KEY = "reverse_loss", "change_months", "risk_snapshot", "limits_override"
REVERSE_LOSSES = (0.10, 0.15, 0.20, 0.30)
CHANGE_MONTHS = (1, 3, 6, 12)


def _returns_by_date(df: pd.DataFrame) -> pd.Series:
    return df.set_index(df["Date"].dt.normalize())["Returns"].dropna()


def compute_integration(ctx):
    """Linked stress for every scenario, the siloed comparison, and the reverse stress test."""
    tickers = list(ctx.price_frames)
    long = {t: cached_fetch(t, period="max") for t in tickers}
    holding_long = {t: _returns_by_date(r["df"]) if r["success"] else _returns_by_date(ctx.price_frames[t]) for t, r in long.items()}
    lookback = {t: _returns_by_date(df) for t, df in ctx.price_frames.items()}
    bench = cached_fetch(ctx.benchmark_symbol, period="max")
    market_prices = price_series(bench["df"]) if bench["success"] else pd.Series(dtype=float)
    today_sigma = {t: float(r.tail(60).std(ddof=1)) for t, r in lookback.items()}
    scenarios = I.scenario_inputs(holding_long, market_prices, load_scenarios(market=ctx.market_key), today_sigma, lookback) \
        if len(market_prices) else []

    lh = ctx.liquidity_holdings.set_index("Ticker")
    base_rows = []
    for p in ctx.positions.to_dict("records"):
        t = p["Ticker"]
        cr = ctx.credit_results.get(t, {})
        spread = ctx.liquidity_spreads[t]
        merton = cr.get("merton", {}).get(ctx.equity_vol_choice, {})
        base_rows.append({"Ticker": t, "Value": p["Value"], "Quantity": p["Quantity"], "Price": p["Price"],
                          "ADV": lh.loc[t, "ADV 60d"], "Spread Mean": spread["mean"], "Spread Std": spread["std"],
                          "Band": lh.loc[t, "Band"], "Freeze Days": lh.loc[t, "Freeze Days"],
                          "Equity": cr.get("market_cap") or np.nan, "Default Point": cr.get("default_point") or np.nan,
                          "PD": merton.get("PD", np.nan), "Financial": bool(cr.get("financial", False)),
                          "Pledged Shares": ctx.event_margin[t]["pledged_shares"]})
    base = pd.DataFrame(base_rows)
    params = {"bangia_k": ctx.bangia_k, "impact_y": ctx.impact_y, "r": ctx.risk_free_pct / 100, "T": ctx.merton_horizon,
              "initial_cover": ctx.initial_cover, "trigger_cover": ctx.trigger_cover}

    def with_ratios(scenario):
        frame = base.copy()
        ratios = []
        for t in frame["Ticker"]:
            table = ctx.liquidity_stressed[t]["table"].set_index("Scenario")
            if scenario["kind"] == "historical" and scenario["name"] in table.index and table.loc[scenario["name"], "Used"]:
                ratios.append(table.loc[scenario["name"], "Ratio"])
            else:
                ratios.append(ctx.liquidity_stressed[t]["factor"])
        return frame.assign(**{"Volume Ratio": ratios})

    inv = ctx.investment_amount
    standalone = {"liquidity": float(ctx.liquidity_holdings["Spread Cost"].sum(min_count=1) + ctx.liquidity_holdings["Impact Cost"].sum(min_count=1)),
                  "credit": ctx.credit_view["expected_loss"] if np.isfinite(ctx.credit_view["expected_loss"]) else 0.0,
                  "events": ctx.event_overlay["gap"] * inv}
    rows = []
    for sc in scenarios:
        table = I.run_linked(with_ratios(sc), [sc], params, standalone)
        rows.append(table.iloc[0])
    linked = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()

    # --- reverse stress -----------------------------------------------------------
    loss = st.session_state.get(REVERSE_LOSS_KEY, 0.15)
    weights = ctx.positions.set_index("Ticker")["Weight"]
    asset = pd.concat([lookback[t].rename(t) for t in weights.index], axis=1, join="inner").dropna()
    cov_full = asset.cov().to_numpy() * I.REVERSE_HORIZON
    nu = ctx.var_selected["Student-t"]["degrees_of_freedom"]
    def with_probabilities(res):
        res["probability"] = I.loss_probability(loss, res["portfolio_sigma"], nu)
        res["plausibility"] = I.plausibility(res["d2"], len(weights), nu)  # any-direction share, shown as context
        return res

    reverse = {"loss": loss, "full": with_probabilities(I.reverse_stress(weights.to_numpy(), cov_full, loss))}
    crisis = (ctx.conc.get("crisis") or {}).get("correlations", {}).get("Crisis windows") if isinstance(ctx.conc, dict) else None
    if crisis is not None:
        vols = asset.std(ddof=1).loc[crisis.index].to_numpy()
        cov_crisis = np.outer(vols, vols) * crisis.to_numpy() * I.REVERSE_HORIZON
        reverse["crisis"] = with_probabilities(I.reverse_stress(weights.loc[crisis.index].to_numpy(), cov_crisis, loss))
    scale = np.median([v["sigma"] / today_sigma[t] for sc in scenarios if sc["kind"] == "custom"
                       for t, v in sc["holdings"].items() if today_sigma[t] > 0]) if scenarios else 2.0
    reverse_scenario = {"name": f"Reverse stress: lose {loss:.0%} in a month", "kind": "reverse", "market": np.nan,
                        "holdings": {t: {"return": float(x), "method": "reverse", "sigma": today_sigma[t] * scale}
                                     for t, x in zip(weights.index, reverse["full"]["shock"])}}
    reverse["linked"] = I.run_linked(with_ratios(reverse_scenario), [reverse_scenario], params, standalone).iloc[0]

    region = region_for(tickers)
    macro_names = MACRO.get(region or "", {})
    macro_series, missing_macro = {}, []
    for name, symbol in macro_names.items():
        res = cached_fetch(symbol, period="max")
        if res["success"]:
            macro_series[name] = price_series(res["df"]).pct_change().dropna()
        else:
            missing_macro.append(name)
    portfolio_long = ctx.position_history
    reverse["macro"] = I.macro_reverse(portfolio_long, pd.DataFrame(macro_series).dropna(), loss) if len(macro_series) >= 2 else None
    reverse["missing_macro"] = missing_macro
    export(ctx, {"linked": linked, "linked_standalone": standalone, "reverse": reverse, "scenario_count": len(scenarios)})


def compute_decisions(ctx):
    """Risk change, trades, hedge, limits, and the dashboard's risks and actions."""
    alpha = 1 - ctx.confidence_level
    inv = ctx.investment_amount
    weights = ctx.positions.set_index("Ticker")["Weight"]
    lookback = {t: _returns_by_date(df) for t, df in ctx.price_frames.items()}
    asset = pd.concat([lookback[t].rename(t) for t in weights.index], axis=1, join="inner").dropna()
    model = ctx.headline_model if ctx.headline_model in Dz.MODELS else "Historical"

    # Risk change: today vs N months ago (same holdings), or vs an uploaded snapshot
    months = st.session_state.get(CHANGE_MONTHS_KEY, 6)
    long = {t: cached_fetch(t, period="max") for t in weights.index}
    long_returns = align_asset_returns({t: r["df"] for t, r in long.items()}) if all(r["success"] for r in long.values()) else asset
    new = Dz.snapshot(asset, weights, model, ctx.prices_as_of, inv, ctx.confidence_level)
    uploaded = st.session_state.get(SNAPSHOT_KEY)
    if uploaded is not None:
        old = uploaded
    else:
        earlier = long_returns[long_returns.index <= ctx.prices_as_of - pd.DateOffset(months=months)].tail(len(asset))
        old = Dz.snapshot(earlier, weights, model, earlier.index[-1] if len(earlier) else ctx.prices_as_of, inv, ctx.confidence_level) \
            if len(earlier) > 60 else None
    change = Dz.risk_change(old, new, alpha, inv) if old is not None else None

    # Trades and hedge
    components = Dz.component_es(asset, weights, alpha) * inv
    trades = Dz.candidate_trades(asset, weights, alpha) if len(weights) > 1 else pd.DataFrame()
    adv_3m = {t: L.average_volume(ctx.price_frames[t], L.AMFI_ADV_DAYS)[0] for t in weights.index}
    pd_by = ctx.credit_view["table"].set_index("Ticker")["PD"]
    tiers = {t: r["tier"] for t, r in ctx.event_tiers.items()}
    # Selling into cash always cuts ES roughly in proportion; switches show where diversification helps. So the
    # list is the best sale into cash plus the two best switches between holdings (each only if it cuts ES).
    if len(trades):
        sales = trades[trades["Buy"] == "cash"].head(1)
        switches = trades[trades["Buy"] != "cash"].head(2)
        trades = pd.concat([sales, switches]).query("`ES Change` < 0").sort_values("ES Change")
    effects = []
    for r in trades.to_dict("records") if len(trades) else []:
        new_w = r["weights"]
        pos = ctx.positions.set_index("Ticker").assign(Value=new_w * inv)
        pos["Quantity"] = pos["Value"] / pos["Price"]
        days = L.amfi_stress(pos.reset_index(), adv_3m, 0.50, ctx.amfi_participation, ctx.amfi_exclude)["days"]
        modelled = pd_by.dropna()
        wpd = float((modelled * new_w.loc[modelled.index]).sum() / new_w.loc[modelled.index].sum()) if len(modelled) and new_w.loc[modelled.index].sum() else np.nan
        flagged = float(sum(new_w[t] for t in new_w.index if tiers.get(t) != "Low") / new_w.sum())
        effects.append({**{k: v for k, v in r.items() if k != "weights"}, "Days to Liquidate 50%": days, "Weighted PD": wpd,
                        "Value in Elevated/High": flagged})
    market = (price_series(cached_fetch(ctx.benchmark_symbol, period=ctx.period_input)["df"]).pct_change().dropna())
    hedge = Dz.hedge_ratio(ctx.stock_by_date.dropna(), market, alpha)

    # Limits
    limits = st.session_state.get(LIMITS_KEY) or Dz.load_limits()
    grades = {"Market ES": ctx.trusted[ctx.headline_model]["ES"].grade, "Liquidity": ctx.liquidity_metrics["amfi50"].grade,
              "Credit": ctx.credit_metrics["weighted_pd"].grade, "Events": ctx.event_metrics["event_es"].grade}
    if "factor_share" in ctx.conc_metrics:
        grades["Concentration"] = ctx.conc_metrics["factor_share"].grade
    worst_grade = max(grades.values(), key=lambda g: Dz.GRADE_ORDER.get(g, 0))
    sectors = ctx.positions.groupby("Sector")["Weight"].sum()
    measures = {"es_pct": ctx.trusted[ctx.headline_model]["ES"].value / inv, "max_weight": float(weights.max()),
                "max_sector_weight": float(sectors.max()), "days_to_liquidate_50": ctx.liquidity_metrics["amfi50"].value,
                "max_weighted_pd": ctx.credit_view["weighted_pd"], "max_high_tier": float(sum(v == "High" for v in tiers.values())),
                "min_trust_grade": worst_grade}
    if not ctx.is_portfolio:
        # One stock is always 100% of the portfolio and of its sector: those limits apply to portfolios only
        limits = {**limits, "limits": {k: v for k, v in limits["limits"].items() if k not in ("max_weight", "max_sector_weight")}}
    limits_table = Dz.evaluate_limits(limits, measures)

    worst = None
    if len(ctx.linked):
        row = ctx.linked.loc[ctx.linked["Linked Total"].idxmax()]
        worst = {"name": row["Scenario"], "loss_pct": row["Linked Total"] / inv, "interaction_pct": row["Interaction"] / inv}
    risks = Dz.top_risks(limits_table, worst, components, weights, tiers, grades, ctx.event_missing)
    actions = Dz.top_actions(pd.DataFrame(effects), hedge, limits_table, ctx.event_missing, inv)
    export(ctx, {"risk_change": change, "risk_change_old": old, "risk_snapshot_now": new, "es_components": components,
                 "trades": pd.DataFrame(effects), "hedge": hedge, "limits": limits, "limits_table": limits_table,
                 "pillar_grades": grades, "worst_scenario": worst, "top_risks": risks, "top_actions": actions})
    export(ctx, {"dashboard": dashboard_rows(ctx)})
    export(ctx, {"memo_content": memo_content(ctx)})  # uses the dashboard rows


# ---------------------------------------------------------------
# Dashboard and memo content
# ---------------------------------------------------------------

def _pd_text(v) -> str:
    """As on the Credit page: below 0.001% the exact (risk-neutral) PD means nothing."""
    if v is None or not np.isfinite(v):
        return "not available"
    return "below 0.001%" if v < 1e-5 else f"{v:.3%}"


def _money(ctx, v):
    return f"{ctx.curr_sym}{v:,.0f}" if v is not None and np.isfinite(v) else "not available"


def dashboard_rows(ctx) -> list:
    """One row per pillar: headline, range, grade and the status of its limit."""
    status = ctx.limits_table.set_index("Key")["Status"].to_dict()
    money = lambda v: _money(ctx, v)  # noqa: E731
    es = ctx.trusted[ctx.headline_model]["ES"]
    lm, cm, em = ctx.liquidity_metrics, ctx.credit_metrics, ctx.event_metrics
    rows = [
        {"Pillar": "Market", "Headline": f"ES {money(es.value)} ({ctx.headline_model})", "Range": es.range_text(money),
         "Grade": es.grade, "Status": status.get("es_pct", "-")},
        {"Pillar": "Liquidity", "Headline": f"{lm['amfi50'].value:,.2f} days to sell 50%; LVaR {money(lm['lvar'].value)}",
         "Range": lm["lvar"].range_text(money), "Grade": lm["lvar"].grade, "Status": status.get("days_to_liquidate_50", "-")},
        {"Pillar": "Credit", "Headline": "weighted PD " + _pd_text(cm["weighted_pd"].value),
         "Range": cm["weighted_pd"].range_label + (" (see Credit)" if cm["weighted_pd"].has_range else ": not available"),
         "Grade": cm["weighted_pd"].grade, "Status": status.get("max_weighted_pd", "-")},
    ]
    if "enb" in ctx.conc_metrics:
        enb = ctx.conc_metrics["enb"]
        rows.append({"Pillar": "Concentration", "Headline": f"{enb.value:.2f} effective bets; largest holding {ctx.positions['Weight'].max():.0%}",
                     "Range": enb.range_text(lambda v: f"{v:.2f}"), "Grade": enb.grade, "Status": status.get("max_weight", "-")})
    elif "factor_share" in ctx.conc_metrics:
        fs = ctx.conc_metrics["factor_share"]
        rows.append({"Pillar": "Concentration", "Headline": f"factor share {fs.value:.0%} (single stock)", "Range": "-",
                     "Grade": fs.grade, "Status": status.get("max_weight", "-")})
    rows.append({"Pillar": "Events", "Headline": f"event-adjusted ES {money(em['event_es'].value)}; "
                 f"{int(em['high'].value)} High-tier holding(s)", "Range": em["event_es"].range_text(money),
                 "Grade": em["event_es"].grade, "Status": status.get("max_high_tier", "-")})
    if ctx.worst_scenario:
        w = ctx.worst_scenario
        rows.append({"Pillar": "Integrated stress", "Headline": f"{w['name']}: {w['loss_pct']:.1%} linked loss",
                     "Range": f"interaction {w['interaction_pct']:+.1%} vs siloed", "Grade": "-", "Status": "-"})
    return rows


def memo_content(ctx) -> dict:
    money = lambda v: _money(ctx, v)  # noqa: E731
    stress = None
    if len(ctx.linked):
        row = ctx.linked.loc[ctx.linked["Linked Total"].idxmax()]
        inv = ctx.investment_amount
        stress = {"name": row["Scenario"], "market": money(row["Market Loss"]), "liquidity": money(row["+ Liquidity"]),
                  "credit": money(row["+ Credit"]), "events": money(row["+ Events"]), "linked": money(row["Linked Total"]),
                  "siloed": money(row["Siloed Sum"]), "interaction": money(row["Interaction"]),
                  "linked_pct": row["Linked Total"] / inv, "interaction_pct": row["Interaction"] / inv,
                  "market_value": row["Market Loss"], "liquidity_value": row["+ Liquidity"], "credit_value": row["+ Credit"],
                  "events_value": row["+ Events"], "siloed_value": row["Siloed Sum"]}
    reverse = None
    rv = ctx.reverse
    if rv.get("macro"):
        shock = ", ".join(f"{k} {v:+.0%}" for k, v in rv["macro"]["shock"].items())
        reverse = f"Most plausible way to lose {rv['loss']:.0%} in a month: {shock}."
        if rv["macro"]["analogue"]:
            a = rv["macro"]["analogue"]
            reverse += f" Nearest historical analogue: {a['start']:%b %Y} to {a['end']:%b %Y} (portfolio {a['portfolio_return']:+.1%})."
    limits = []
    for r in ctx.limits_table.to_dict("records"):
        fmt = (lambda v: f"{v:.1%}") if r["Unit"] == "fraction" else (lambda v: f"{v:,.2f}") if r["Unit"] == "days" else str
        shown = r["Value"] if isinstance(r["Value"], str) else (fmt(r["Value"]) if np.isfinite(r["Value"]) else "not available")
        limits.append({**r, "Shown": shown, "Limit Shown": r["Limit Value"] if isinstance(r["Limit Value"], str) else fmt(r["Limit Value"]),
                       "Utilisation Shown": f"{r['Utilisation']:.0%}" if np.isfinite(r["Utilisation"]) else "-"})
    sources = [ctx.data_note, "Fundamentals: Yahoo Finance statements or your CSV overrides"]
    if ctx.conc_factor_meta:
        sources.append(f"Factors: {ctx.conc_factor_meta['source']} to {ctx.conc_factor_meta['end_date']}")
    sources.append("Disclosures loaded: " + (", ".join(ctx.event_loaded) or "none") +
                   (f"; missing: {', '.join(ctx.event_missing)}" if ctx.event_missing else ""))
    sources.append("Assumptions (participation rate, k, Y, jump sizes, cover ratios, tier and limit thresholds) are editable "
                   "and listed in docs/methodology.md.")
    return {"name": ctx.company_name, "as_of": ctx.prices_as_of, "confidence": f"{ctx.cl_label}, {ctx.holding_period}-day",
            "value": money(ctx.investment_amount), "currency": ctx.currency, "pillars": ctx.dashboard, "stress": stress,
            "reverse": reverse, "limits": limits, "risks": ctx.top_risks, "actions": ctx.top_actions, "sources": sources}
