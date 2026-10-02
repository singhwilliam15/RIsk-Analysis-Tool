"""Liquidity calculations shared by every page (Pillar 2), with 90% ranges and grades on the headline numbers."""

import numpy as np
import pandas as pd

import liquidity as L
from disclosures import normalise_symbol
from stress import load_scenarios
from trust import TrustedMetric, grade
from ui.cache import cached_fetch, cached_liquidity
from ui.context import export

INDIAN = (".NS", ".BO")


def _official_bands(disclosures: dict) -> dict:
    """Latest official price band per symbol (fraction, NaN for 'No Band') from the price-band files."""
    bands = disclosures["data"]["price_bands"]
    if bands.empty:
        return {}
    bands = bands[pd.to_datetime(bands["effective_date"]) <= pd.Timestamp.today()]
    latest = bands.sort_values("effective_date").groupby("symbol").tail(1)
    return {row["symbol"]: row["band_pct"] / 100 for row in latest.to_dict("records")}


def _percentiles(values) -> tuple:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(values) < 10:
        return np.nan, np.nan
    return float(np.percentile(values, 5)), float(np.percentile(values, 95))


def analyse(positions: pd.DataFrame, frames: dict, long_frames: dict, scenarios: pd.DataFrame, official: dict,
            participation: float, amfi_participation: float, amfi_exclude: float, bangia_k: float, impact_y: float,
            user_spread_pct: float, freeze_override: int) -> dict:
    """
    Every liquidity figure that does not depend on VaR, and the 90% ranges of the cost and capacity figures.
    Pure function of its inputs, so the app caches it.
    """
    capacity = L.trading_capacity(positions, frames, participation)
    sellable = L.sellable_share(capacity["Value"], capacity["Days to Liquidate"])
    adv_3m = {t: L.average_volume(frames[t], L.AMFI_ADV_DAYS)[0] for t in frames}
    amfi = {(f, excl): L.amfi_stress(positions, adv_3m, f, amfi_participation, amfi_exclude if excl else 0.0)
            for f in L.AMFI_FRACTIONS for excl in (True, False)}

    rows, stressed, spreads = [], {}, {}
    by_ticker = capacity.set_index("Ticker")
    for p in positions.to_dict("records"):
        t = p["Ticker"]
        df, long_df = frames[t], long_frames.get(t, frames[t])
        volume = long_df.set_index("Date")["Volume"] if "Volume" in long_df else pd.Series(dtype=float)
        stressed[t] = L.stressed_volume(volume, scenarios)
        factor = stressed[t]["factor"]
        cap = by_ticker.loc[t]
        stressed_adv = cap["ADV 60d"] * factor if np.isfinite(factor) else cap["ADV 60d"]
        sigma = float(df["Returns"].tail(60).std(ddof=1))
        if user_spread_pct > 0:
            profile = {"mean": user_spread_pct / 100, "std": 0.0, "monthly": pd.Series(dtype=float), "source": "entered by you"}
        else:
            profile = L.spread_profile(df)
        spreads[t] = profile
        if t.endswith(INDIAN):
            lock = L.circuit_lock(df, official.get(normalise_symbol(t)), freeze_override or None)
        else:
            lock = {"band": np.nan, "source": "not applicable (no price bands outside India)", "circuit_days": 0,
                    "longest_run": 0, "freeze_days": 0, "loss_pct": 0.0}
        amihud = L.amihud(df).dropna()
        rows.append({
            "Ticker": t, "Quantity": p["Quantity"], "Value": p["Value"], "Sigma": sigma, "ADV 60d": cap["ADV 60d"],
            "Traded Value 60d": cap["Traded Value 60d"], "Days to Liquidate": cap["Days to Liquidate"],
            "Position / ADV": p["Quantity"] / cap["ADV 60d"] if cap["ADV 60d"] > 0 else np.inf,
            "Measured Crisis Ratio": stressed[t]["measured"], "Stress Factor": factor, "Stressed ADV": stressed_adv,
            "Stressed Days": float(L.days_to_liquidate(p["Quantity"], stressed_adv, participation)),
            "Spread": profile["mean"], "Spread Source": profile["source"],
            "Spread Cost": L.bangia_cost(p["Value"], profile["mean"], profile["std"], bangia_k),
            "Impact Cost": L.sqrt_impact(p["Value"], sigma, p["Quantity"], stressed_adv, impact_y),
            "Amihud": float(amihud.iloc[-1]) if len(amihud) else np.nan,
            "Band": lock["band"], "Band Source": lock["source"], "Lower-Circuit Days": lock["circuit_days"],
            "Longest Run": lock["longest_run"], "Freeze Days": lock["freeze_days"],
            "Circuit Loss": lock["loss_pct"] * p["Value"],
        })
    holdings = pd.DataFrame(rows)

    # 90% ranges: capacity figures over the past year of rolling volume; spread cost by bootstrapping the monthly
    # spreads; impact over the past year of rolling volume at the stress factor
    adv63 = pd.DataFrame({t: L.rolling_adv(frames[t].set_index("Date"), L.AMFI_ADV_DAYS) for t in frames}).dropna()
    amfi50_range = _percentiles([L.amfi_stress(positions, row.to_dict(), 0.50, amfi_participation, amfi_exclude)["days"]
                                 for _, row in adv63.iterrows()])
    adv60 = pd.DataFrame({t: L.rolling_adv(frames[t].set_index("Date"), 60) for t in frames}).dropna()
    tickers = positions["Ticker"].to_numpy()
    sell5_range = _percentiles([L.sellable_share(positions["Value"], L.days_to_liquidate(
        positions["Quantity"], adv60.loc[d, tickers].to_numpy(), participation))[5] for d in adv60.index])
    rng = np.random.default_rng(0)
    spread_lo = spread_hi = impact_lo = impact_hi = 0.0
    for h in holdings.to_dict("records"):
        monthly = spreads[h["Ticker"]]["monthly"].to_numpy()
        if len(monthly) >= 3:
            boot = rng.choice(monthly, (500, len(monthly)))
            lo, hi = _percentiles(L.bangia_cost(h["Value"], b.mean(), b.std(ddof=1), bangia_k) for b in boot)
        else:
            lo = hi = h["Spread Cost"]
        spread_lo, spread_hi = spread_lo + lo, spread_hi + hi
        factor = h["Stress Factor"] if np.isfinite(h["Stress Factor"]) else 1.0
        path = adv60[h["Ticker"]] if h["Ticker"] in adv60 else []
        lo, hi = _percentiles(L.sqrt_impact(h["Value"], h["Sigma"], h["Quantity"], a * factor, impact_y) for a in path)
        impact_lo += lo if np.isfinite(lo) else h["Impact Cost"]
        impact_hi += hi if np.isfinite(hi) else h["Impact Cost"]

    return {"capacity": capacity, "sellable": sellable, "amfi": amfi, "stressed": stressed, "holdings": holdings,
            "spreads": spreads, "amihud": {t: L.amihud(frames[t]) for t in frames},
            "ranges": {"amfi50": amfi50_range, "sell5": sell5_range, "spread": (spread_lo, spread_hi),
                       "impact": (impact_lo, impact_hi)}}


def compute_liquidity(ctx):
    """Trading capacity, AMFI stress test, stressed volume, costs, LVaR, Amihud and circuit-lock risk."""
    positions, frames = ctx.positions, ctx.price_frames
    # Crisis-volume ratios are relative to the months before each crisis, so the primary listing's long history
    # (already downloaded for the crisis replay) is enough; no second exchange download is needed
    long_frames = {}
    for t in frames:
        res = cached_fetch(t, period="max")
        if res["success"]:
            long_frames[t] = res["df"]
    res = cached_liquidity(positions, frames, long_frames, load_scenarios(market=ctx.market_key),
                           _official_bands(ctx.disclosures), ctx.participation, ctx.amfi_participation, ctx.amfi_exclude,
                           ctx.bangia_k, ctx.impact_y, ctx.user_spread_pct, ctx.freeze_override)
    holdings = res["holdings"]
    headline_var = ctx.trusted[ctx.headline_model]["VaR"]
    circuit_total = float(holdings["Circuit Loss"].sum())
    waterfall = L.lvar_waterfall(headline_var.value, float(holdings["Spread Cost"].sum(min_count=1)),
                                 float(holdings["Impact Cost"].sum(min_count=1)), circuit_total)
    lvar = float(waterfall["Cumulative"].iloc[-1])
    add_on = float(waterfall["Amount"].iloc[-1])
    ranges = res["ranges"]
    lvar_lo = headline_var.low + ranges["spread"][0] + ranges["impact"][0] + add_on
    lvar_hi = headline_var.high + ranges["spread"][1] + ranges["impact"][1] + add_on

    data_q = float(min(q["score"] for q in ctx.quality.values())) if ctx.quality else np.nan
    volume_days = int(min(frames[t]["Volume"].notna().sum() if "Volume" in frames[t] else 0 for t in frames))
    banded = holdings["Band"].notna()
    inferred = (holdings["Band Source"].str.startswith("inferred") & banded).any()
    floored = bool(ctx.freeze_override) or ((holdings["Freeze Days"] > holdings["Longest Run"]) & banded).any()
    sources = [ctx.data_note, "volume: " + "; ".join(" + ".join(v) for v in ctx.volume_sources.values())]

    def trusted(name, value, low, high, assumptions, n_inputs, has_range=True):
        if has_range:
            width = (high - low) / abs(value) if value and np.isfinite(low) and np.isfinite(high) else np.nan
        else:
            width = None
        letter, reasons = grade(width, None, None, data_q, volume_days, len(assumptions) / n_inputs,
                                sample_label="days of volume data")
        return TrustedMetric(name, value, low, high, letter, reasons, sources, assumptions)

    sellable = res["sellable"]
    metrics = {
        "amfi50": trusted("Days to liquidate 50% (AMFI method)", res["amfi"][(0.50, True)]["days"],
                          *res["ranges"]["amfi50"], [], 3),
        "sell5": trusted("Share sellable in 5 days", sellable[5], *res["ranges"]["sell5"],
                         [f"participation rate {ctx.participation:.0%}"], 2),
        "lvar": trusted("Liquidity-adjusted VaR", lvar, lvar_lo, lvar_hi,
                        [f"Bangia k = {ctx.bangia_k:g}", f"impact constant Y = {ctx.impact_y:g}"], 5),
        "circuit": trusted("Circuit-lock loss", circuit_total, np.nan, np.nan,
                           (["band inferred from history"] if inferred else []) +
                           (["freeze length set by the floor of 3 or your override"] if floored else []), 2,
                           has_range=False),
    }
    capacity = res["capacity"]
    export(ctx, {"liquidity_capacity": capacity, "liquidity_sellable": sellable,
                 "liquidity_slowest": capacity.loc[capacity["Days to Liquidate"].idxmax()],
                 "liquidity_amfi": res["amfi"], "liquidity_stressed": res["stressed"], "liquidity_holdings": holdings,
                 "liquidity_spreads": res["spreads"], "liquidity_waterfall": waterfall, "liquidity_metrics": metrics,
                 "liquidity_amihud": res["amihud"]})
