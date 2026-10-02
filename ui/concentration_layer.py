"""Concentration and factor calculations shared by every page (Pillar 4), with ranges and grades."""

import numpy as np
import pandas as pd

import concentration as K
from factor_data import FACTOR_COLUMNS, load_factors, region_for
from portfolio import align_asset_returns
from stress import load_scenarios, price_series
from trust import TrustedMetric, grade
from ui.cache import cached_concentration, cached_fetch
from ui.context import export

STALE_FACTOR_DAYS = 92   # factor data more than about 3 months older than the latest price cost a grade point
N_BOOT = 200


def analyse(returns: dict, weights: pd.Series, factors, market: pd.Series, long_returns, long_market,
            scenarios: pd.DataFrame, sectors: dict, investment: float, confidence_level: float, n_boot: int = N_BOOT) -> dict:
    """Every concentration and factor figure; a pure function of its inputs so the app can cache it."""
    fits = {t: K.factor_regression(r, factors) for t, r in returns.items()}
    if all(f is not None for f in fits.values()):
        model, cols = "Fama-French 3 + momentum", FACTOR_COLUMNS
        joined = pd.concat([r.rename(t) for t, r in returns.items()], axis=1, join="inner").join(factors, how="inner")
        factor_cov = joined[FACTOR_COLUMNS].cov()
    else:
        # The portfolio split needs one model for every holding, so one missing overlap moves all to the fallback
        fits = {t: K.single_index(r, market) for t, r in returns.items()}
        model, cols = "Single index (benchmark)", ["MKT"]
        common = pd.concat(list(returns.values()), axis=1, join="inner").index.intersection(market.index)
        factor_cov = market.loc[common].to_frame("MKT").cov()
    ok = {t: f for t, f in fits.items() if f is not None}
    out = {"model": model, "fits": fits, "factor_cov": factor_cov, "rolling": {}}
    if factors is not None and model.startswith("Fama"):
        out["rolling"] = {t: K.rolling_betas(r, factors) for t, r in returns.items()}
    if ok and len(ok) == len(weights):
        betas = pd.DataFrame({t: f["betas"] for t, f in ok.items()}).T.loc[weights.index, cols]
        resid = pd.Series({t: f["resid_var"] for t, f in ok.items()}).loc[weights.index]
        out["factor_risk"] = K.portfolio_factor_risk(weights, betas, resid, factor_cov, investment, confidence_level)
    else:
        out["factor_risk"] = None

    if len(weights) > 1:
        asset = pd.concat([returns[t].rename(t) for t in weights.index], axis=1, join="inner").dropna()
        cov = asset.cov()
        out["hhi"] = K.hhi(weights)
        out["sectors"] = K.sector_view(weights, cov, sectors)
        out["holding_risk"] = K.risk_contributions(weights, cov)
        out["pca"] = K.pca_bets(weights, cov)
        out["crisis"] = K.crisis_correlation(long_returns, long_market, scenarios, weights, investment, confidence_level) \
            if long_returns is not None and len(long_returns) else None
        out["boot"] = K.bootstrap_concentration(asset, weights, factors if model.startswith("Fama") else None, n_boot=n_boot)
    return out


def compute_concentration(ctx):
    """Factor regressions, portfolio factor risk, concentration, PCA and crisis correlation."""
    tickers = list(ctx.price_frames)
    weights = (ctx.weights_now if ctx.is_portfolio else pd.Series({ctx.symbol: 1.0})).astype(float)
    region = region_for(tickers)
    factors, meta = load_factors(region)
    returns = {t: df.set_index(df["Date"].dt.normalize())["Returns"].dropna() for t, df in ctx.price_frames.items()}
    bench = cached_fetch(ctx.benchmark_symbol, period=ctx.period_input)
    market = price_series(bench["df"]).pct_change().dropna() if bench["success"] else pd.Series(dtype=float)
    long_returns = long_market = None
    if ctx.is_portfolio:
        long_frames = {t: cached_fetch(t, period="max") for t in tickers}
        if all(r["success"] for r in long_frames.values()):
            long_returns = align_asset_returns({t: r["df"] for t, r in long_frames.items()})
        long_bench = cached_fetch(ctx.benchmark_symbol, period="max")
        long_market = price_series(long_bench["df"]).pct_change().dropna() if long_bench["success"] else pd.Series(dtype=float)
    sectors = dict(zip(ctx.positions["Ticker"], ctx.positions["Sector"]))
    res = cached_concentration(returns, weights, factors, market, long_returns, long_market,
                               load_scenarios(market=ctx.market_key), sectors, ctx.investment_amount, ctx.confidence_level)

    # --- trust -------------------------------------------------------------------------
    data_q = float(min(q["score"] for q in ctx.quality.values())) if ctx.quality else np.nan
    n_days = int(min(len(r) for r in returns.values()))
    extra = []
    factor_end = pd.Timestamp(meta["end_date"]) if meta else None
    if factor_end is not None and (ctx.prices_as_of - factor_end).days > STALE_FACTOR_DAYS:
        extra.append(("Factor data age", 1, f"factor data end {factor_end:%d %b %Y}, prices {ctx.prices_as_of:%d %b %Y}"))
    if res["model"].startswith("Single"):
        extra.append(("Factor model", 1, "fell back to the single-index model"))
    sources = [ctx.data_note] + ([f"{meta['source']} (to {meta['end_date']}): {meta['citation']}"] if meta else [])

    def metric(name, value, low, high, assumptions, n_inputs, has_range=True, label="90% range", more=()):
        width = None if not has_range else ((high - low) / abs(value) if value and np.isfinite(low) else np.nan)
        letter, reasons = grade(width, None, None, data_q, n_days, len(assumptions) / n_inputs, extra=tuple(extra) + more)
        return TrustedMetric(name, value, low, high, letter, reasons, sources, assumptions, range_label=label)

    metrics = {}
    fr = res["factor_risk"]
    if fr is not None:
        lo, hi = res.get("boot", {}).get("factor_share", (np.nan, np.nan))
        metrics["factor_share"] = metric("Factor share of variance", fr["factor_share"], lo, hi,
                                         [f"factor model: {res['model']}"], 3, has_range=len(weights) > 1)
    if len(weights) > 1:
        lo, hi = res["boot"]["enb"]
        metrics["enb"] = metric("Effective number of bets (Meucci)", res["pca"]["enb"], lo, hi, [], 1)
        metrics["first_pc"] = metric("First principal component's share", res["pca"]["first_share"], np.nan, np.nan, [], 1,
                                     has_range=False)
        crisis = res["crisis"]
        if crisis is not None:
            kept = crisis["table"].set_index("Sample")["Benefit Kept"]
            crisis_values = [v for k, v in kept.items() if k != "Full sample" and np.isfinite(v)]
            value = min(crisis_values) if crisis_values else np.nan
            metrics["kept"] = metric("Diversification kept in a crisis", value,
                                     min(crisis_values) if crisis_values else np.nan,
                                     max(crisis_values) if crisis_values else np.nan,
                                     ["crisis windows from stress_scenarios.csv", "normal ES with full-sample volatilities"],
                                     4, label="range across crisis definitions")
    export(ctx, {"conc": res, "conc_metrics": metrics, "conc_region": region, "conc_factor_meta": meta,
                 "conc_weights": weights})
