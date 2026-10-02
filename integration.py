"""
Integration (Phase 6): one stress scenario hits every pillar together, and the most plausible scenario that
produces a given loss.

Linked stress engine, per holding and scenario:
    market loss (historical replay, or downside beta × the market's fall)
    → liquidity: volume falls to its level in that crisis (never above normal), so the exit takes longer and
      costs more (spread + square-root impact at stressed volatility); a banded stock that fell by at least its
      band is assumed locked for its exit-freeze scenario
    → credit: the lower equity value and higher volatility go back into Merton; the rise in PD on the remaining
      value is the credit deterioration (loss given default 100% for equity holders)
    → events: if the fall reaches the pledge margin-call trigger, lenders' selling adds price impact
    → crisis correlation: reported alongside (it shapes the next day's tail, not the scenario loss itself).
The "siloed" figure adds the separate pillars' standalone headlines as an analyst reading each page would;
the difference is the interaction effect.

Reverse stress test: the most plausible shock (minimum Mahalanobis distance) that loses L, in asset space and in
macro-factor space, with its plausibility and the nearest historical analogue.

Methods: docs/methodology.md section 13.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist

import credit as C
import events as E
import liquidity as L
from stress import downside_beta, market_drawdown, replay_return

ALL_ON = {"liquidity": True, "credit": True, "events": True}
CUSTOM_SHOCKS = (-0.10, -0.20, -0.30)
REVERSE_HORIZON = 21  # trading days: reverse-stress losses are over one month


# ---------------------------------------------------------------
# Scenario returns per holding
# ---------------------------------------------------------------

def scenario_inputs(holding_returns: dict, market_prices: pd.Series, scenarios: pd.DataFrame, today_sigma: dict,
                    lookback_returns: dict) -> list:
    """
    For each historical window: the market's peak and trough, each holding's return between them (replay, or
    downside beta × drawdown when the holding has no prices then) and its daily volatility inside the window
    (or today's volatility × the market's volatility ratio). Then the custom market shocks, with volatility
    scaled by the median crisis-to-normal volatility ratio of the market across the windows.
    """
    market_returns = market_prices.pct_change().dropna()
    out, vol_ratios = [], []
    for s in scenarios.to_dict("records"):
        dd = market_drawdown(market_prices, s["start"], s["end"])
        if not dd["covered"]:
            continue
        window = market_returns.loc[s["start"]:s["end"]]
        before = market_returns[market_returns.index < pd.Timestamp(s["start"])].tail(250)
        ratio = float(window.std() / before.std()) if len(before) > 60 and before.std() > 0 else np.nan
        if np.isfinite(ratio):
            vol_ratios.append(ratio)
        per = {}
        for t, r in holding_returns.items():
            actual = replay_return(r, dd["peak"], dd["trough"])
            if actual is not None:
                inside = r.loc[s["start"]:s["end"]]
                per[t] = {"return": actual, "method": "replay", "sigma": float(inside.std()) if len(inside) > 10 else np.nan}
            else:
                beta = downside_beta(lookback_returns[t], market_returns)
                beta = beta if np.isfinite(beta) else 1.0
                per[t] = {"return": max(dd["drawdown"] * beta, -1.0), "method": "β-proxy", "sigma": np.nan}
            if not np.isfinite(per[t]["sigma"]):
                per[t]["sigma"] = today_sigma[t] * (ratio if np.isfinite(ratio) else 1.0)
        out.append({"name": s["scenario"], "kind": "historical", "market": dd["drawdown"], "holdings": per,
                    "peak": dd["peak"], "trough": dd["trough"]})
    scale = float(np.median(vol_ratios)) if vol_ratios else 2.0
    for shock in CUSTOM_SHOCKS:
        per = {}
        for t in holding_returns:
            beta = downside_beta(lookback_returns[t], market_returns)
            beta = beta if np.isfinite(beta) else 1.0
            per[t] = {"return": max(shock * beta, -1.0), "method": "β-proxy", "sigma": today_sigma[t] * scale}
        out.append({"name": f"Market {shock:.0%}", "kind": "custom", "market": shock, "holdings": per})
    return out


# ---------------------------------------------------------------
# Linked engine
# ---------------------------------------------------------------

def linked_stress(holdings: pd.DataFrame, scenario: dict, params: dict, switches: dict = None) -> dict:
    """
    One scenario through every pillar. `holdings` has one row per ticker with: Value, Quantity, ADV, Volume Ratio
    (that crisis's volume ÷ normal, NaN if unknown), Spread Mean, Spread Std, Band, Freeze Days, Equity (market cap),
    Default Point, PD (today), Financial, Pledged Shares. `params`: bangia_k, impact_y, r, T, initial_cover,
    trigger_cover. `switches` turns the liquidity / credit / event links off (all off = plain market stress).
    """
    on = {**ALL_ON, **(switches or {})}
    rows = []
    for h in holdings.to_dict("records"):
        t = h["Ticker"]
        sc = scenario["holdings"][t]
        ret, sigma = sc["return"], sc["sigma"]
        after = h["Value"] * (1 + ret)
        market = -ret * h["Value"]
        liq = cred = event = 0.0
        ratio = h["Volume Ratio"]
        adv_s = h["ADV"] * (min(ratio, 1.0) if np.isfinite(ratio) else 1.0)
        if on["liquidity"]:
            spread = L.bangia_cost(after, h["Spread Mean"], h["Spread Std"], params["bangia_k"]) if np.isfinite(h["Spread Mean"]) else 0.0
            impact = L.sqrt_impact(after, sigma, h["Quantity"], adv_s, params["impact_y"])
            impact = impact if np.isfinite(impact) else 0.0
            locked = 0.0
            if np.isfinite(h["Band"]) and ret <= -h["Band"]:
                locked = after * (1 - (1 - h["Band"]) ** h["Freeze Days"])
            liq = spread + impact + locked
        if on["credit"] and not h["Financial"] and np.isfinite(h["Equity"]) and np.isfinite(h["Default Point"]):
            stressed = C.solve_merton(h["Equity"] * (1 + ret), sigma * np.sqrt(C.TRADING_DAYS), h["Default Point"],
                                      params["r"], params["T"])
            if np.isfinite(stressed["PD"]):
                cred = max(0.0, stressed["PD"] - (h["PD"] if np.isfinite(h["PD"]) else 0.0)) * after
        if on["events"] and np.isfinite(h["Pledged Shares"]) and h["Pledged Shares"] > 0:
            fall = 1 - params["trigger_cover"] / params["initial_cover"]
            if ret <= -fall:
                mc = E.margin_call(h["Pledged Shares"], h["Price"] * (1 + ret), adv_s, params["initial_cover"],
                                   params["trigger_cover"])
                forced = L.sqrt_impact(after, sigma, mc["shares_to_restore"], adv_s, params["impact_y"])
                event = forced if np.isfinite(forced) else 0.0
        rows.append({"Ticker": t, "Return": ret, "Method": sc["method"], "Market Loss": market, "Liquidity": liq,
                     "Credit": cred, "Events": event, "Total": market + liq + cred + event})
    table = pd.DataFrame(rows)
    totals = table[["Market Loss", "Liquidity", "Credit", "Events", "Total"]].sum()
    return {"name": scenario["name"], "kind": scenario["kind"], "market": scenario["market"], "table": table,
            "totals": totals}


def siloed_sum(market_loss: float, standalone: dict) -> float:
    """What adding the separate pages' headlines gives: scenario market loss + each pillar's standalone figure."""
    return float(market_loss + sum(v for v in standalone.values() if np.isfinite(v)))


def run_linked(holdings: pd.DataFrame, scenarios: list, params: dict, standalone: dict, switches: dict = None) -> pd.DataFrame:
    """Linked totals for every scenario, next to the siloed sum and the interaction effect."""
    rows = []
    for sc in scenarios:
        res = linked_stress(holdings, sc, params, switches)
        tot = res["totals"]
        siloed = siloed_sum(tot["Market Loss"], standalone)
        rows.append({"Scenario": sc["name"], "Kind": sc["kind"], "Market Move": sc["market"],
                     "Market Loss": tot["Market Loss"], "+ Liquidity": tot["Liquidity"], "+ Credit": tot["Credit"],
                     "+ Events": tot["Events"], "Linked Total": tot["Total"], "Siloed Sum": siloed,
                     "Interaction": tot["Total"] - siloed, "detail": res["table"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Reverse stress test
# ---------------------------------------------------------------

def reverse_stress(weights: np.ndarray, cov: np.ndarray, loss: float) -> dict:
    """
    The most plausible return vector x (minimum xᵀΣ⁻¹x) with wᵀx = −loss: x* = −loss·Σw / (wᵀΣw).
    Its squared Mahalanobis distance is d² = loss² / (wᵀΣw).
    """
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    var = float(w @ sigma @ w)
    x = -loss * sigma @ w / var
    return {"shock": x, "d2": loss ** 2 / var, "portfolio_sigma": np.sqrt(var)}


def plausibility(d2: float, k: int, nu: float = None) -> dict:
    """
    Probability of a move at least this far from the centre: χ²(k) under a multivariate normal, and
    1 − F_{k,ν}(d²/k) under a multivariate Student-t with ν degrees of freedom (where d²/k ~ F(k, ν)).
    """
    out = {"normal": float(chi2.sf(d2, k))}
    out["student_t"] = float(f_dist.sf(d2 / k, k, nu)) if nu is not None and np.isfinite(nu) and nu > 2 else np.nan
    return out


def ols_betas(y: pd.Series, X: pd.DataFrame) -> tuple:
    """Slopes of y on X (with an intercept) and the R²."""
    joined = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    A = np.column_stack([np.ones(len(joined)), joined[X.columns].to_numpy()])
    coef, *_ = np.linalg.lstsq(A, joined["y"].to_numpy(), rcond=None)
    resid = joined["y"].to_numpy() - A @ coef
    r2 = 1 - resid.var() / joined["y"].to_numpy().var()
    return pd.Series(coef[1:], index=X.columns), float(r2)


def macro_reverse(portfolio: pd.Series, macro: pd.DataFrame, loss: float, horizon: int = REVERSE_HORIZON,
                  fit_days: int = 504) -> dict:
    """
    Portfolio returns regressed on the macro factors (daily, over the last `fit_days` common days); the most
    plausible `horizon`-day macro move that loses `loss` through those betas is y* = −loss·Σ_h b / (bᵀΣ_h b), with
    Σ_h = horizon × the daily covariance. The nearest historical analogue is the past `horizon`-day window (over the
    whole common history) of actual macro moves closest to y* in Mahalanobis distance; the portfolio's actual
    return over that window is reported.
    """
    joined = pd.concat([portfolio.rename("portfolio"), macro], axis=1, join="inner").dropna()
    recent = joined.tail(fit_days)
    b, r2 = ols_betas(recent["portfolio"], recent[macro.columns])
    cov_h = recent[macro.columns].cov().to_numpy() * horizon
    rev = reverse_stress(b.to_numpy(), cov_h, loss)
    shock = pd.Series(rev["shock"], index=macro.columns)
    rolling = np.expm1(np.log1p(joined).rolling(horizon).sum()).dropna()  # compounded horizon-day moves
    inv = np.linalg.pinv(cov_h)
    diffs = rolling[macro.columns].to_numpy() - shock.to_numpy()
    distance = np.einsum("ij,jk,ik->i", diffs, inv, diffs)
    best = int(np.argmin(distance)) if len(distance) else None
    analogue = None
    if best is not None:
        end = rolling.index[best]
        analogue = {"end": end, "start": joined.index[joined.index.get_loc(end) - horizon + 1],
                    "moves": rolling.iloc[best][macro.columns], "portfolio_return": float(rolling.iloc[best]["portfolio"]),
                    "distance": float(np.sqrt(distance[best]))}
    return {"betas": b, "shock": shock, "d2": rev["d2"], "analogue": analogue, "r2": r2, "days": len(joined)}
