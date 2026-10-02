"""
Integration (Phase 6): one stress scenario hits every pillar together, and the most plausible scenario that
produces a given loss.

Linked stress engine, per holding and scenario, a price fixed point (holding_loss):
    market shock (historical replay, or downside beta × the market's fall)
    → events: below the pledge margin-call trigger, lenders sell enough shares to restore the cover
    → liquidity: that selling and our own exit move the PRICE by the permanent part of square-root impact, at the
      crisis's volume (never above normal) and volatility; a banded stock whose fall reaches its band is locked for
      its exit-freeze scenario (in a replay, only the lock days beyond those already in the replayed path)
    → the lower price can breach the trigger or the band again: repeat until the price settles.
    → credit: Merton re-solved at the final equity value, reported as a signal with a jump-to-default scenario,
      never added to an equity holder's loss (the equity price already carries the default risk).
    → crisis correlation: reported alongside (it shapes the next day's tail, not the scenario loss itself).
Interaction = full loss − (market alone + each link alone with the market), and an exact Shapley split of the
full loss across market, liquidity, credit and events.

Reverse stress test: the most plausible shock (minimum Mahalanobis distance) that loses L, in asset space and in
macro-factor space, with the probability of the loss, how extreme the scenario is, and the nearest historical analogue.

Methods: docs/methodology.md section 13.
"""

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, norm, t as t_dist

import credit as C
import events as E
import liquidity as L
from stress import downside_beta, market_drawdown, replay_return

PLAYERS = ("market", "liquidity", "credit", "events")
ALL_ON = {p: True for p in PLAYERS}
PERMANENT_SHARE = 2 / 3  # assumption: impact decays to about 2/3 of its peak (Farmer et al., 2013; Bershova and Rakhlin, 2013)
MAX_ROUNDS = 20
TOLERANCE = 0.001        # stop when the price moves less than 0.1% in a round
JTD_DD = 1.5             # assumption: jump-to-default is shown when the stressed distance to default is below this
JTD_RECOVERY = (0.10, 0.0)  # assumption: equity recovery in default, shown as a range
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
                path = r.loc[(r.index > dd["peak"]) & (r.index <= dd["trough"])]  # the days replay_return compounds
                per[t] = {"return": actual, "method": "replay", "sigma": float(inside.std()) if len(inside) > 10 else np.nan,
                          "path": path}
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

def impact_fraction(sigma: float, shares: float, daily_volume: float, y: float) -> float:
    """Square-root law as a fraction of price: min(1, Y·σ·√(shares / volume)); 0 without volume or volatility."""
    if not (shares > 0) or not (daily_volume > 0) or not np.isfinite(sigma):
        return 0.0
    return float(min(1.0, y * sigma * np.sqrt(shares / daily_volume)))


def holding_loss(h: dict, sc: dict, params: dict, on: dict) -> dict:
    """
    One holding through the price fixed point with the players in `on` switched on:
        x_m = 1 + r (1 if market is off);  each round, at the current price ratio x:
        forced  = shares lenders sell to restore the pledge cover at price P₀·x       (events)
        sold    = forced + our own exit quantity                                       (liquidity)
        x_perm  = x_m·(1 − θ·I(sold))             θ = permanent share of square-root impact
        x_next  = x_perm·(1 − band)^extra if the fall 1 − x_perm reaches the band      (liquidity)
    until |x_next/x − 1| < TOLERANCE or MAX_ROUNDS. x only falls round to round and forced selling is capped at
    the pledged shares, so it converges. Our loss = V₀ − proceeds, proceeds = V₀·x*·(1 − (1−θ)·I(sold*)) − spread
    cost when we exit (liquidity on), else V₀·x*: the permanent part of impact is in x*, the temporary part is the
    cost of our own execution.
    """
    v0, p0 = h["Value"], h["Price"]
    ret = sc["return"] if on.get("market", True) else 0.0
    sigma, y = sc["sigma"], params["impact_y"]
    theta = params.get("permanent_share", PERMANENT_SHARE)
    ratio = h["Volume Ratio"]
    adv_s = h["ADV"] * (min(ratio, 1.0) if np.isfinite(ratio) else 1.0)
    band = h["Band"]
    # Lower-circuit days inside a replayed path are already in the market loss (from the return alone: no high/low)
    already = 0
    if on.get("market", True) and np.isfinite(band) and sc.get("path") is not None:
        already = int(L.lower_circuit_days(pd.DataFrame({"Returns": sc["path"]}), band).sum())
    extra = max(0, int(h["Freeze Days"]) - already) if np.isfinite(band) and np.isfinite(h["Freeze Days"]) else 0
    own = h["Quantity"] if on["liquidity"] else 0.0
    pledged = h["Pledged Shares"] if on["events"] and np.isfinite(h["Pledged Shares"]) else 0.0
    x_m = 1.0 + ret
    x, forced, locked, converged = x_m, 0.0, False, False
    for rounds in range(1, MAX_ROUNDS + 1):
        forced = E.forced_sale(pledged, p0, p0 * x, params["initial_cover"], params["trigger_cover"]) if pledged > 0 else 0.0
        x_perm = x_m * (1 - theta * impact_fraction(sigma, forced + own, adv_s, y))
        locked = bool(on["liquidity"] and extra > 0 and 1 - x_perm >= band - 1e-12)
        x_next = x_perm * ((1 - band) ** extra if locked else 1.0)
        done = x <= 0 or abs(x_next / x - 1) < TOLERANCE
        x = x_next
        if done:
            converged = True
            break
    sold_impact = impact_fraction(sigma, forced + own, adv_s, y)
    value_after = v0 * x
    if on["liquidity"]:
        spread = L.bangia_cost(value_after, h["Spread Mean"], h["Spread Std"], params["bangia_k"]) \
            if np.isfinite(h["Spread Mean"]) else 0.0
        proceeds = value_after * (1 - (1 - theta) * sold_impact) - spread
    else:
        proceeds = value_after
    return {"loss": v0 - proceeds, "x": x, "x_market": x_m, "forced": forced, "locked": locked, "rounds": rounds,
            "converged": converged, "already": already}


def _subsets(players):
    return [frozenset(c) for n in range(len(players) + 1) for c in combinations(players, n)]


def shapley(values: dict, players) -> dict:
    """Exact Shapley values from v(S) for every subset S of `players` (v(∅) included)."""
    n = len(players)
    out = {}
    for p in players:
        others = [q for q in players if q != p]
        out[p] = sum(factorial(len(s)) * factorial(n - len(s) - 1) / factorial(n) * (values[s | {p}] - values[s])
                     for s in _subsets(others))
    return out


def linked_stress(holdings: pd.DataFrame, scenario: dict, params: dict, switches: dict = None) -> dict:
    """
    One scenario through every pillar. `holdings` has one row per ticker with: Value, Quantity, Price, ADV,
    Volume Ratio (that crisis's volume ÷ normal, NaN if unknown), Spread Mean, Spread Std, Band, Freeze Days,
    Equity (market cap), Default Point, PD (today), Financial, Pledged Shares. `params`: bangia_k, impact_y, r, T,
    initial_cover, trigger_cover, and optionally permanent_share and jtd_dd. `switches` turns players off (all
    links off = plain market stress).

    Per holding: the loss with every active player on (holding_loss), its exact Shapley split across the players,
    and the interaction = full loss − [market alone + Σ each link alone with the market]: the cross effect only.
    Credit is a signal for equity holders (C2): Merton is re-solved at the final equity value, and below the
    jump-to-default threshold the loss if the equity goes to a 10% or 0% recovery is reported, never added.
    """
    on = {**ALL_ON, **(switches or {})}
    players = [p for p in PLAYERS if on[p]]
    links = [p for p in players if p != "market"]
    base = frozenset({"market"}) if on["market"] else frozenset()
    rows = []
    for h in holdings.to_dict("records"):
        t = h["Ticker"]
        sc = scenario["holdings"][t]
        runs = {s: holding_loss(h, sc, params, {p: p in s for p in PLAYERS}) for s in _subsets(players)}
        values = {s: r["loss"] for s, r in runs.items()}
        parts = shapley(values, players)
        full = runs[frozenset(players)]
        interaction = values[frozenset(players)] - values[base] - sum(values[base | {p}] - values[base] for p in links)
        dd = pd_ = np.nan
        if on["credit"] and not h["Financial"] and np.isfinite(h["Equity"]) and np.isfinite(h["Default Point"]):
            stressed = C.solve_merton(h["Equity"] * full["x"], sc["sigma"] * np.sqrt(C.TRADING_DAYS), h["Default Point"],
                                      params["r"], params["T"])
            dd, pd_ = stressed["DD"], stressed["PD"]
        jtd = np.isfinite(dd) and dd < params.get("jtd_dd", JTD_DD)
        rows.append({"Ticker": t, "Return": sc["return"], "Method": sc["method"],
                     "Market Loss": values[base], **{p.capitalize(): parts.get(p, 0.0) for p in PLAYERS},
                     "Total": values[frozenset(players)], "Interaction": interaction,
                     "Final Price": h["Price"] * full["x"], "Feedback Fall": full["x"] / full["x_market"] - 1,
                     "Forced Shares": full["forced"], "Locked": full["locked"], "Locked Days in Replay": full["already"],
                     "Rounds": full["rounds"], "Converged": full["converged"], "Stressed DD": dd, "Stressed PD": pd_,
                     "JTD Loss (10% recovery)": h["Value"] * (1 - JTD_RECOVERY[0]) if jtd else np.nan,
                     "JTD Loss (0% recovery)": h["Value"] * (1 - JTD_RECOVERY[1]) if jtd else np.nan})
    table = pd.DataFrame(rows)
    totals = table[["Market Loss", "Market", "Liquidity", "Credit", "Events", "Total", "Interaction"]].sum()
    totals["Rounds"] = int(table["Rounds"].max())
    totals["Converged"] = bool(table["Converged"].all())
    return {"name": scenario["name"], "kind": scenario["kind"], "market": scenario["market"], "table": table,
            "totals": totals}


def run_linked(holdings: pd.DataFrame, scenarios: list, params: dict, switches: dict = None) -> pd.DataFrame:
    """Linked totals for every scenario: plain market loss, Shapley parts, interaction and feedback rounds."""
    rows = []
    for sc in scenarios:
        res = linked_stress(holdings, sc, params, switches)
        tot = res["totals"]
        rows.append({"Scenario": sc["name"], "Kind": sc["kind"], "Market Move": sc["market"],
                     "Market Loss": tot["Market Loss"], "Market": tot["Market"], "Liquidity": tot["Liquidity"],
                     "Credit": tot["Credit"], "Events": tot["Events"], "Linked Total": tot["Total"],
                     "Interaction": tot["Interaction"], "Rounds": tot["Rounds"], "Converged": tot["Converged"],
                     "detail": res["table"]})
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


def _valid_nu(nu) -> bool:
    return nu is not None and np.isfinite(nu) and nu > 2


def loss_probability(loss: float, portfolio_sigma: float, nu: float = None) -> dict:
    """
    Probability of the loss itself, P(wᵀx ≤ −L), over the covariance's horizon (one month here):
    Φ(−L/σ_p) under a normal; under a multivariate Student-t whose COVARIANCE is Σ the portfolio return is
    univariate t with scale σ_p·√((ν−2)/ν), so P = T_ν(−(L/σ_p)·√(ν/(ν−2))). Also "once every N years" = 1/(12p).
    """
    d = loss / portfolio_sigma
    out = {"normal": float(norm.cdf(-d))}
    out["student_t"] = float(t_dist.cdf(-d * np.sqrt(nu / (nu - 2)), nu)) if _valid_nu(nu) else np.nan
    for key in ("normal", "student_t"):
        p = out[key]
        out[f"{key}_years"] = 1 / (12 * p) if np.isfinite(p) and p > 0 else np.nan
    return out


def plausibility(d2: float, k: int, nu: float = None) -> dict:
    """
    Share of outcomes at least this extreme IN ANY DIRECTION (not the probability of the loss; see
    loss_probability): P(D² ≥ d²), χ²(k) under a multivariate normal. Under a multivariate Student-t, d² measured
    with the covariance is rescaled to the dispersion matrix (× ν/(ν−2)), and then D²/k ~ F(k, ν).
    """
    out = {"normal": float(chi2.sf(d2, k))}
    out["student_t"] = float(f_dist.sf(d2 * nu / (nu - 2) / k, k, nu)) if _valid_nu(nu) else np.nan
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
