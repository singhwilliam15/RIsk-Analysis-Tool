"""
Decisions (Phase 6): what changed, what to do about it, and whether the portfolio is within its limits.

- Risk-change explanation: the change in ES between two snapshots, split exactly (Shapley values) between
  positions, volatility, correlation, the data window and the model.
- Best risk-reducing trades: marginal (Euler) ES per holding; the three trades that cut ES most, each with its
  effect on liquidity, credit and event exposure; the ES-minimising index-futures hedge (measurement only).
- Limits: utilisation of editable limits across the pillars, with traffic lights.
- Top risks and actions for the CRO dashboard and memo, from explicit rules.

Methods: docs/methodology.md section 14.
"""

import itertools
import json
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from var_calculator import fit_student_t, student_t_var_es

LIMITS_PATH = Path(__file__).parent / "config" / "limits.json"
PLAYERS = ("positions", "volatility", "correlation", "window", "model")
MODELS = ("Historical", "Parametric (Normal)", "Student-t")
GRADE_ORDER = {"A": 1, "B": 2, "C": 3, "D": 4}
GREEN, AMBER, RED = "green", "amber", "red"


# ---------------------------------------------------------------
# ES on a return sample
# ---------------------------------------------------------------

def portfolio_es(portfolio_returns: np.ndarray, alpha: float, model: str = "Historical") -> float:
    """1-day ES (positive loss, fraction of value) of a portfolio return sample under one of MODELS."""
    r = np.asarray(portfolio_returns, dtype=float)
    if model == "Historical":
        cutoff = np.percentile(r, alpha * 100)
        return float(-r[r <= cutoff].mean())
    if model == "Parametric (Normal)":
        z = norm.ppf(1 - alpha)
        return float(norm.pdf(z) / alpha * r.std(ddof=1) - r.mean())
    if model == "Student-t":
        fit = fit_student_t(r)
        return float(student_t_var_es(alpha, fit["nu"], fit["loc"], fit["scale"])[1])
    raise ValueError(f"Unknown model {model}")


def recolor(returns: pd.DataFrame, sigma: pd.Series, corr: pd.DataFrame) -> np.ndarray:
    """
    The window's return shapes with chosen volatilities and correlations: standardise each asset, whiten with the
    window's own correlation (Cholesky), re-correlate with `corr`, and rescale by `sigma`. With the window's own
    σ and correlation it gives back the window's (demeaned) returns.
    """
    R = returns.to_numpy(dtype=float)
    mu, sd = R.mean(axis=0), R.std(axis=0, ddof=1)
    Z = (R - mu) / sd
    own = np.corrcoef(Z, rowvar=False) if Z.shape[1] > 1 else np.ones((1, 1))
    white = Z @ np.linalg.inv(np.linalg.cholesky(own)).T
    colored = white @ np.linalg.cholesky(corr.to_numpy()).T
    return colored * sigma.to_numpy() + mu


def snapshot(returns: pd.DataFrame, weights: pd.Series, model: str, as_of, investment: float, confidence_level: float) -> dict:
    """Everything the risk-change explanation needs about one point in time (JSON-serialisable via to_json)."""
    returns = returns[weights.index].dropna()
    return {"as_of": pd.Timestamp(as_of), "tickers": list(weights.index), "weights": weights.astype(float),
            "returns": returns, "model": model if model in MODELS else "Historical", "investment": float(investment),
            "confidence_level": float(confidence_level)}


def snapshot_to_json(snap: dict) -> str:
    return json.dumps({"as_of": f"{snap['as_of']:%Y-%m-%d}", "tickers": snap["tickers"],
                       "weights": snap["weights"].to_dict(), "model": snap["model"], "investment": snap["investment"],
                       "confidence_level": snap["confidence_level"],
                       "returns": {"dates": [f"{d:%Y-%m-%d}" for d in snap["returns"].index],
                                   **{t: snap["returns"][t].round(10).tolist() for t in snap["tickers"]}}})


def snapshot_from_json(text: str) -> dict:
    raw = json.loads(text)
    returns = pd.DataFrame({t: raw["returns"][t] for t in raw["tickers"]}, index=pd.to_datetime(raw["returns"]["dates"]))
    return {"as_of": pd.Timestamp(raw["as_of"]), "tickers": raw["tickers"], "weights": pd.Series(raw["weights"], dtype=float),
            "returns": returns, "model": raw["model"], "investment": raw["investment"], "confidence_level": raw["confidence_level"]}


def _es_of_state(state: dict, alpha: float) -> float:
    R = recolor(state["window"], state["volatility"], state["correlation"])
    w = state["positions"].loc[state["window"].columns].to_numpy()
    return portfolio_es(R @ w, alpha, state["model"])


def shapley(values: dict, f) -> dict:
    """Exact Shapley values: for every player, the weighted average of its marginal effect over all coalitions."""
    names = list(values)
    n = len(names)
    cache = {}

    def v(coalition):
        key = frozenset(coalition)
        if key not in cache:
            cache[key] = f({p: values[p][1] if p in key else values[p][0] for p in names})
        return cache[key]

    out = {}
    for p in names:
        others = [q for q in names if q != p]
        total = 0.0
        for k in range(n):
            for coalition in itertools.combinations(others, k):
                weight = factorial(k) * factorial(n - k - 1) / factorial(n)
                total += weight * (v(set(coalition) | {p}) - v(coalition))
        out[p] = total
    return {"contributions": out, "start": v(set()), "end": v(set(names))}


def risk_change(old: dict, new: dict, alpha: float, investment: float) -> dict:
    """
    ES(new) − ES(old), split between the five players. Only tickers in both snapshots are compared; the old
    weights are rescaled over them. ES is in money on `investment`, with each snapshot's own returns sample.
    """
    common = [t for t in new["tickers"] if t in old["tickers"]]
    w_old = old["weights"].loc[common] / old["weights"].loc[common].sum()
    w_new = new["weights"].loc[common] / new["weights"].loc[common].sum()
    r_old, r_new = old["returns"][common].dropna(), new["returns"][common].dropna()
    values = {
        "positions": (w_old, w_new),
        "volatility": (r_old.std(ddof=1), r_new.std(ddof=1)),
        "correlation": (r_old.corr(), r_new.corr()),
        "window": (r_old, r_new),
        "model": (old["model"], new["model"]),
    }
    res = shapley(values, lambda state: _es_of_state(state, alpha) * investment)
    return {**res, "common": common, "dropped": [t for t in new["tickers"] if t not in common] +
            [t for t in old["tickers"] if t not in new["tickers"]]}


# ---------------------------------------------------------------
# Trades and hedge
# ---------------------------------------------------------------

def component_es(returns: pd.DataFrame, weights: pd.Series, alpha: float) -> pd.Series:
    """Euler (tail-conditional) ES contributions, wᵢ·E[−rᵢ | portfolio ≤ its α-quantile]; they sum to the ES."""
    R = returns[weights.index].to_numpy()
    port = R @ weights.to_numpy()
    tail = port <= np.percentile(port, alpha * 100)
    return pd.Series(-(R[tail] * weights.to_numpy()).mean(axis=0), index=weights.index)


def candidate_trades(returns: pd.DataFrame, weights: pd.Series, alpha: float, step: float = 0.05) -> pd.DataFrame:
    """
    Every sale of `step` of the portfolio from one holding into cash, and every switch of `step` from one holding
    to another, with the historical ES before and after (cash earns nothing and has no risk).
    """
    R = returns[weights.index].dropna()
    base = portfolio_es(R.to_numpy() @ weights.to_numpy(), alpha)
    rows = []
    for sell in weights.index:
        if weights[sell] <= 1e-9:
            continue
        amount = min(step, weights[sell])
        for buy in [None] + [t for t in weights.index if t != sell]:
            new = weights.copy()
            new[sell] -= amount
            if buy is not None:
                new[buy] += amount
            es = portfolio_es(R.to_numpy() @ new.to_numpy(), alpha)
            rows.append({"Trade": f"Sell {amount:.0%} of the portfolio in {sell}" + (f", buy {buy}" if buy else " into cash"),
                         "Sell": sell, "Buy": buy or "cash", "Amount": amount, "ES Before": base, "ES After": es,
                         "ES Change": es - base, "weights": new})
    return pd.DataFrame(rows).sort_values("ES Change").reset_index(drop=True)


def hedge_ratio(portfolio: pd.Series, market: pd.Series, alpha: float, max_ratio: float = 2.0) -> dict:
    """
    Short index-futures notional h (as a fraction of the portfolio's value) that minimises the historical ES of
    r_p − h·r_m over the common days; measurement only (no basis, margin or roll costs).
    """
    joined = pd.concat([portfolio.rename("p"), market.rename("m")], axis=1, join="inner").dropna()
    p, m = joined["p"].to_numpy(), joined["m"].to_numpy()
    res = minimize_scalar(lambda h: portfolio_es(p - h * m, alpha), bounds=(0, max_ratio), method="bounded",
                          options={"xatol": 1e-4})
    return {"ratio": float(res.x), "es_before": portfolio_es(p, alpha), "es_after": float(res.fun), "days": len(joined)}


# ---------------------------------------------------------------
# Limits
# ---------------------------------------------------------------

def load_limits(path: Path = LIMITS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def light(utilisation: float) -> str:
    """Green below 80% of the limit, amber from 80% to 100%, red above 100%."""
    if not np.isfinite(utilisation):
        return "not available"
    return GREEN if utilisation < 0.8 else AMBER if utilisation <= 1.0 else RED


def evaluate_limits(limits: dict, measures: dict) -> pd.DataFrame:
    """
    Utilisation = measure ÷ limit for "maximum" limits; for the minimum trust grade it is the worst grade's rank ÷ the
    limit's rank (A = 1 … D = 4), so a grade worse than the limit is over 100%.
    """
    rows = []
    for key, spec in limits["limits"].items():
        value = measures.get(key, np.nan)
        if key == "min_trust_grade":
            limit_rank = GRADE_ORDER[spec["value"]]
            rank = GRADE_ORDER.get(value, np.nan) if isinstance(value, str) else np.nan
            util = rank / limit_rank if np.isfinite(rank) else np.nan
            shown = value if isinstance(value, str) else "not available"
        else:
            util = value / spec["value"] if np.isfinite(value) and spec["value"] else np.nan
            shown = value
        rows.append({"Limit": spec["label"], "Key": key, "Value": shown, "Limit Value": spec["value"], "Unit": spec["unit"],
                     "Utilisation": util, "Status": light(util)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Top risks and actions
# ---------------------------------------------------------------

def top_risks(limits_table: pd.DataFrame, worst_scenario: dict, contributors: pd.Series, weights: pd.Series,
              tiers: dict, grades: dict, missing_data: list, n: int = 3) -> list:
    """
    Candidate risks with a severity score (explicit rules), highest first:
    red limit 100 + utilisation; worst linked stress loss 60 + loss %; amber limit 50 + utilisation;
    High-tier holding 55; a holding whose ES share is ≥ 1.25× its weight 40 + excess; grade D 45, C 25;
    missing disclosure data 20.
    """
    risks = []
    for r in limits_table.to_dict("records"):
        if r["Status"] == RED:
            risks.append((100 + 100 * r["Utilisation"], f"Limit breached: {r['Limit']} at {r['Utilisation']:.0%} of the limit"))
        elif r["Status"] == AMBER:
            risks.append((50 + 50 * r["Utilisation"], f"Close to limit: {r['Limit']} at {r['Utilisation']:.0%} of the limit"))
    if worst_scenario:
        risks.append((60 + 100 * worst_scenario["loss_pct"], f"Worst linked stress, {worst_scenario['name']}: loss of "
                      f"{worst_scenario['loss_pct']:.1%} (cross-pillar interaction {worst_scenario['interaction_pct']:+.2%})"))
    total = contributors.sum()
    for t, c in contributors.items():
        share = c / total if total else 0
        if weights[t] > 0 and share >= 1.25 * weights[t]:
            risks.append((40 + 100 * (share - weights[t]), f"{t} carries {share:.0%} of ES on {weights[t]:.0%} of the value"))
    for t, tier in tiers.items():
        if tier == "High":
            risks.append((55, f"{t} is in the High event-risk tier"))
    for name, g in grades.items():
        if g in ("C", "D"):
            risks.append((45 if g == "D" else 25, f"{name} has trust grade {g}: treat the figure with caution"))
    if missing_data:
        risks.append((20, "Disclosure data not loaded: " + ", ".join(missing_data)))
    return [text for _, text in sorted(risks, key=lambda x: -x[0])[:n]]


MIN_TRADE_CUT = 0.01  # a trade is recommended only if it cuts ES by at least 1%


def top_actions(trades: pd.DataFrame, hedge: dict, limits_table: pd.DataFrame, missing_data: list, investment: float,
                n: int = 3) -> list:
    """The best ES-reducing trade(s) that cut ES by at least 1%, the hedge if it cuts ES by more than 10%, and data to load."""
    actions = []
    for r in trades.head(2).to_dict("records"):
        if r["ES Change"] <= -MIN_TRADE_CUT * r["ES Before"]:
            actions.append(f"{r['Trade']}: ES {r['ES Change'] * investment:+,.0f} ({r['ES Change'] / r['ES Before']:+.0%})")
    if hedge and hedge["es_before"] > 0 and hedge["es_after"] < 0.9 * hedge["es_before"]:
        actions.append(f"Hedge with short index futures of {hedge['ratio']:.0%} of the value: ES "
                       f"{(hedge['es_after'] - hedge['es_before']) / hedge['es_before']:+.0%} (measurement only)")
    reds = limits_table[limits_table["Status"] == RED]
    for r in reds.to_dict("records"):
        actions.append(f"Bring \"{r['Limit']}\" back within its limit ({r['Utilisation']:.0%} used)")
    if missing_data:
        actions.append("Load the missing disclosure files so the event pillar can see pledges, surveillance and ratings")
    return actions[:n]
