"""
Concentration and factor risk pillar (Pillar 4): what the portfolio is really exposed to, and how much of its
diversification survives a crisis.

- Factor regressions per holding (Fama-French 3 + momentum, Indian or US data) with Newey-West standard errors;
  single-index fallback on the benchmark; rolling one-year betas.
- Portfolio factor risk: weighted betas and an Euler split of variance and parametric VaR into each factor and
  specific risk (the parts add up exactly).
- Concentration: HHI, effective number of holdings, sector weights and risk contributions, PCA and Meucci's
  (2009) effective number of independent bets.
- Crisis correlation: correlations in the full sample, in crisis windows and on the market's worst days, the
  portfolio ES under each, and the share of the diversification benefit kept in a crisis.

Methods and sources: docs/methodology.md section 11.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from factor_data import FACTOR_COLUMNS, excess_returns

MIN_FACTOR_DAYS = 120       # fewer overlapping days with the factor file: fall back to the single-index model
ROLLING_WINDOW, ROLLING_STEP = 252, 21
WORST_MARKET_SHARE = 0.10   # "the market's worst 10% of days"
MIN_CRISIS_DAYS = 30
SPECIFIC = "Specific"


# ---------------------------------------------------------------
# Regression with Newey-West standard errors
# ---------------------------------------------------------------

def newey_west_lags(n: int) -> int:
    """Newey and West (1994) rule of thumb: floor(4·(n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols(y, X, lags: int = None) -> dict:
    """
    OLS of y on X (X already includes a constant column), with plain and Newey-West (1987) HAC standard errors.
    NW: V = (XᵀX)⁻¹ S (XᵀX)⁻¹, S = Σ uₜ²xₜxₜᵀ + Σₗ (1 − l/(L+1)) Σₜ uₜuₜ₋ₗ (xₜxₜ₋ₗᵀ + xₜ₋ₗxₜᵀ), Bartlett weights.
    """
    y, X = np.asarray(y, dtype=float), np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    lags = newey_west_lags(n) if lags is None else lags
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, lags + 1):
        gamma = xu[lag:].T @ xu[:-lag]
        S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    cov_nw = xtx_inv @ S @ xtx_inv
    sigma2 = u @ u / (n - k)
    tss = ((y - y.mean()) ** 2).sum()
    return {"params": beta, "se": np.sqrt(np.diag(xtx_inv) * sigma2), "se_nw": np.sqrt(np.diag(cov_nw)),
            "resid": u, "r2": 1 - (u @ u) / tss if tss > 0 else np.nan, "nobs": n, "lags": lags,
            "resid_var": float(u.var(ddof=k))}


# ---------------------------------------------------------------
# Factor regressions
# ---------------------------------------------------------------

def _regression_result(model: str, names: list, y: pd.Series, X: pd.DataFrame) -> dict:
    fit = ols(y.to_numpy(), np.column_stack([np.ones(len(X)), X.to_numpy()]))
    betas = pd.Series(fit["params"][1:], index=names)
    factor_cov = X.cov().to_numpy()
    factor_var = float(betas.to_numpy() @ factor_cov @ betas.to_numpy())
    return {"model": model, "betas": betas, "alpha": float(fit["params"][0]),
            "alpha_annual": float(fit["params"][0] * 252), "se_nw": pd.Series(fit["se_nw"], index=["alpha"] + names),
            "t_nw": pd.Series(fit["params"] / fit["se_nw"], index=["alpha"] + names), "r2": float(fit["r2"]),
            "resid_var": fit["resid_var"], "factor_var": factor_var, "nobs": fit["nobs"], "lags": fit["lags"],
            "start": y.index.min(), "end": y.index.max()}


def factor_regression(returns: pd.Series, factors: pd.DataFrame, min_obs: int = MIN_FACTOR_DAYS):
    """Excess return on MKT_RF, SMB, HML, MOM over the overlapping days; None if fewer than `min_obs`."""
    if factors is None:
        return None
    y = excess_returns(returns, factors)
    X = factors.loc[y.index, FACTOR_COLUMNS]
    if len(y) < min_obs:
        return None
    return _regression_result("Fama-French 3 + momentum", FACTOR_COLUMNS, y, X)


def single_index(returns: pd.Series, market: pd.Series, min_obs: int = 60):
    """Single-index (market) model on the benchmark's returns: the fallback when factor data do not overlap."""
    joined = pd.concat([returns.rename("r"), market.rename("MKT")], axis=1, join="inner").dropna()
    if len(joined) < min_obs:
        return None
    return _regression_result("Single index (benchmark)", ["MKT"], joined["r"], joined[["MKT"]])


def rolling_betas(returns: pd.Series, factors: pd.DataFrame, window: int = ROLLING_WINDOW, step: int = ROLLING_STEP):
    """One-year factor betas re-estimated every `step` days, each using only the window up to its end date."""
    if factors is None:
        return pd.DataFrame()
    y = excess_returns(returns, factors)
    X = factors.loc[y.index, FACTOR_COLUMNS]
    rows = {}
    for end in range(window, len(y) + 1, step):
        fit = ols(y.iloc[end - window:end].to_numpy(), np.column_stack([np.ones(window), X.iloc[end - window:end].to_numpy()]), lags=0)
        rows[y.index[end - 1]] = dict(zip(FACTOR_COLUMNS, fit["params"][1:]))
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------
# Portfolio factor risk
# ---------------------------------------------------------------

def portfolio_factor_risk(weights: pd.Series, betas: pd.DataFrame, resid_var: pd.Series, factor_cov: pd.DataFrame,
                          investment: float = 1.0, confidence_level: float = 0.95) -> dict:
    """
    Portfolio betas b = Bᵀw and the Euler split of variance σ² = bᵀΣ_f b + Σ wᵢ²σ²(εᵢ): factor k contributes
    b_k(Σ_f b)_k, specific risk Σ wᵢ²σ²(εᵢ); the parts sum to σ² exactly. Parametric (zero-mean) VaR = z·σ·investment
    is split in the same proportions.
    """
    w = weights.loc[betas.index].to_numpy()
    B = betas.to_numpy()
    b = B.T @ w
    sigma_f = factor_cov.loc[betas.columns, betas.columns].to_numpy()
    factor_parts = b * (sigma_f @ b)
    specific = float((w ** 2 * resid_var.loc[betas.index].to_numpy()).sum())
    total = float(factor_parts.sum() + specific)
    parts = pd.Series(np.append(factor_parts, specific), index=list(betas.columns) + [SPECIFIC])
    var_total = norm.ppf(confidence_level) * np.sqrt(total) * investment
    return {"betas": pd.Series(b, index=betas.columns), "variance": total, "parts": parts, "shares": parts / total,
            "var_parts": parts / total * var_total, "var_total": var_total,
            "factor_share": float(factor_parts.sum() / total)}


# ---------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------

def hhi(weights) -> float:
    """Herfindahl-Hirschman index Σw² (weights normalised to sum to one)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float((w ** 2).sum())


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Euler shares of portfolio variance by holding: wᵢ(Σw)ᵢ / wᵀΣw; they sum to one."""
    w = weights.loc[cov.index].to_numpy()
    sigma = cov.to_numpy()
    contrib = w * (sigma @ w)
    return pd.Series(contrib / contrib.sum(), index=cov.index)


def sector_view(weights: pd.Series, cov: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Sector weights and sector shares of portfolio variance (the holdings' Euler shares summed by sector)."""
    contrib = risk_contributions(weights, cov)
    table = pd.DataFrame({"Sector": [sectors.get(t) or "not available" for t in cov.index],
                          "Weight": weights.loc[cov.index].to_numpy(), "Risk Share": contrib.to_numpy()})
    return table.groupby("Sector", as_index=False).sum().sort_values("Risk Share", ascending=False).reset_index(drop=True)


def pca_bets(weights: pd.Series, cov: pd.DataFrame) -> dict:
    """
    Principal components of the asset covariance matrix. Reports the first component's share of total variance
    (Σλ) and Meucci's (2009) effective number of bets: the portfolio's variance splits across the uncorrelated
    components as pₖ = λₖ(eₖᵀw)² / wᵀΣw, and ENB = exp(−Σ pₖ ln pₖ) (1 for one bet, N for N equal bets).
    """
    sigma = cov.to_numpy()
    w = weights.loc[cov.index].to_numpy()
    eigval, eigvec = np.linalg.eigh(sigma)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]
    p = eigval * (eigvec.T @ w) ** 2
    p = p / p.sum()
    nonzero = p[p > 1e-12]
    return {"eigenvalues": eigval, "first_share": float(eigval[0] / eigval.sum()), "portfolio_shares": p,
            "enb": float(np.exp(-(nonzero * np.log(nonzero)).sum())), "first_portfolio_share": float(p[0])}


# ---------------------------------------------------------------
# Crisis correlation
# ---------------------------------------------------------------

def _normal_es(weights: np.ndarray, vols: np.ndarray, corr: np.ndarray, alpha: float) -> float:
    cov = np.outer(vols, vols) * corr
    return float(np.sqrt(weights @ cov @ weights) * norm.pdf(norm.ppf(1 - alpha)) / alpha)


def crisis_correlation(asset_returns: pd.DataFrame, market: pd.Series, scenarios: pd.DataFrame, weights: pd.Series,
                       investment: float = 1.0, confidence_level: float = 0.95, worst: float = WORST_MARKET_SHARE) -> dict:
    """
    Correlations on the full sample, inside the crisis windows (all windows pooled) and on the market's worst
    `worst` share of days; for each, the zero-mean normal ES with full-sample volatilities (so only correlation
    changes), and the share of the diversification benefit (Σ standalone ES − portfolio ES) that is kept.
    """
    r = asset_returns[weights.index].dropna()
    w = weights.to_numpy()
    alpha = 1 - confidence_level
    vols = r.std(ddof=1).to_numpy()
    standalone = float((w * vols).sum() * norm.pdf(norm.ppf(confidence_level)) / alpha)
    in_crisis = pd.Series(False, index=r.index)
    for s in scenarios.to_dict("records"):
        in_crisis |= (r.index >= pd.Timestamp(s["start"])) & (r.index <= pd.Timestamp(s["end"]))
    m = market.reindex(r.index).dropna()
    worst_days = m[m <= m.quantile(worst)].index if len(m) else pd.DatetimeIndex([])
    samples = {"Full sample": r, "Crisis windows": r[in_crisis], f"Market's worst {worst:.0%} of days": r.loc[worst_days]}
    rows, corrs = [], {}
    for name, sample in samples.items():
        if len(sample) < MIN_CRISIS_DAYS:
            rows.append({"Sample": name, "Days": len(sample), "Average Correlation": np.nan, "ES": np.nan, "Benefit Kept": np.nan})
            continue
        corr = sample.corr().to_numpy()
        corrs[name] = pd.DataFrame(corr, index=r.columns, columns=r.columns)
        es = _normal_es(w, vols, corr, alpha)
        off = corr[~np.eye(len(corr), dtype=bool)]
        rows.append({"Sample": name, "Days": len(sample), "Average Correlation": float(off.mean()) if off.size else np.nan,
                     "ES": es * investment, "Benefit": (standalone - es) * investment})
    table = pd.DataFrame(rows)
    full_benefit = table.loc[table["Sample"] == "Full sample", "Benefit"].iloc[0] if "Benefit" in table else np.nan
    table["Benefit Kept"] = table.get("Benefit", np.nan) / full_benefit if full_benefit else np.nan
    return {"table": table, "correlations": corrs, "standalone_es": standalone * investment}


# ---------------------------------------------------------------
# Bootstrap ranges
# ---------------------------------------------------------------

def bootstrap_concentration(asset_returns: pd.DataFrame, weights: pd.Series, factors: pd.DataFrame = None,
                            n_boot: int = 200, seed: int = 0) -> dict:
    """
    90% ranges for the effective number of bets and the factor share of variance, by resampling days with the
    stationary block bootstrap (trust.py) and recomputing both on each resample.
    """
    from trust import _interval, block_length, stationary_bootstrap_indices
    r = asset_returns[weights.index].dropna()
    rng = np.random.default_rng(seed)
    block, _ = block_length(r.mean(axis=1).to_numpy())
    idx = stationary_bootstrap_indices(len(r), n_boot, block, rng)
    enb, shares = [], []
    joined = None
    if factors is not None:
        joined = r.join(factors[FACTOR_COLUMNS + ["RF"]], how="inner")
        if len(joined) < MIN_FACTOR_DAYS:
            joined = None
    for row in idx:
        sample = r.iloc[row]
        enb.append(pca_bets(weights, sample.cov())["enb"])
    if joined is not None:
        j_idx = stationary_bootstrap_indices(len(joined), n_boot, block, rng)
        for row in j_idx:
            sample = joined.iloc[row]
            X = np.column_stack([np.ones(len(sample)), sample[FACTOR_COLUMNS].to_numpy()])
            betas, resid = [], []
            for t in weights.index:
                fit = ols((sample[t] - sample["RF"]).to_numpy(), X, lags=0)
                betas.append(fit["params"][1:])
                resid.append(fit["resid_var"])
            res = portfolio_factor_risk(weights, pd.DataFrame(betas, index=weights.index, columns=FACTOR_COLUMNS),
                                        pd.Series(resid, index=weights.index), sample[FACTOR_COLUMNS].cov())
            shares.append(res["factor_share"])
    return {"enb": _interval(enb), "factor_share": _interval(shares) if shares else (np.nan, np.nan)}
