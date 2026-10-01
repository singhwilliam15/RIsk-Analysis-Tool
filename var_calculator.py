"""
Value at Risk (VaR) & Financial Risk Calculator Module
VaR and Expected Shortfall models, multi-day scaling, out-of-sample backtests
(Kupiec, Christoffersen, Basel traffic light, tick loss), beta and stress testing.
"""

import math

import numpy as np
import pandas as pd

from garch import (
    PCT,
    fhs_var_es,
    fit_garch_t,
    garch_filter,
    garch_variance_term_structure,
    simulate_garch_paths,
    standardized_t_es,
    standardized_t_quantile,
)

try:
    from scipy.stats import norm
except ImportError:
    from statistics import NormalDist
    class NormFallback:
        @staticmethod
        def ppf(p):
            return NormalDist().inv_cdf(p)
        @staticmethod
        def pdf(x):
            return NormalDist().pdf(x)
        @staticmethod
        def cdf(x):
            return NormalDist().cdf(x)
    norm = NormFallback()


def norm_cdf(x: float) -> float:
    return float(norm.cdf(x))


def compute_returns(prices: pd.Series, log_returns: bool = False) -> pd.Series:
    """Compute simple or log daily returns from close price series."""
    if log_returns:
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = prices.pct_change().dropna()
    return returns

def calculate_portfolio_statistics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """
    Compute core return and risk metrics.
    `risk_free_rate` is an annual rate, converted to a daily rate by compounding.
    """
    n_obs = len(returns)
    if n_obs < 2:
        return {}

    daily_mean = returns.mean()
    daily_vol = returns.std(ddof=1)
    # CAGR from the actual compounded path; the arithmetic mean overstates growth when returns are volatile
    total_growth = float(np.prod(1 + returns.to_numpy()))
    cagr = total_growth ** (252 / n_obs) - 1 if total_growth > 0 else -1.0
    ann_mean_return = daily_mean * 252
    ann_vol = daily_vol * np.sqrt(252)

    # Risk-adjusted ratios
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    sharpe_ratio = (daily_mean - rf_daily) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0

    # Sortino: downside deviation below the risk-free target, averaged over ALL days
    downside_deviation = float(np.sqrt(np.mean(np.minimum(returns.to_numpy() - rf_daily, 0.0) ** 2)))
    sortino_ratio = (daily_mean - rf_daily) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0

    # Cumulative returns and max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Coefficient of Variation (Volatility / Mean)
    coef_variation = daily_vol / abs(daily_mean) if daily_mean != 0 else np.nan

    return {
        "n_obs": n_obs,
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
        "cagr": cagr,
        "ann_mean_return": ann_mean_return,
        "ann_vol": ann_vol,
        "risk_free_rate": risk_free_rate,
        "sharpe_ratio": sharpe_ratio,
        "downside_deviation": downside_deviation,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "min_return": returns.min(),
        "max_return": returns.max(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "coef_variation": coef_variation
    }


SCALING_SQRT_TIME = "√t × 1-day quantile"
SCALING_PARAMETRIC = "z·σ·√t − μ·t"


def _scaled_result(var_daily_pct: float, cvar_daily_pct: float, investment: float,
                   confidence_level: float, holding_period: int, mu: float = None, **extra) -> dict:
    """
    Common VaR/ES result dictionary with one multi-day scaling rule for every model.

    Parametric models pass their daily mean `mu`: the daily VaR is z·σ − μ, so over t days
    the volatility term grows with √t and the mean with t:  VaR_t = z·σ·√t − μ·t (same for ES).
    Empirical models (Historical, Monte Carlo) pass no mean and scale the 1-day quantile by √t.
    """
    t = holding_period
    if mu is None:
        scaling_rule = SCALING_SQRT_TIME
        var_t = var_daily_pct * np.sqrt(t)
        cvar_t = cvar_daily_pct * np.sqrt(t)
    else:
        scaling_rule = SCALING_PARAMETRIC
        var_t = (var_daily_pct + mu) * np.sqrt(t) - mu * t
        cvar_t = (cvar_daily_pct + mu) * np.sqrt(t) - mu * t
        extra["mu"] = mu
    return {
        "confidence_level": confidence_level,
        "var_daily_pct": var_daily_pct,
        "var_daily_amount": var_daily_pct * investment,
        "cvar_daily_pct": cvar_daily_pct,
        "cvar_daily_amount": cvar_daily_pct * investment,
        "holding_period": holding_period,
        "scaling_rule": scaling_rule,
        "var_scaled_pct": var_t,
        "cvar_scaled_pct": cvar_t,
        "var_scaled_amount": var_t * investment,
        "cvar_scaled_amount": cvar_t * investment,
        **extra
    }


def _empirical_var_es(sample, alpha: float):
    """Loss quantile and average loss beyond it, both as positive numbers."""
    sample = np.asarray(sample)
    cutoff = np.percentile(sample, alpha * 100)
    tail = sample[sample <= cutoff]
    return -cutoff, (-tail.mean() if len(tail) > 0 else -cutoff)


def calculate_overlapping_historical_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int) -> dict:
    """
    Empirical t-day VaR/ES from overlapping compounded t-day returns.
    A check on √t scaling. Overlapping windows share days, so the observations are not
    independent and the estimate is noisier than its count suggests.
    """
    window_returns = np.expm1(np.log1p(returns).rolling(holding_period).sum()).dropna()
    var_pct, cvar_pct = _empirical_var_es(window_returns, 1.0 - confidence_level)
    return {
        "holding_period": holding_period,
        "observations": len(window_returns),
        "var_pct": var_pct,
        "cvar_pct": cvar_pct,
        "var_amount": var_pct * investment,
        "cvar_amount": cvar_pct * investment,
    }


def calculate_historical_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Calculate Historical (Non-Parametric) VaR and Expected Shortfall (CVaR).
    Empirical percentile of daily returns; multi-day figures use √t scaling, with the
    overlapping empirical t-day VaR alongside for comparison.
    """
    alpha = 1.0 - confidence_level
    var_daily_pct, cvar_daily_pct = _empirical_var_es(returns, alpha)
    extra = {"alpha": alpha, "percentile_return": -var_daily_pct}
    if holding_period > 1:
        overlapping = calculate_overlapping_historical_var(returns, investment, confidence_level, holding_period)
        extra.update({
            "overlapping_var_amount": overlapping["var_amount"],
            "overlapping_cvar_amount": overlapping["cvar_amount"],
            "overlapping_observations": overlapping["observations"],
        })
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period, **extra)


def calculate_parametric_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Calculate Parametric (Variance-Covariance) VaR assuming Normal Distribution.
    1-day: VaR = z·σ − μ, ES = φ(z)/α·σ − μ.  t-day: z·σ·√t − μ·t.
    """
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z_score = norm.ppf(confidence_level)
    alpha = 1.0 - confidence_level

    var_daily_pct = z_score * sigma - mu
    cvar_daily_pct = norm.pdf(z_score) / alpha * sigma - mu
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          mu=mu, z_score=z_score, sigma=sigma)


MIN_TAIL_DRAWS = 50


SCALING_GARCH = "GARCH variance term structure"
SCALING_SIMULATED = "simulated t-day GARCH-t paths"


def _set_horizon(result: dict, var_t: float, cvar_t: float, investment: float, scaling_rule: str) -> dict:
    """Overwrite the multi-day figures for models that build their own t-day distribution."""
    result.update({
        "scaling_rule": scaling_rule,
        "var_scaled_pct": var_t,
        "cvar_scaled_pct": cvar_t,
        "var_scaled_amount": var_t * investment,
        "cvar_scaled_amount": cvar_t * investment,
    })
    return result


def fit_models(returns: pd.Series) -> dict:
    """
    Fit the estimated models once, so they can be reused across confidence levels:
    GARCH(1,1)-t (also drives Monte Carlo) and the Student-t maximum-likelihood fit.
    """
    return {"garch": fit_garch_t(returns), "student_t": fit_student_t(returns)}


def calculate_monte_carlo_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                              num_simulations: int = 5000, seed: int = None, fit: dict = None) -> dict:
    """
    Monte Carlo VaR/ES from the fitted GARCH(1,1)-t process. Multi-day figures come from simulated
    t-day paths with volatility updating along each path, not from √t. If GARCH does not converge,
    falls back to sampling a fitted normal distribution (flagged with fallback=True).
    Uses a local random generator, so there is no global seed side effect.
    """
    fit = fit if fit is not None else fit_garch_t(returns)
    alpha = 1.0 - confidence_level
    tail_draws = int(np.floor(num_simulations * alpha))
    common = {"num_simulations": num_simulations, "tail_draws": tail_draws, "few_tail_draws": tail_draws < MIN_TAIL_DRAWS}

    if fit["converged"]:
        params = fit["params"]
        sigma_next = garch_filter(returns, params)[-1]
        paths = simulate_garch_paths(params, sigma_next ** 2, holding_period, num_simulations, seed)
        one_day = paths[:, 0]
        var_daily_pct, cvar_daily_pct = _empirical_var_es(one_day, alpha)
        result = _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                                simulated_returns=one_day, process="GARCH(1,1)-t", fallback=False, **common)
        horizon_returns = np.prod(1 + paths, axis=1) - 1
        var_t, cvar_t = _empirical_var_es(horizon_returns, alpha)
        return _set_horizon(result, var_t, cvar_t, investment, SCALING_SIMULATED)

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(returns.mean(), returns.std(ddof=1), num_simulations)
    var_daily_pct, cvar_daily_pct = _empirical_var_es(simulated_returns, alpha)
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          simulated_returns=simulated_returns, process="Normal (GARCH fit failed)",
                          fallback=True, fallback_reason=fit["error"], **common)


def student_t_dof(excess_kurtosis):
    """
    Method-of-moments degrees of freedom: a Student-t has excess kurtosis 6 / (nu - 4).
    Thin-tailed samples (excess kurtosis <= 0) are capped at nu = 200, which is effectively normal.
    Works on a scalar or a pandas Series. Used as the starting point and fallback for the MLE fit.
    """
    if isinstance(excess_kurtosis, pd.Series):
        return (4 + 6 / excess_kurtosis.where(excess_kurtosis > 0)).clip(upper=200).fillna(200)
    if not excess_kurtosis > 0:
        return 200.0
    return min(4 + 6 / excess_kurtosis, 200.0)


def fit_student_t(returns) -> dict:
    """
    Maximum-likelihood Student-t fit (degrees of freedom, location, scale), started from the
    method-of-moments estimate. Falls back to method of moments if the fit fails or ν ≤ 2.05.
    """
    from scipy.stats import t as student_t

    r = np.asarray(returns, dtype=float)
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    nu0 = float(np.clip(student_t_dof(float(pd.Series(r).kurtosis())), 2.5, 100))
    try:
        nu, loc, scale = student_t.fit(r, nu0, loc=mean, scale=sd * np.sqrt((nu0 - 2) / nu0))
        if np.isfinite([nu, loc, scale]).all() and nu > 2.05 and scale > 0:
            return {"nu": float(nu), "loc": float(loc), "scale": float(scale), "method": "maximum likelihood"}
    except Exception:
        pass
    nu = student_t_dof(float(pd.Series(r).kurtosis()))
    return {"nu": nu, "loc": mean, "scale": sd * np.sqrt((nu - 2) / nu), "method": "method of moments (MLE failed)"}


def student_t_var_es(alpha: float, nu: float, loc: float, scale: float):
    """Student-t VaR and closed-form ES as positive losses: ES = scale·f(q)/α·(ν + q²)/(ν − 1) − loc."""
    from scipy.stats import t as student_t
    q = student_t.ppf(1 - alpha, nu)
    return scale * q - loc, scale * student_t.pdf(q, nu) / alpha * (nu + q ** 2) / (nu - 1) - loc


def calculate_student_t_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                            fit: dict = None) -> dict:
    """
    Parametric VaR with Student-t returns, which allow fatter tails than the normal.
    Degrees of freedom, location and scale come from a maximum-likelihood fit.
    """
    fit = fit if fit is not None else fit_student_t(returns)
    nu, loc, scale = fit["nu"], fit["loc"], fit["scale"]
    var_daily_pct, cvar_daily_pct = student_t_var_es(1.0 - confidence_level, nu, loc, scale)
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          mu=loc, degrees_of_freedom=nu, loc=loc, scale=scale,
                          sigma=scale * np.sqrt(nu / (nu - 2)), fit_method=fit["method"])


def cornish_fisher_quantile(z, skew, excess_kurtosis):
    """Cornish-Fisher expansion: adjusts a normal quantile z for skewness and excess kurtosis."""
    return (z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * excess_kurtosis / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)


def cornish_fisher_is_valid(skew: float, excess_kurtosis: float) -> bool:
    """
    The expansion is a valid quantile function only if it is increasing in z. Its derivative is the
    quadratic a·z² + b·z + c with a = K/8 − S²/6, b = S/3, c = 1 − K/8 + 5S²/36, which is ≥ 0 for
    every z iff a ≥ 0 and b² − 4ac ≤ 0 (Maillard, 2012).
    """
    a = excess_kurtosis / 8 - skew ** 2 / 6
    b = skew / 3
    c = 1 - excess_kurtosis / 8 + 5 * skew ** 2 / 36
    return bool(a >= 0 and b * b - 4 * a * c <= 0)


def calculate_cornish_fisher_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Modified VaR: the normal quantile is adjusted for the sample's skewness and excess kurtosis.
    Expected Shortfall is the average Cornish-Fisher loss over the tail, integrated numerically.
    `cf_valid` is False when the skew/kurtosis pair is outside the expansion's valid region.
    """
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    skew = returns.skew()
    kurt = returns.kurtosis()
    alpha = 1.0 - confidence_level

    z_cf = cornish_fisher_quantile(norm.ppf(alpha), skew, kurt)
    var_daily_pct = -(mu + z_cf * sigma)

    # Midpoint rule over tail probabilities u in (0, alpha)
    tail_u = alpha * (np.arange(2000) + 0.5) / 2000
    tail_z = cornish_fisher_quantile(np.asarray(norm.ppf(tail_u)), skew, kurt)
    cvar_daily_pct = -(mu + tail_z.mean() * sigma)
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          mu=mu, sigma=sigma, skewness=skew, excess_kurtosis=kurt, z_cornish_fisher=z_cf,
                          cf_valid=cornish_fisher_is_valid(skew, kurt))


def ewma_volatility(returns: pd.Series, lam: float = 0.94):
    """
    RiskMetrics EWMA volatility: sigma²(t) = lam * sigma²(t-1) + (1 - lam) * r²(t-1).
    Returns the forecast for each day (using only earlier returns) and the forecast for the next day.
    """
    values = returns.to_numpy()
    variance = np.empty(len(values) + 1)
    variance[0] = np.var(values[:30], ddof=1) if len(values) >= 2 else 0.0
    for i, r in enumerate(values):
        variance[i + 1] = lam * variance[i] + (1 - lam) * r * r
    sigma = np.sqrt(variance)
    return pd.Series(sigma[:-1], index=returns.index), float(sigma[-1])


def calculate_ewma_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1, lam: float = 0.94) -> dict:
    """
    RiskMetrics VaR: normal quantile on tomorrow's EWMA volatility forecast, zero mean.
    Recent returns get more weight, so VaR reacts quickly when volatility rises.
    Multi-day figures hold tomorrow's volatility constant (z·σ·√t, the RiskMetrics convention).
    """
    _, sigma_next = ewma_volatility(returns, lam)
    z = norm.ppf(confidence_level)
    alpha = 1.0 - confidence_level
    var_daily_pct = z * sigma_next
    cvar_daily_pct = norm.pdf(z) / alpha * sigma_next
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          mu=0.0, ewma_lambda=lam, sigma_forecast=sigma_next)


def calculate_fhs_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1, lam: float = 0.94) -> dict:
    """
    Filtered Historical Simulation: divide each return by its EWMA volatility forecast, take the
    empirical quantile of these standardized residuals, and rescale by tomorrow's volatility.
    Keeps the empirical tail shape while reacting to current volatility. Multi-day uses √t.
    """
    sigma, sigma_next = ewma_volatility(returns, lam)
    standardized = (returns / sigma).to_numpy()
    var_daily_pct, cvar_daily_pct = fhs_var_es(standardized, sigma_next, 1.0 - confidence_level)
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          ewma_lambda=lam, sigma_forecast=sigma_next)


def calculate_garch_t_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                          fit: dict = None) -> dict:
    """
    GARCH(1,1) with Student-t errors: volatility clustering and fat tails together.
    1-day VaR = −(μ + σ(T+1)·q), ES = σ(T+1)·ES_t − μ, with q and ES_t from the unit-variance t.
    t-day figures use the GARCH variance term structure: σ_t² = Σ E[σ²(T+h)], h = 1..t.
    Falls back to EWMA (flagged with fallback=True) if the fit does not converge.
    """
    fit = fit if fit is not None else fit_garch_t(returns)
    if not fit["converged"]:
        result = calculate_ewma_var(returns, investment, confidence_level, holding_period)
        result.update({"fallback": True, "fallback_reason": fit["error"]})
        return result

    p = fit["params"]
    alpha = 1.0 - confidence_level
    q = standardized_t_quantile(alpha, p.nu)
    es_unit = standardized_t_es(alpha, p.nu)
    sigma_next = garch_filter(returns, p)[-1]
    var_daily_pct = -(p.mu + sigma_next * q) / PCT
    cvar_daily_pct = (sigma_next * es_unit - p.mu) / PCT
    result = _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                            fallback=False, sigma_forecast=sigma_next / PCT, garch_params=p,
                            long_run_vol=np.sqrt(p.omega / (1 - p.persistence)) / PCT)

    sigma_t = np.sqrt(garch_variance_term_structure(p, sigma_next ** 2, holding_period).sum())
    var_t = -(p.mu * holding_period + sigma_t * q) / PCT
    cvar_t = (sigma_t * es_unit - p.mu * holding_period) / PCT
    return _set_horizon(result, var_t, cvar_t, investment, SCALING_GARCH)


MODEL_ORDER = ["Historical", "Parametric (Normal)", "Student-t", "Cornish-Fisher", "EWMA (RiskMetrics)",
               "FHS (EWMA-filtered)", "GARCH(1,1)-t"]


def monte_carlo_name(num_simulations: int) -> str:
    return f"Monte Carlo (GARCH-t, {num_simulations:,} sims)"


def calculate_all_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                      num_simulations: int = 5000, seed: int = 42, fits: dict = None) -> dict:
    """Every VaR model at one confidence level, keyed by display name. Pass `fits` from fit_models to avoid refitting."""
    fits = fits if fits is not None else fit_models(returns)
    return {
        "Historical": calculate_historical_var(returns, investment, confidence_level, holding_period),
        "Parametric (Normal)": calculate_parametric_var(returns, investment, confidence_level, holding_period),
        "Student-t": calculate_student_t_var(returns, investment, confidence_level, holding_period, fit=fits["student_t"]),
        "Cornish-Fisher": calculate_cornish_fisher_var(returns, investment, confidence_level, holding_period),
        "EWMA (RiskMetrics)": calculate_ewma_var(returns, investment, confidence_level, holding_period),
        "FHS (EWMA-filtered)": calculate_fhs_var(returns, investment, confidence_level, holding_period),
        "GARCH(1,1)-t": calculate_garch_t_var(returns, investment, confidence_level, holding_period, fit=fits["garch"]),
        monte_carlo_name(num_simulations): calculate_monte_carlo_var(
            returns, investment, confidence_level, holding_period, num_simulations=num_simulations, seed=seed, fit=fits["garch"]),
    }


def rolling_historical_var(returns: pd.Series, confidence_level: float, window: int = 250) -> pd.Series:
    """
    Out-of-sample Historical VaR forecast for each day.
    VaR for day t uses only the `window` returns before t, so the forecast never sees the return it is tested on.
    """
    alpha = 1.0 - confidence_level
    return -returns.rolling(window).quantile(alpha, interpolation="linear").shift(1)


def binomial_cdf(n: int, T: int, p: float) -> float:
    """P(X <= n) for X ~ Binomial(T, p), computed in log space so large T does not overflow."""
    if n < 0:
        return 0.0
    if n >= T:
        return 1.0
    log_terms = [
        math.lgamma(T + 1) - math.lgamma(k + 1) - math.lgamma(T - k + 1)
        + k * math.log(p) + (T - k) * math.log(1 - p)
        for k in range(n + 1)
    ]
    peak = max(log_terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in log_terms))


def perform_kupiec_backtest(returns: pd.Series, var_daily_pct, confidence_level: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) Likelihood Ratio Test and Basel traffic light.
    `var_daily_pct` is either a single VaR or a Series of daily VaR forecasts aligned with `returns`;
    days without a forecast are excluded from the test.
    """
    if isinstance(var_daily_pct, pd.Series):
        valid = var_daily_pct.notna()
        returns, var_daily_pct = returns[valid], var_daily_pct[valid]

    T = len(returns)
    alpha = 1.0 - confidence_level
    expected_failures = alpha * T

    # A breach occurs when actual loss > VaR (i.e. return < -var_daily_pct)
    actual_breaches = int((returns < -var_daily_pct).sum())
    breach_rate = actual_breaches / T if T > 0 else 0

    # Kupiec Likelihood Ratio (LR) Test Statistic
    # LR = -2 * ln[ ( alpha^N * (1-alpha)^(T-N) ) / ( (N/T)^N * (1 - N/T)^(T-N) ) ]
    # Edge cases use 0 * ln(0) = 0, so zero breaches still produce a valid statistic.
    if T == 0:
        lr_stat = 0.0
    elif actual_breaches == 0:
        lr_stat = -2 * T * np.log(1 - alpha)
    elif actual_breaches == T:
        lr_stat = -2 * T * np.log(alpha)
    else:
        p_hat = actual_breaches / T
        lr_stat = 2 * (
            actual_breaches * np.log(p_hat / alpha) +
            (T - actual_breaches) * np.log((1 - p_hat) / (1 - alpha))
        )
    # LR >= 0 in theory; when the breach rate equals alpha exactly, rounding can push it slightly below 0
    lr_stat = max(float(lr_stat), 0.0)

    chi_sq_critical_95 = 3.841  # 1 degree of freedom Chi-Square at 95% confidence
    # Chi-square(1) p-value: P(chi2 > LR) = 2 * (1 - Phi(sqrt(LR)))
    p_value = float(2 * (1 - norm_cdf(np.sqrt(lr_stat))))
    test_result = "PASS - Model Accurate" if lr_stat < chi_sq_critical_95 else "FAIL - Model Inaccurate"

    # Basel traffic light, using the same cumulative-probability rule Basel used to set its
    # 99% / 250-day zones (0-4 green, 5-9 yellow, 10+ red), applied to this sample's T and alpha.
    cumulative_prob = binomial_cdf(actual_breaches, T, alpha) if T > 0 else 0.0
    if cumulative_prob < 0.95:
        traffic_light = "🟢 GREEN ZONE"
        status_desc = "Breach count is consistent with the model. No capital surcharge."
    elif cumulative_prob < 0.9999:
        traffic_light = "🟡 YELLOW ZONE"
        status_desc = "More breaches than expected. Model may understate risk."
    else:
        traffic_light = "🔴 RED ZONE"
        status_desc = "Far more breaches than expected. Model materially understates risk."

    return {
        "total_observations": T,
        "confidence_level": confidence_level,
        "expected_failures": expected_failures,
        "actual_breaches": actual_breaches,
        "breach_rate": breach_rate,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "chi_sq_critical": chi_sq_critical_95,
        "test_result": test_result,
        "cumulative_prob": cumulative_prob,
        "traffic_light": traffic_light,
        "status_desc": status_desc
    }

REFIT_EVERY = 20
GARCH_MAX_WINDOW = 1000
MONTE_CARLO_BACKTEST_NAME = "Monte Carlo (GARCH-t)"
BACKTEST_MODELS = MODEL_ORDER + [MONTE_CARLO_BACKTEST_NAME]


def rolling_forecasts(returns: pd.Series, confidence_level: float, window: int = 250, ewma_lambda: float = 0.94,
                      refit_every: int = REFIT_EVERY, num_simulations: int = 5000, seed: int = 42,
                      models=None) -> dict:
    """
    Out-of-sample one-day VaR, ES and volatility forecasts for each model (one column per model).
    The forecast for day t uses only returns before t; the first `window` days have no forecast.

    - Historical, Normal, Cornish-Fisher, FHS: rolling `window`-day estimation, updated daily.
    - Student-t: maximum-likelihood fit on the rolling window, refitted every `refit_every` days.
    - GARCH(1,1)-t and Monte Carlo: fitted on all data before the refit day (at most GARCH_MAX_WINDOW
      days, since GARCH needs more data than 250 days to estimate well), refitted every `refit_every`
      days, with σ filtered daily using the latest parameters. A non-converged fit falls back to EWMA
      for that block; the count is returned as `garch_fallbacks`.

    Returns {"var", "es", "sigma"} DataFrames plus "garch_refits" and "garch_fallbacks".
    """
    from numpy.lib.stride_tricks import sliding_window_view

    names = list(models) if models is not None else list(BACKTEST_MODELS)
    empty = lambda: pd.DataFrame(np.nan, index=returns.index, columns=names)
    var_df, es_df, sigma_df = empty(), empty(), empty()
    out = {"var": var_df, "es": es_df, "sigma": sigma_df, "garch_refits": 0, "garch_fallbacks": 0, "window": window}

    r = returns.to_numpy(dtype=float)
    n = len(r)
    if n <= window:
        return out

    alpha = 1.0 - confidence_level
    z = norm.ppf(confidence_level)
    es_normal = norm.pdf(z) / alpha
    days = np.arange(window, n)
    W = sliding_window_view(r, window)[:-1]  # row k holds the `window` returns before day window + k
    mu = W.mean(axis=1)
    sd = W.std(axis=1, ddof=1)
    ewma_sigma = ewma_volatility(returns, ewma_lambda)[0].to_numpy()
    s_ewma = ewma_sigma[days]

    def put(name, var, es, sigma):
        if name in names:
            var_df.iloc[window:, names.index(name)] = var
            es_df.iloc[window:, names.index(name)] = es
            sigma_df.iloc[window:, names.index(name)] = sigma

    def empirical(matrix):
        q = np.quantile(matrix, alpha, axis=1)
        tail = np.where(matrix <= q[:, None], matrix, np.nan)
        return -q, -np.nanmean(tail, axis=1)

    if "Historical" in names:
        put("Historical", *empirical(W), sd)
    put("Parametric (Normal)", z * sd - mu, es_normal * sd - mu, sd)

    if "Cornish-Fisher" in names:
        roll = returns.rolling(window)
        skew = roll.skew().shift(1).to_numpy()[window:]
        kurt = roll.kurt().shift(1).to_numpy()[window:]
        tail_u = alpha * (np.arange(2000) + 0.5) / 2000
        tail_z = cornish_fisher_quantile(norm.ppf(tail_u)[None, :], skew[:, None], kurt[:, None]).mean(axis=1)
        put("Cornish-Fisher", -(mu + cornish_fisher_quantile(-z, skew, kurt) * sd), -(mu + tail_z * sd), sd)

    put("EWMA (RiskMetrics)", z * s_ewma, es_normal * s_ewma, s_ewma)

    if "FHS (EWMA-filtered)" in names:
        Z = sliding_window_view(r / ewma_sigma, window)[:-1]
        q_var, q_es = empirical(Z)
        put("FHS (EWMA-filtered)", q_var * s_ewma, q_es * s_ewma, s_ewma)

    blocks = [(b, min(b + refit_every, len(days))) for b in range(0, len(days), refit_every)]

    if "Student-t" in names:
        t_var, t_es, t_sigma = (np.empty(len(days)) for _ in range(3))
        for b0, b1 in blocks:
            fit = fit_student_t(W[b0])
            v, e = student_t_var_es(alpha, fit["nu"], fit["loc"], fit["scale"])
            t_var[b0:b1], t_es[b0:b1] = v, e
            t_sigma[b0:b1] = fit["scale"] * np.sqrt(fit["nu"] / (fit["nu"] - 2))
        put("Student-t", t_var, t_es, t_sigma)

    want_garch = "GARCH(1,1)-t" in names
    want_mc = MONTE_CARLO_BACKTEST_NAME in names
    if want_garch or want_mc:
        g = {k: np.empty(len(days)) for k in ("var", "es", "sigma", "mc_var", "mc_es")}
        for b0, b1 in blocks:
            t0, t1 = window + b0, window + b1
            start = max(0, t0 - GARCH_MAX_WINDOW)
            fit = fit_garch_t(r[start:t0])
            out["garch_refits"] += 1
            if not fit["converged"]:
                out["garch_fallbacks"] += 1
                s = s_ewma[b0:b1]
                g["var"][b0:b1] = g["mc_var"][b0:b1] = z * s
                g["es"][b0:b1] = g["mc_es"][b0:b1] = es_normal * s
                g["sigma"][b0:b1] = s
                continue
            p = fit["params"]
            eps = r[start:t0] * PCT - p.mu
            s = garch_filter(r[start:t1], p, initial_variance=float(np.var(eps, ddof=1)))[t0 - start:t1 - start]
            g["var"][b0:b1] = -(p.mu + s * standardized_t_quantile(alpha, p.nu)) / PCT
            g["es"][b0:b1] = (s * standardized_t_es(alpha, p.nu) - p.mu) / PCT
            g["sigma"][b0:b1] = s / PCT
            draws = np.random.default_rng(seed).standard_t(p.nu, num_simulations) * np.sqrt((p.nu - 2) / p.nu)
            sim_var, sim_es = _empirical_var_es(draws, alpha)
            g["mc_var"][b0:b1] = (s * sim_var - p.mu) / PCT
            g["mc_es"][b0:b1] = (s * sim_es - p.mu) / PCT
        put("GARCH(1,1)-t", g["var"], g["es"], g["sigma"])
        put(MONTE_CARLO_BACKTEST_NAME, g["mc_var"], g["mc_es"], g["sigma"])

    return out


def rolling_var_forecasts(returns: pd.Series, confidence_level: float, window: int = 250, ewma_lambda: float = 0.94,
                          models=None) -> pd.DataFrame:
    """Out-of-sample one-day VaR forecasts, one column per model (see rolling_forecasts)."""
    return rolling_forecasts(returns, confidence_level, window, ewma_lambda, models=models)["var"]


def _xlogy(x: float, p: float) -> float:
    """x * ln(p), defined as 0 when x == 0 (the convention used in likelihood ratio tests)."""
    return 0.0 if x == 0 else x * np.log(p)


def christoffersen_test(hits) -> dict:
    """
    Christoffersen (1998) independence test: are VaR breaches clustered in time?
    Compares the chance of a breach after a breach (pi11) with the chance after a normal day (pi01).
    """
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    log_l_restricted = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_l_markov = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr_ind = max(-2 * (log_l_restricted - log_l_markov), 0.0)
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": pi01, "pi11": pi11,
        "lr_ind": lr_ind,
        "p_value_ind": float(2 * (1 - norm_cdf(np.sqrt(lr_ind)))),
    }


BACKTEST_WINDOW = 250
LOW_POWER = "LOW POWER"


def min_backtest_days(confidence_level: float) -> int:
    """
    Out-of-sample days needed before a PASS/FAIL verdict means anything.
    At 99% even 250 days give only ~2.5 expected breaches; at 90-95%, 100 days give 5-10.
    """
    return 250 if confidence_level >= 0.975 else 100


def tick_loss(returns: pd.Series, var_series: pd.Series, confidence_level: float) -> float:
    """
    Average quantile (tick / pinball) loss of a VaR forecast:
        L = mean[ (α − 1{r < −VaR}) · (r + VaR) ]
    The −VaR return quantile is the forecast. The loss is never negative and is lowest, in
    expectation, for the true quantile, so it ranks models; a p-value does not.
    """
    alpha = 1.0 - confidence_level
    r = returns.to_numpy(dtype=float)
    q = -var_series.to_numpy(dtype=float)
    hit = (r < q).astype(float)
    return float(np.mean((alpha - hit) * (r - q)))


MIN_ES_BREACHES = 5


def mcneil_frey_test(returns, var_forecast, es_forecast, sigma_forecast, n_boot: int = 10_000, seed: int = 0) -> dict:
    """
    McNeil & Frey (2000) exceedance-residual test of Expected Shortfall.
    On breach days the residual e = (loss − ES) / σ should have mean zero if ES is right.
    One-sided: a positive mean means losses beyond VaR are larger than the model's ES (ES too small).
    The p-value bootstraps the t-statistic of the residuals after centring them at zero.
    """
    r = np.asarray(returns, dtype=float)
    var = np.asarray(var_forecast, dtype=float)
    es = np.asarray(es_forecast, dtype=float)
    sigma = np.asarray(sigma_forecast, dtype=float)
    hits = r < -var
    e = ((-r[hits]) - es[hits]) / sigma[hits]
    n = len(e)
    if n < MIN_ES_BREACHES or not np.isfinite(e).all() or e.std(ddof=1) == 0:
        return {"breaches": n, "mean_residual": float(e.mean()) if n else float("nan"), "t_stat": float("nan"), "p_value": float("nan")}

    t_stat = e.mean() / (e.std(ddof=1) / np.sqrt(n))
    centred = e - e.mean()
    rng = np.random.default_rng(seed)
    sample = centred[rng.integers(0, n, size=(n_boot, n))]
    boot_sd = sample.std(axis=1, ddof=1)
    boot_t = np.divide(sample.mean(axis=1), boot_sd / np.sqrt(n), out=np.zeros(n_boot), where=boot_sd > 0)
    return {"breaches": n, "mean_residual": float(e.mean()), "t_stat": float(t_stat),
            "p_value": float((boot_t >= t_stat).mean())}


def backtest_all_methods(returns: pd.Series, forecasts: pd.DataFrame, confidence_level: float,
                         es_forecasts: pd.DataFrame = None, sigma_forecasts: pd.DataFrame = None) -> pd.DataFrame:
    """
    Kupiec (coverage), Christoffersen (independence) and conditional coverage tests for each model,
    plus the tick loss used to rank them, and (when ES and σ forecasts are given) the McNeil-Frey ES test.
    Conditional coverage LR = Kupiec LR + independence LR ~ chi-square(2).
    With fewer test days than `min_backtest_days`, the verdict is LOW POWER instead of PASS/FAIL.
    """
    valid = forecasts.notna().all(axis=1)
    r = returns[valid]
    enough_data = len(r) >= min_backtest_days(confidence_level)
    rows = []
    for method in forecasts.columns:
        var_series = forecasts.loc[valid, method]
        kupiec = perform_kupiec_backtest(r, var_series, confidence_level)
        ind = christoffersen_test(r < -var_series)
        lr_cc = kupiec["lr_stat"] + ind["lr_ind"]
        p_cc = float(np.exp(-lr_cc / 2))  # chi-square(2) survival function
        passed = min(kupiec["p_value"], ind["p_value_ind"], p_cc) >= 0.05
        es_cols = {}
        if es_forecasts is not None and sigma_forecasts is not None:
            mf = mcneil_frey_test(r, var_series, es_forecasts.loc[valid, method], sigma_forecasts.loc[valid, method])
            if not enough_data:
                es_verdict = LOW_POWER
            elif not np.isfinite(mf["p_value"]):
                es_verdict = "TOO FEW BREACHES"
            else:
                es_verdict = "PASS" if mf["p_value"] >= 0.05 else "FAIL"
            es_cols = {"ES p-value": mf["p_value"], "ES Mean Residual": mf["mean_residual"], "ES Test": es_verdict}
        rows.append({
            "Method": method,
            "Test Days": kupiec["total_observations"],
            "Expected Breaches": kupiec["expected_failures"],
            "Actual Breaches": kupiec["actual_breaches"],
            "Breach Rate": kupiec["breach_rate"],
            "Kupiec p-value": kupiec["p_value"],
            "Independence p-value": ind["p_value_ind"],
            "Conditional Coverage p-value": p_cc,
            "Back-to-Back Breaches": ind["n11"],
            "Traffic Light": kupiec["traffic_light"],
            "Tick Loss": tick_loss(r, var_series, confidence_level) if len(r) else float("nan"),
            "Verdict": ("PASS" if passed else "FAIL") if enough_data else LOW_POWER,
            "Avg VaR": var_series.mean(),
            **es_cols,
        })
    return pd.DataFrame(rows)


def recommend_model(backtest_table: pd.DataFrame, exclude=()) -> dict:
    """
    Recommended model = lowest tick loss among models that PASS all three tests.
    If none pass, report the lowest-loss model with a warning; with too little data, recommend nothing.
    Models in `exclude` are scored in the table but never recommended.
    """
    backtest_table = backtest_table[~backtest_table["Method"].isin(exclude)]
    if backtest_table.empty or (backtest_table["Verdict"] == LOW_POWER).all():
        return {"model": None, "status": "low_power"}
    passing = backtest_table[backtest_table["Verdict"] == "PASS"]
    if len(passing):
        return {"model": passing.loc[passing["Tick Loss"].idxmin(), "Method"], "status": "recommended"}
    return {"model": backtest_table.loc[backtest_table["Tick Loss"].idxmin(), "Method"], "status": "none_pass"}


def estimate_beta(stock_returns: pd.Series, index_returns: pd.Series) -> float:
    """OLS beta of the stock against the benchmark: Cov(stock, index) / Var(index), on matching dates."""
    joined = pd.concat([stock_returns, index_returns], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return float("nan")
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def historical_worst_losses(returns: pd.Series, investment: float, horizons=(1, 5, 10, 21)) -> pd.DataFrame:
    """The stock's own worst compounded loss over each horizon in the sample."""
    growth = np.log1p(returns)
    rows = []
    for days in horizons:
        if len(returns) < days:
            continue
        window_return = np.expm1(growth.rolling(days).sum())
        worst = window_return.min()
        rows.append({
            "Horizon": f"{days} day" + ("s" if days > 1 else ""),
            "Worst Return": worst,
            "Loss": -worst * investment,
            "Window End": window_return.idxmin(),
        })
    return pd.DataFrame(rows)
