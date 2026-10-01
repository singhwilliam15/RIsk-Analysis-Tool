"""
Value at Risk (VaR) & Financial Risk Calculator Module
VaR and Expected Shortfall models, multi-day scaling, out-of-sample backtests
(Kupiec, Christoffersen, Basel traffic light, tick loss), beta and stress testing.
"""

import math

import numpy as np
import pandas as pd

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


def calculate_monte_carlo_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1, num_simulations: int = 5000, seed: int = None) -> dict:
    """
    Calculate Monte Carlo Simulation VaR by sampling returns from a fitted normal distribution.
    Uses a local random generator (no global seed side effect); multi-day figures use √t scaling.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    simulated_returns = rng.normal(mu, sigma, num_simulations)

    alpha = 1.0 - confidence_level
    var_daily_pct, cvar_daily_pct = _empirical_var_es(simulated_returns, alpha)
    tail_draws = int(np.floor(num_simulations * alpha))
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          num_simulations=num_simulations, simulated_returns=simulated_returns,
                          tail_draws=tail_draws, few_tail_draws=tail_draws < MIN_TAIL_DRAWS)


def student_t_dof(excess_kurtosis):
    """
    Method-of-moments degrees of freedom: a Student-t has excess kurtosis 6 / (nu - 4).
    Thin-tailed samples (excess kurtosis <= 0) are capped at nu = 200, which is effectively normal.
    Works on a scalar or a pandas Series.
    """
    if isinstance(excess_kurtosis, pd.Series):
        return (4 + 6 / excess_kurtosis.where(excess_kurtosis > 0)).clip(upper=200).fillna(200)
    if not excess_kurtosis > 0:
        return 200.0
    return min(4 + 6 / excess_kurtosis, 200.0)


def calculate_student_t_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Parametric VaR with Student-t returns, which allow fatter tails than the normal.
    Degrees of freedom come from the sample's excess kurtosis; the scale is set so the variance matches the sample.
    """
    from scipy.stats import t as student_t

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    nu = student_t_dof(returns.kurtosis())
    scale = sigma * np.sqrt((nu - 2) / nu)
    alpha = 1.0 - confidence_level

    q = student_t.ppf(confidence_level, nu)
    var_daily_pct = scale * q - mu
    # Closed-form Student-t Expected Shortfall
    cvar_daily_pct = scale * student_t.pdf(q, nu) / alpha * (nu + q ** 2) / (nu - 1) - mu
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          degrees_of_freedom=nu, mu=mu, sigma=sigma)


def cornish_fisher_quantile(z, skew, excess_kurtosis):
    """Cornish-Fisher expansion: adjusts a normal quantile z for skewness and excess kurtosis."""
    return (z
            + (z ** 2 - 1) * skew / 6
            + (z ** 3 - 3 * z) * excess_kurtosis / 24
            - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)


def calculate_cornish_fisher_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Modified VaR: the normal quantile is adjusted for the sample's skewness and excess kurtosis.
    Expected Shortfall is the average Cornish-Fisher loss over the tail, integrated numerically.
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
                          mu=mu, sigma=sigma, skewness=skew, excess_kurtosis=kurt, z_cornish_fisher=z_cf)


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


def calculate_all_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                      num_simulations: int = 5000, seed: int = 42) -> dict:
    """Every VaR model at one confidence level, keyed by display name."""
    return {
        "Historical": calculate_historical_var(returns, investment, confidence_level, holding_period),
        "Parametric (Normal)": calculate_parametric_var(returns, investment, confidence_level, holding_period),
        "Student-t": calculate_student_t_var(returns, investment, confidence_level, holding_period),
        "Cornish-Fisher": calculate_cornish_fisher_var(returns, investment, confidence_level, holding_period),
        "EWMA (RiskMetrics)": calculate_ewma_var(returns, investment, confidence_level, holding_period),
        f"Monte Carlo ({num_simulations:,} sims)": calculate_monte_carlo_var(
            returns, investment, confidence_level, holding_period, num_simulations=num_simulations, seed=seed),
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

def rolling_var_forecasts(returns: pd.Series, confidence_level: float, window: int = 250, ewma_lambda: float = 0.94) -> pd.DataFrame:
    """
    Out-of-sample one-day VaR forecasts from each model, one column per model.
    Every forecast for day t uses only returns before t. Monte Carlo is left out because it
    samples the same normal distribution as the Parametric model.
    """
    from scipy.stats import t as student_t

    alpha = 1.0 - confidence_level
    z = norm.ppf(confidence_level)
    roll = returns.rolling(window)
    mu = roll.mean().shift(1)
    sigma = roll.std().shift(1)
    skew = roll.skew().shift(1)
    kurt = roll.kurt().shift(1)

    nu = student_t_dof(kurt)
    scale = sigma * np.sqrt((nu - 2) / nu)
    ewma_sigma, _ = ewma_volatility(returns, ewma_lambda)

    forecasts = pd.DataFrame({
        "Historical": -roll.quantile(alpha, interpolation="linear").shift(1),
        "Parametric (Normal)": z * sigma - mu,
        "Student-t": scale * student_t.ppf(confidence_level, nu) - mu,
        "Cornish-Fisher": -(mu + cornish_fisher_quantile(-z, skew, kurt) * sigma),
        "EWMA (RiskMetrics)": z * ewma_sigma,
    }, index=returns.index)
    # Score every model on the same days
    forecasts.iloc[:window] = np.nan
    return forecasts


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


def backtest_all_methods(returns: pd.Series, forecasts: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    """
    Kupiec (coverage), Christoffersen (independence) and conditional coverage tests for each model,
    plus the tick loss used to rank them.
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
        })
    return pd.DataFrame(rows)


def recommend_model(backtest_table: pd.DataFrame) -> dict:
    """
    Recommended model = lowest tick loss among models that PASS all three tests.
    If none pass, report the lowest-loss model with a warning; with too little data, recommend nothing.
    """
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


def run_stress_testing(investment: float, beta: float = 1.0) -> pd.DataFrame:
    """
    Run predefined historical crisis scenarios against current portfolio value.
    Shocks are approximate index drawdowns; each is scaled by the stock's beta to the
    benchmark (capped at a 100% loss) to estimate the stock-level impact.
    """
    scenarios = [
        {"Scenario": "COVID-19 Crash (Mar 2020)", "Shock": -0.34, "Recovery_Days": 60, "Probability": "Low", "Risk_Level": "HIGH"},
        {"Scenario": "Lehman / GFC Crash (2008)", "Shock": -0.57, "Recovery_Days": 500, "Probability": "Very Low", "Risk_Level": "HIGH"},
        {"Scenario": "Dot-com Bust (2000-02)", "Shock": -0.49, "Recovery_Days": 900, "Probability": "Very Low", "Risk_Level": "HIGH"},
        {"Scenario": "Black Monday (1987)", "Shock": -0.22, "Recovery_Days": 120, "Probability": "Very Low", "Risk_Level": "HIGH"},
        {"Scenario": "Asian Financial Crisis (1997)", "Shock": -0.18, "Recovery_Days": 180, "Probability": "Low", "Risk_Level": "MEDIUM"},
        {"Scenario": "9/11 Market Shock (2001)", "Shock": -0.12, "Recovery_Days": 30, "Probability": "Low", "Risk_Level": "MEDIUM"},
        {"Scenario": "Russian Ruble Crisis (1998)", "Shock": -0.15, "Recovery_Days": 90, "Probability": "Low", "Risk_Level": "MEDIUM"},
        {"Scenario": "Mild Bear Market", "Shock": -0.10, "Recovery_Days": 45, "Probability": "Moderate", "Risk_Level": "LOW"},
        {"Scenario": "Flash Crash Shock", "Shock": -0.05, "Recovery_Days": 2, "Probability": "Moderate", "Risk_Level": "LOW"}
    ]
    
    df_stress = pd.DataFrame(scenarios)
    df_stress["Beta"] = beta
    df_stress["Stock_Shock"] = (df_stress["Shock"] * beta).clip(lower=-1.0)
    df_stress["Portfolio_Impact"] = investment * df_stress["Stock_Shock"]
    df_stress["Post_Shock_Value"] = investment + df_stress["Portfolio_Impact"]
    return df_stress
