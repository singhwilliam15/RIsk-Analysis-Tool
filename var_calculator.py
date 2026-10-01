"""
Value at Risk (VaR) & Financial Risk Calculator Module
Replicates and extends the exact quantitative methodology from VaR_Risk_Management_Tool.xlsx
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
    """Compute core portfolio return and risk metrics."""
    n_obs = len(returns)
    if n_obs < 2:
        return {}
        
    daily_mean = returns.mean()
    daily_vol = returns.std(ddof=1)
    ann_return = (1 + daily_mean) ** 252 - 1
    ann_vol = daily_vol * np.sqrt(252)
    
    # Risk-adjusted ratios
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - rf_daily
    sharpe_ratio = (daily_mean - rf_daily) / daily_vol * np.sqrt(252) if daily_vol > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std(ddof=1) if len(downside_returns) > 1 else daily_vol
    sortino_ratio = (daily_mean - rf_daily) / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
    
    # Cumulative returns and max drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # Coefficient of Variation (Volatility / Mean)
    cov = daily_vol / abs(daily_mean) if daily_mean != 0 else np.nan

    return {
        "n_obs": n_obs,
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "min_return": returns.min(),
        "max_return": returns.max(),
        "skewness": returns.skew(),
        "kurtosis": returns.kurtosis(),
        "cov": cov
    }

def calculate_historical_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Calculate Historical (Non-Parametric) VaR and Expected Shortfall (CVaR).
    Sorted actual empirical returns percentile method.
    """
    alpha = 1.0 - confidence_level
    percentile_cutoff = np.percentile(returns, alpha * 100)
    
    # Loss amount (positive number representing maximum expected loss)
    var_daily_pct = -percentile_cutoff
    var_daily_amount = var_daily_pct * investment
    
    # Expected Shortfall (CVaR) - average loss beyond percentile
    tail_returns = returns[returns <= percentile_cutoff]
    cvar_daily_pct = -tail_returns.mean() if len(tail_returns) > 0 else var_daily_pct
    cvar_daily_amount = cvar_daily_pct * investment
    
    # Holding period scaling
    holding_scale = np.sqrt(holding_period)
    var_scaled_amount = var_daily_amount * holding_scale
    cvar_scaled_amount = cvar_daily_amount * holding_scale
    
    return {
        "confidence_level": confidence_level,
        "alpha": alpha,
        "percentile_return": percentile_cutoff,
        "var_daily_pct": var_daily_pct,
        "var_daily_amount": var_daily_amount,
        "cvar_daily_pct": cvar_daily_pct,
        "cvar_daily_amount": cvar_daily_amount,
        "holding_period": holding_period,
        "var_scaled_amount": var_scaled_amount,
        "cvar_scaled_amount": cvar_scaled_amount
    }

def calculate_parametric_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1) -> dict:
    """
    Calculate Parametric (Variance-Covariance) VaR assuming Normal Distribution.
    Formula: VaR = Portfolio * (Z * sigma - mu)
    """
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    # Z-score for given confidence level (e.g., Z=1.64485 for 95%)
    z_score = norm.ppf(confidence_level)
    
    var_daily_pct = (z_score * sigma - mu)
    var_daily_amount = var_daily_pct * investment
    
    # Analytic Gaussian CVaR = Portfolio * (pdf(Z) / (1-alpha) * sigma - mu)
    alpha = 1.0 - confidence_level
    pdf_z = norm.pdf(z_score)
    cvar_daily_pct = (pdf_z / alpha * sigma - mu)
    cvar_daily_amount = cvar_daily_pct * investment
    
    # Holding period scaling
    holding_scale = np.sqrt(holding_period)
    var_scaled_amount = var_daily_amount * holding_scale
    cvar_scaled_amount = cvar_daily_amount * holding_scale
    
    return {
        "confidence_level": confidence_level,
        "z_score": z_score,
        "mu": mu,
        "sigma": sigma,
        "var_daily_pct": var_daily_pct,
        "var_daily_amount": var_daily_amount,
        "cvar_daily_pct": cvar_daily_pct,
        "cvar_daily_amount": cvar_daily_amount,
        "holding_period": holding_period,
        "var_scaled_amount": var_scaled_amount,
        "cvar_scaled_amount": cvar_scaled_amount
    }

def calculate_monte_carlo_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1, num_simulations: int = 5000, seed: int = None) -> dict:
    """
    Calculate Monte Carlo Simulation VaR by sampling simulated return paths.
    """
    if seed is not None:
        np.random.seed(seed)
        
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    
    # Simulate return paths from Normal(mu, sigma)
    simulated_returns = np.random.normal(mu, sigma, num_simulations)
    
    alpha = 1.0 - confidence_level
    percentile_cutoff = np.percentile(simulated_returns, alpha * 100)
    
    var_daily_pct = -percentile_cutoff
    var_daily_amount = var_daily_pct * investment
    
    tail_sims = simulated_returns[simulated_returns <= percentile_cutoff]
    cvar_daily_pct = -tail_sims.mean() if len(tail_sims) > 0 else var_daily_pct
    cvar_daily_amount = cvar_daily_pct * investment
    
    holding_scale = np.sqrt(holding_period)
    var_scaled_amount = var_daily_amount * holding_scale
    cvar_scaled_amount = cvar_daily_amount * holding_scale
    
    return {
        "num_simulations": num_simulations,
        "confidence_level": confidence_level,
        "simulated_returns": simulated_returns,
        "var_daily_pct": var_daily_pct,
        "var_daily_amount": var_daily_amount,
        "cvar_daily_pct": cvar_daily_pct,
        "cvar_daily_amount": cvar_daily_amount,
        "holding_period": holding_period,
        "var_scaled_amount": var_scaled_amount,
        "cvar_scaled_amount": cvar_scaled_amount
    }

def _scaled_result(var_daily_pct: float, cvar_daily_pct: float, investment: float,
                   confidence_level: float, holding_period: int, **extra) -> dict:
    """Common VaR/ES result dictionary, scaled to the holding period by sqrt(time)."""
    holding_scale = np.sqrt(holding_period)
    return {
        "confidence_level": confidence_level,
        "var_daily_pct": var_daily_pct,
        "var_daily_amount": var_daily_pct * investment,
        "cvar_daily_pct": cvar_daily_pct,
        "cvar_daily_amount": cvar_daily_pct * investment,
        "holding_period": holding_period,
        "var_scaled_amount": var_daily_pct * investment * holding_scale,
        "cvar_scaled_amount": cvar_daily_pct * investment * holding_scale,
        **extra
    }


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
                          skewness=skew, excess_kurtosis=kurt, z_cornish_fisher=z_cf)


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
    """
    _, sigma_next = ewma_volatility(returns, lam)
    z = norm.ppf(confidence_level)
    alpha = 1.0 - confidence_level
    var_daily_pct = z * sigma_next
    cvar_daily_pct = norm.pdf(z) / alpha * sigma_next
    return _scaled_result(var_daily_pct, cvar_daily_pct, investment, confidence_level, holding_period,
                          ewma_lambda=lam, sigma_forecast=sigma_next)


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


def backtest_all_methods(returns: pd.Series, forecasts: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    """
    Kupiec (coverage), Christoffersen (independence) and conditional coverage tests for each model.
    Conditional coverage LR = Kupiec LR + independence LR ~ chi-square(2).
    """
    valid = forecasts.notna().all(axis=1)
    r = returns[valid]
    rows = []
    for method in forecasts.columns:
        var_series = forecasts.loc[valid, method]
        kupiec = perform_kupiec_backtest(r, var_series, confidence_level)
        ind = christoffersen_test(r < -var_series)
        lr_cc = kupiec["lr_stat"] + ind["lr_ind"]
        p_cc = float(np.exp(-lr_cc / 2))  # chi-square(2) survival function
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
            "Verdict": "PASS" if min(kupiec["p_value"], ind["p_value_ind"], p_cc) >= 0.05 else "FAIL",
            "Avg VaR": var_series.mean(),
        })
    return pd.DataFrame(rows)


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
