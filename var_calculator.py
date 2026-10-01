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

def run_stress_testing(investment: float) -> pd.DataFrame:
    """
    Run predefined historical crisis scenarios against current portfolio value.
    Matches Stress Testing sheet in VaR_Risk_Management_Tool.xlsx.
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
    df_stress["Portfolio_Impact"] = investment * df_stress["Shock"]
    df_stress["Post_Shock_Value"] = investment + df_stress["Portfolio_Impact"]
    return df_stress
