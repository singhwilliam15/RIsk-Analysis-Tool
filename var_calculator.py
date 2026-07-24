"""
Value at Risk (VaR) & Financial Risk Calculator Module
Replicates and extends the exact quantitative methodology from VaR_Risk_Management_Tool.xlsx
"""

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
    norm = NormFallback()


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

def perform_kupiec_backtest(returns: pd.Series, var_daily_pct: float, confidence_level: float) -> dict:
    """
    Kupiec Proportion of Failures (POF) Likelihood Ratio Test.
    Checks if number of actual breaches matches expected model exceedances.
    """
    T = len(returns)
    alpha = 1.0 - confidence_level
    expected_failures = alpha * T
    
    # A breach occurs when actual loss > VaR (i.e. return < -var_daily_pct)
    actual_breaches = int((returns < -var_daily_pct).sum())
    breach_rate = actual_breaches / T if T > 0 else 0
    
    # Kupiec Likelihood Ratio (LR) Test Statistic
    # LR = 2 * ln[ ( (N/T)^N * (1 - N/T)^(T-N) ) / ( alpha^N * (1-alpha)^(T-N) ) ]
    if 0 < actual_breaches < T:
        p_hat = actual_breaches / T
        lr_stat = 2 * (
            actual_breaches * np.log(p_hat / alpha) +
            (T - actual_breaches) * np.log((1 - p_hat) / (1 - alpha))
        )
    else:
        lr_stat = 0.0
        
    chi_sq_critical_95 = 3.841  # 1 degree of freedom Chi-Square at 95% confidence
    test_result = "PASS - Model Accurate" if lr_stat < chi_sq_critical_95 else "FAIL - Model Inaccurate"
    
    # Basel II Traffic Light System (standardized to 250 days equivalent)
    scaled_breaches_250 = int(round(actual_breaches * (250 / T))) if T > 0 else actual_breaches
    if scaled_breaches_250 <= 4:
        traffic_light = "🟢 GREEN ZONE"
        status_desc = "No capital surcharge. Model is statistically valid."
    elif scaled_breaches_250 <= 9:
        traffic_light = "🟡 YELLOW ZONE"
        status_desc = "Increasing capital surcharge. Model may understate risk."
    else:
        traffic_light = "🔴 RED ZONE"
        status_desc = "Critical failure. Model fundamentally flawed or market experiencing severe structural shift."
        
    return {
        "total_observations": T,
        "confidence_level": confidence_level,
        "expected_failures": expected_failures,
        "actual_breaches": actual_breaches,
        "breach_rate": breach_rate,
        "lr_stat": lr_stat,
        "chi_sq_critical": chi_sq_critical_95,
        "test_result": test_result,
        "traffic_light": traffic_light,
        "status_desc": status_desc,
        "scaled_breaches_250": scaled_breaches_250
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
