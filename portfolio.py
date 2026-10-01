"""
Portfolio Module
Builds a multi-asset portfolio return series and decomposes its VaR by holding.
"""

import numpy as np
import pandas as pd

from var_calculator import calculate_historical_var, norm


def normalize_weights(holdings: pd.DataFrame) -> pd.Series:
    """
    Clean a Ticker / Weight table into weights that sum to 1, indexed by ticker.
    Blank rows are dropped and duplicate tickers are combined. Raises ValueError on invalid input.
    """
    table = holdings.copy()
    table["Ticker"] = table["Ticker"].astype(str).str.strip().str.upper()
    table["Weight"] = pd.to_numeric(table["Weight"], errors="coerce")
    table = table[(table["Ticker"] != "") & (table["Ticker"] != "NONE") & (table["Ticker"] != "NAN")]
    if table["Weight"].isna().any():
        raise ValueError("Every holding needs a numeric weight.")
    if (table["Weight"] < 0).any():
        raise ValueError("Weights must be positive; short positions are not supported.")
    weights = table.groupby("Ticker", sort=False)["Weight"].sum()
    weights = weights[weights > 0]
    if len(weights) < 2:
        raise ValueError("Enter at least two holdings with positive weights.")
    return weights / weights.sum()


def align_asset_returns(price_frames: dict) -> pd.DataFrame:
    """
    Daily returns for each asset on the dates every asset traded, one column per ticker.
    `price_frames` maps ticker -> DataFrame with Date and Close columns.
    """
    closes = {
        ticker: frame.set_index(frame["Date"].dt.normalize())["Close"]
        for ticker, frame in price_frames.items()
    }
    prices = pd.concat(closes, axis=1, join="inner").sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.pct_change().dropna()


def build_portfolio_frame(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Portfolio return series with constant weights (rebalanced daily), in the same
    Date / Close / Returns layout the single-stock pipeline uses. Close is an index starting at 100.
    """
    port_returns = asset_returns[weights.index] @ weights.to_numpy()
    level = 100 * (1 + port_returns).cumprod()
    frame = pd.DataFrame({
        "Date": port_returns.index,
        "Close": level.to_numpy(),
        "Returns": port_returns.to_numpy(),
    })
    frame["Log_Returns"] = np.log1p(frame["Returns"])
    frame["Rolling_30d_Vol"] = frame["Returns"].rolling(30).std() * np.sqrt(252)
    return frame


def component_var(asset_returns: pd.DataFrame, weights: pd.Series, investment: float,
                  confidence_level: float, holding_period: int = 1) -> pd.DataFrame:
    """
    Parametric (Euler) VaR decomposition plus each holding's standalone Historical VaR.

    Over t days, Component VaR_i = w_i * (z * (Σw)_i / σ_p * √t − μ_i * t), so the components add up
    exactly to the portfolio's parametric VaR (z * σ_p * √t − μ_p * t). Marginal VaR is reported per day.
    A negative component means the holding hedges the rest.
    """
    r = asset_returns[weights.index]
    w = weights.to_numpy()
    cov = r.cov().to_numpy()
    mu = r.mean().to_numpy()
    z = norm.ppf(confidence_level)
    sigma_p = float(np.sqrt(w @ cov @ w))
    t = holding_period

    marginal = z * (cov @ w) / sigma_p - mu
    component = w * (z * (cov @ w) / sigma_p * np.sqrt(t) - mu * t) * investment
    standalone = np.array([
        calculate_historical_var(r[t], investment * w_i, confidence_level, holding_period)["var_scaled_amount"]
        for t, w_i in zip(weights.index, w)
    ])
    asset_vol = r.std().to_numpy() * np.sqrt(252)

    table = pd.DataFrame({
        "Ticker": weights.index,
        "Weight": w,
        "Annualized Volatility": asset_vol,
        "Standalone VaR": standalone,
        "Marginal VaR": marginal,
        "Component VaR": component,
        "Contribution %": component / component.sum(),
    })
    # Risk share above 1 means the holding contributes more risk than capital
    table["Risk / Weight"] = table["Contribution %"] / table["Weight"]
    return table


def diversification_summary(asset_returns: pd.DataFrame, weights: pd.Series, investment: float,
                            confidence_level: float, holding_period: int = 1) -> dict:
    """
    Compare the sum of each holding's standalone Historical VaR with the portfolio's Historical VaR.
    The gap is the loss that diversification removes.
    """
    r = asset_returns[weights.index]
    port_returns = r @ weights.to_numpy()
    portfolio_var = calculate_historical_var(port_returns, investment, confidence_level, holding_period)["var_scaled_amount"]
    undiversified = sum(
        calculate_historical_var(r[t], investment * w, confidence_level, holding_period)["var_scaled_amount"]
        for t, w in weights.items()
    )
    corr = r.corr().to_numpy()
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    return {
        "portfolio_var": portfolio_var,
        "undiversified_var": undiversified,
        "diversification_benefit": undiversified - portfolio_var,
        "diversification_ratio": 1 - portfolio_var / undiversified if undiversified else 0.0,
        "average_correlation": float(off_diagonal.mean()) if off_diagonal.size else float("nan"),
        "common_days": len(r),
    }
