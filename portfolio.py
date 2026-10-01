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


REBALANCE_DAILY = "Rebalanced daily"
BUY_AND_HOLD = "Buy-and-hold (weights drift)"


def alignment_report(price_frames: dict) -> dict:
    """
    How the inner join on dates shortens the sample: each holding's own history, the holding
    that starts last (and so limits the sample), and how many days the longest history loses.
    """
    spans = pd.DataFrame([
        {"Ticker": t, "First Date": f["Date"].min(), "Last Date": f["Date"].max(), "Trading Days": len(f)}
        for t, f in price_frames.items()
    ])
    common = len(align_asset_returns(price_frames)) + 1  # price days on which every holding traded
    limiting = spans.loc[spans["First Date"].idxmax(), "Ticker"]
    return {
        "spans": spans,
        "common_days": common,
        "limiting_ticker": limiting,
        "days_dropped": int(spans["Trading Days"].max() - common),
    }


def drifted_weights(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """
    Buy-and-hold weights at the start of each day: w(t) ∝ w(0)·Π(1 + r) up to the previous day.
    """
    r = asset_returns[weights.index]
    value = (1 + r).cumprod().shift(1).fillna(1.0) * weights.to_numpy()
    return value.div(value.sum(axis=1), axis=0)


def current_weights(asset_returns: pd.DataFrame, weights: pd.Series, rebalance: str = REBALANCE_DAILY) -> pd.Series:
    """Today's weights: the targets if rebalanced daily, or where buy-and-hold weights have drifted to."""
    if rebalance != BUY_AND_HOLD:
        return weights
    r = asset_returns[weights.index]
    value = (1 + r).prod() * weights
    return value / value.sum()


def build_portfolio_frame(asset_returns: pd.DataFrame, weights: pd.Series, rebalance: str = REBALANCE_DAILY) -> pd.DataFrame:
    """
    Portfolio return series in the same Date / Close / Returns layout the single-stock pipeline uses.
    Close is an index starting at 100. Weights are either reset to the targets every day, or
    bought once and left to drift (buy-and-hold).
    """
    r = asset_returns[weights.index]
    if rebalance == BUY_AND_HOLD:
        port_returns = (r * drifted_weights(asset_returns, weights)).sum(axis=1)
    else:
        port_returns = r @ weights.to_numpy()
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


# ---------------------------------------------------------------
# Risk decomposition on a chosen basis, incremental VaR, what-if
# ---------------------------------------------------------------

HISTORICAL_ES = "Historical ES"
HISTORICAL_VAR = "Historical VaR"
PARAMETRIC_VAR = "Parametric VaR"
DECOMPOSITION_BASES = (HISTORICAL_ES, HISTORICAL_VAR, PARAMETRIC_VAR)


def portfolio_risk(asset_returns: pd.DataFrame, weights: pd.Series, investment: float, confidence_level: float,
                   holding_period: int = 1, basis: str = HISTORICAL_ES) -> float:
    """
    Risk of the position Σ wᵢ·rᵢ (weights need not sum to 1) on one basis, in money.
    Historical figures scale the 1-day result by √t (the Historical model's rule); parametric uses z·σ·√t − μ·t.
    """
    r = asset_returns[weights.index].to_numpy() @ weights.to_numpy()
    alpha = 1.0 - confidence_level
    t = holding_period
    if basis == PARAMETRIC_VAR:
        return float((norm.ppf(confidence_level) * r.std(ddof=1) * np.sqrt(t) - r.mean() * t) * investment)
    cutoff = np.percentile(r, alpha * 100)
    loss = -r[r <= cutoff].mean() if basis == HISTORICAL_ES else -cutoff
    return float(loss * np.sqrt(t) * investment)


def _kernel_rows(port: np.ndarray, alpha: float, half_width: int) -> np.ndarray:
    """Rows whose portfolio return ranks within ±half_width of the alpha-quantile position."""
    order = np.argsort(port, kind="stable")
    pos = alpha * (len(port) - 1)
    lo = max(int(np.floor(pos)) - half_width, 0)
    hi = min(int(np.ceil(pos)) + half_width, len(port) - 1)
    return order[lo:hi + 1]


def risk_decomposition(asset_returns: pd.DataFrame, weights: pd.Series, investment: float, confidence_level: float,
                       holding_period: int = 1, basis: str = HISTORICAL_ES) -> dict:
    """
    Euler decomposition of portfolio risk into holdings. The components always add up exactly to `total`.

    - Historical ES: Component ESᵢ = wᵢ·E[−rᵢ | portfolio return ≤ its α-quantile], the tail-conditional
      estimator; it sums exactly to the portfolio's historical ES.
    - Historical VaR: Component VaRᵢ = wᵢ·E[−rᵢ | portfolio return ≈ its α-quantile], averaged over the
      days ranked closest to the quantile (±0.25% of the sample, at least ±2 days), then rescaled so the
      components sum exactly to the portfolio's historical VaR.
    - Parametric VaR: normal Euler allocation wᵢ·(z·(Σw)ᵢ/σₚ·√t − μᵢ·t).

    Also returns each holding's standalone risk and incremental risk (portfolio risk with the holding
    minus without it, other positions unchanged), on the same basis.
    """
    tickers = list(weights.index)
    R = asset_returns[tickers].to_numpy()
    w = weights.to_numpy()
    alpha = 1.0 - confidence_level
    t = holding_period
    total = portfolio_risk(asset_returns, weights, investment, confidence_level, t, basis)

    if basis == PARAMETRIC_VAR:
        cov = np.cov(R, rowvar=False, ddof=1)
        sigma_p = float(np.sqrt(w @ cov @ w))
        per_unit = norm.ppf(confidence_level) * (cov @ w) / sigma_p * np.sqrt(t) - R.mean(axis=0) * t
        component = w * per_unit * investment
    else:
        contributions = R * w  # day-by-day P&L of each holding, as a fraction of the investment
        port = contributions.sum(axis=1)
        if basis == HISTORICAL_ES:
            rows = port <= np.percentile(port, alpha * 100)
        else:
            rows = _kernel_rows(port, alpha, max(2, int(round(0.0025 * len(port)))))
        component = -contributions[rows].mean(axis=0) * np.sqrt(t) * investment
        if basis == HISTORICAL_VAR and component.sum() != 0:
            component = component * total / component.sum()

    standalone = np.array([
        portfolio_risk(asset_returns[[tk]], pd.Series([wi], index=[tk]), investment, confidence_level, t, basis)
        for tk, wi in zip(tickers, w)
    ])
    incremental = np.array([
        total - portfolio_risk(asset_returns, weights.drop(tk), investment, confidence_level, t, basis)
        for tk in tickers
    ])
    table = pd.DataFrame({
        "Ticker": tickers,
        "Weight": w,
        "Annualized Volatility": R.std(axis=0, ddof=1) * np.sqrt(252),
        "Standalone": standalone,
        "Component": component,
        "Contribution %": component / total if total else np.nan,
        "Incremental": incremental,
    })
    table["Risk / Weight"] = table["Contribution %"] / table["Weight"]
    return {"basis": basis, "total": total, "table": table,
            "standalone_sum": float(standalone.sum()), "diversification_benefit": float(standalone.sum() - total)}


def what_if_weights(weights: pd.Series, changes: dict) -> pd.Series:
    """
    New weights after setting some holdings to the given fractions (a new ticker is added).
    The other holdings keep their relative sizes and are scaled to fill the rest. Raises ValueError
    if the requested weights are negative or add up to more than 100%.
    """
    if any(v < 0 for v in changes.values()) or sum(changes.values()) > 1 + 1e-12:
        raise ValueError("What-if weights must be between 0% and 100% and add up to at most 100%.")
    rest = weights.drop([t for t in changes if t in weights.index])
    scaled = rest / rest.sum() * (1 - sum(changes.values())) if rest.sum() > 0 else rest
    new = pd.concat([scaled, pd.Series(changes, dtype=float)])
    return new[new > 0]


def what_if(asset_returns: pd.DataFrame, weights: pd.Series, changes: dict, investment: float,
            confidence_level: float, holding_period: int = 1) -> pd.DataFrame:
    """Portfolio VaR and ES before and after a weight change, on every decomposition basis."""
    new_weights = what_if_weights(weights, changes)
    rows = []
    for basis in (HISTORICAL_VAR, HISTORICAL_ES, PARAMETRIC_VAR):
        before = portfolio_risk(asset_returns, weights, investment, confidence_level, holding_period, basis)
        after = portfolio_risk(asset_returns, new_weights, investment, confidence_level, holding_period, basis)
        rows.append({"Measure": basis, "Current": before, "What-if": after, "Change": after - before,
                     "Change %": (after - before) / before if before else np.nan})
    return pd.DataFrame(rows)
