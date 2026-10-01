"""
Data-driven stress testing.

Crisis windows live in stress_scenarios.csv. Each scenario's market drawdown (peak to trough
inside the window) and recovery time are measured from downloaded index prices; nothing is
hard-coded. The position is then either replayed through its own actual return over the same
peak-to-trough dates, or, if it has no price history then, estimated with its downside beta.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from var_calculator import calculate_historical_var, calculate_parametric_var

SCENARIO_FILE = Path(__file__).with_name("stress_scenarios.csv")
MARKETS = {"NIFTY50": ("^NSEI", "Nifty 50"), "SP500": ("^GSPC", "S&P 500")}
REPLAY = "Historical replay"
PROXY = "β-proxy"
NO_MARKET_DATA = "No market data"


def load_scenarios(path=SCENARIO_FILE, market: str = None) -> pd.DataFrame:
    """Read the scenario windows; optionally keep one market (NIFTY50 or SP500)."""
    table = pd.read_csv(path, parse_dates=["start", "end"])
    if market is not None:
        table = table[table["market"] == market].reset_index(drop=True)
    return table


def price_series(df: pd.DataFrame) -> pd.Series:
    """Close prices indexed by calendar date, from a fetcher DataFrame."""
    return df.set_index(df["Date"].dt.normalize())["Close"].sort_index()


def market_drawdown(prices: pd.Series, start, end) -> dict:
    """
    Largest peak-to-trough fall of `prices` inside [start, end], and the number of trading
    days after the trough until the price regained that peak (None if it never did).
    `covered` is False when the price history does not reach back to the window start.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = prices.loc[start:end]
    covered = len(window) > 1 and prices.index[0] <= start + pd.Timedelta(days=7)
    if not covered:
        return {"covered": False}
    drawdown = window / window.cummax() - 1
    trough = drawdown.idxmin()
    peak = window.loc[:trough].idxmax()
    after = prices.loc[trough:]
    regained = after[after >= window.loc[peak]]
    recovery = int(prices.index.get_loc(regained.index[0]) - prices.index.get_loc(trough)) if len(regained) else None
    return {"covered": True, "peak": peak, "trough": trough, "drawdown": float(drawdown.min()),
            "recovery_days": recovery}


def replay_return(returns: pd.Series, peak, trough):
    """
    Compounded actual return of a position from the close on `peak` to the close on `trough`.
    None when the position's history does not span the whole period.
    """
    if returns.empty or returns.index[0] > peak or returns.index[-1] < trough:
        return None
    period = returns.loc[(returns.index > peak) & (returns.index <= trough)]
    return float(np.prod(1 + period.to_numpy()) - 1)


def downside_beta(stock_returns: pd.Series, market_returns: pd.Series, quantile: float = 0.10) -> float:
    """
    Crisis beta: the position's sensitivity to the market on the market's worst `quantile` of days,
    β = Σ r_s·r_m / Σ r_m² (regression through the origin on those days). Normal-period beta can
    understate how much a stock falls in a sell-off.

    An OLS slope with an intercept on the same days is far less stable, because the truncated market
    returns span a narrow range: on real NSE/US stocks its bootstrap standard error was 2–4× larger.
    """
    joined = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    worst = joined[joined.iloc[:, 1] <= joined.iloc[:, 1].quantile(quantile)]
    if len(worst) < 20:
        return float("nan")
    s, m = worst.iloc[:, 0].to_numpy(), worst.iloc[:, 1].to_numpy()
    return float((s * m).sum() / (m * m).sum())


def run_scenarios(position_returns: pd.Series, market_prices: pd.Series, scenarios: pd.DataFrame,
                  proxy_beta: float, investment: float) -> pd.DataFrame:
    """
    One row per scenario. Market drawdown and recovery are measured from `market_prices`;
    the position's return is its actual compounded return over the market's peak-to-trough
    dates when it has data, otherwise market drawdown × `proxy_beta` (capped at −100%).
    """
    rows = []
    for sc in scenarios.itertuples():
        dd = market_drawdown(market_prices, sc.start, sc.end)
        row = {"Scenario": sc.scenario, "Window": f"{sc.start:%b %Y} – {sc.end:%b %Y}", "Description": sc.description}
        if not dd["covered"]:
            rows.append({**row, "Method": NO_MARKET_DATA})
            continue
        actual = replay_return(position_returns, dd["peak"], dd["trough"])
        if actual is not None:
            position_return, method = actual, REPLAY
        else:
            position_return, method = max(dd["drawdown"] * proxy_beta, -1.0), PROXY
        rows.append({
            **row,
            "Peak": dd["peak"], "Trough": dd["trough"],
            "Market Drawdown": dd["drawdown"],
            "Market Recovery (days)": dd["recovery_days"],
            "Method": method,
            "Position Return": position_return,
            "P&L": position_return * investment,
            "Post-Shock Value": investment * (1 + position_return),
        })
    return pd.DataFrame(rows)


def volatility_shock(returns: pd.Series, investment: float, confidence_level: float, holding_period: int = 1,
                     multipliers=(1, 2, 3)) -> pd.DataFrame:
    """
    VaR and ES with volatility scaled by k around the mean: r' = μ + k·(r − μ).
    Historical figures scale exactly by k on the loss side; the normal model gives the same picture analytically.
    """
    mu = returns.mean()
    rows = []
    for k in multipliers:
        shocked = mu + k * (returns - mu)
        hist = calculate_historical_var(shocked, investment, confidence_level, holding_period)
        param = calculate_parametric_var(shocked, investment, confidence_level, holding_period)
        rows.append({"Volatility": f"σ × {k}", "Historical VaR": hist["var_scaled_amount"], "Historical ES": hist["cvar_scaled_amount"],
                     "Normal VaR": param["var_scaled_amount"], "Normal ES": param["cvar_scaled_amount"]})
    return pd.DataFrame(rows)
