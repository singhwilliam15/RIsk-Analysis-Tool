"""
Data-quality score per holding (0-100), from explicit checks on the downloaded data. Feeds the Trust layer.

Each check reports what it found and a penalty; the score is 100 minus the penalties, floored at 0.
The thresholds and penalties are assumptions, documented in docs/methodology.md (section 7).
Nothing here alters the data.
"""

import numpy as np
import pandas as pd

from data_fetcher import suspicious_returns
from var_calculator import BACKTEST_WINDOW, min_backtest_days

# Assumptions: thresholds and penalties (points off 100)
REVERSED_SPIKE_PENALTY, REVERSED_SPIKE_CAP = 15, 30
STALE_RUN_DAYS = 3            # a close unchanged for this many consecutive days or more counts as stale
STALE_LIMITS = ((0.05, 20), (0.01, 10))         # (share of days above, penalty), worst first
# Zero volume with an unchanged close: on liquid NSE stocks these are mostly exchange holidays that Yahoo fills
# with the previous close (about 3 a year), so a few are tolerated; many mean days without any trade.
NO_TRADE_LIMITS = ((0.05, 20), (0.02, 10))
# Zero volume while the close moved: the price and volume data contradict each other
ZERO_VOLUME_MOVED_LIMITS = ((0.01, 10), (0.0, 5))
GAP_CALENDAR_DAYS = 7          # a gap between trading days longer than this (beyond normal holidays)
GAP_PENALTY, GAP_CAP = 5, 15
# History: one estimation window, or a window plus the out-of-sample days a 99% backtest needs (as in Backtesting)
SHORT_HISTORY = ((BACKTEST_WINDOW, 25), (BACKTEST_WINDOW + min_backtest_days(0.99), 10))  # (fewer returns than, penalty)
NO_VOLUME_PENALTY = 5
MISSING_FUNDAMENTALS_CAP = 15
# Fields whose absence matters for the later credit pillar
KEY_FUNDAMENTALS = ("revenue", "ebit", "total_assets", "total_liabilities", "current_assets", "current_liabilities",
                    "retained_earnings", "total_equity", "operating_cash_flow", "shares_outstanding")


def _tiered(share: float, limits) -> int:
    return next((penalty for threshold, penalty in limits if share > threshold), 0)


def _tiered_below(value: int, limits) -> int:
    return next((penalty for threshold, penalty in limits if value < threshold), 0)


def stale_days(close: pd.Series, run: int = STALE_RUN_DAYS) -> int:
    """Days inside runs of at least `run` consecutive identical closes (counting every day of the run)."""
    values = close.to_numpy(dtype=float)
    if len(values) == 0:
        return 0
    same = np.concatenate([[False], values[1:] == values[:-1]])
    group = np.cumsum(~same)
    sizes = pd.Series(group).value_counts()
    return int(sizes[sizes >= run].sum())


def date_gaps(dates: pd.Series, limit: int = GAP_CALENDAR_DAYS) -> pd.DataFrame:
    """Gaps between consecutive trading dates longer than `limit` calendar days."""
    d = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    days = d.diff().dt.days
    rows = days > limit
    return pd.DataFrame({"From": d.shift(1)[rows].to_numpy(), "To": d[rows].to_numpy(), "Calendar Days": days[rows].to_numpy()})


def assess_holding(df: pd.DataFrame, fundamentals: dict = None) -> dict:
    """
    Score one holding's price history (Date, Close, Returns and optionally Volume columns) and, if given,
    its fundamentals (fundamentals.fetch_fundamentals result). Returns {"score", "checks", "reasons"}:
    `checks` has one row per check (Check, Finding, Penalty) and `reasons` lists the checks that cost points.
    """
    checks = []
    n_returns = int(df["Returns"].notna().sum())

    flagged = suspicious_returns(df)
    reversed_count = int(flagged["Reversed"].sum())
    checks.append(("Reversing spikes", f"{reversed_count} one-day move(s) above 25% undone the next day",
                   min(reversed_count * REVERSED_SPIKE_PENALTY, REVERSED_SPIKE_CAP)))
    real_moves = len(flagged) - reversed_count
    checks.append(("Large moves (not reversed)", f"{real_moves} move(s) above 25%; may be real events", 0))

    n_stale = stale_days(df["Close"])
    stale_share = n_stale / len(df) if len(df) else 0.0
    checks.append(("Stale prices", f"{n_stale} day(s) in runs of {STALE_RUN_DAYS}+ unchanged closes ({stale_share:.1%})",
                   _tiered(stale_share, STALE_LIMITS)))

    if "Volume" in df and df["Volume"].notna().any():
        zero = (df["Volume"] == 0).to_numpy()
        unchanged = (df["Close"].diff() == 0).to_numpy()
        no_trade, moved = int((zero & unchanged).sum()), int((zero & ~unchanged).sum())
        missing = int(df["Volume"].isna().sum())
        checks.append(("Zero-volume days", f"{no_trade} day(s) with zero volume and an unchanged close ({no_trade / len(df):.1%}): "
                       "exchange holidays filled in by the source, or days without trades; "
                       f"{missing} day(s) without volume data", _tiered(no_trade / len(df), NO_TRADE_LIMITS)))
        checks.append(("Zero volume, price moved", f"{moved} day(s) where the close changed on zero volume",
                       _tiered(moved / len(df), ZERO_VOLUME_MOVED_LIMITS)))
    else:
        checks.append(("Zero-volume days", "volume not available", NO_VOLUME_PENALTY))

    gaps = date_gaps(df["Date"])
    longest = int(gaps["Calendar Days"].max()) if len(gaps) else 0
    checks.append(("Gaps in history", f"{len(gaps)} gap(s) longer than {GAP_CALENDAR_DAYS} calendar days"
                   + (f"; longest {longest} days" if len(gaps) else ""), min(len(gaps) * GAP_PENALTY, GAP_CAP)))

    checks.append(("History length", f"{n_returns} daily returns", _tiered_below(n_returns, SHORT_HISTORY)))

    if fundamentals is not None:
        missing_key = [f for f in KEY_FUNDAMENTALS if f in fundamentals.get("missing_fields", [])]
        penalty = round(MISSING_FUNDAMENTALS_CAP * len(missing_key) / len(KEY_FUNDAMENTALS))
        checks.append(("Missing fundamentals", f"{len(missing_key)} of {len(KEY_FUNDAMENTALS)} key fields missing"
                       + (f": {', '.join(missing_key)}" if missing_key else ""), penalty))
    else:
        checks.append(("Missing fundamentals", "not checked", 0))

    table = pd.DataFrame(checks, columns=["Check", "Finding", "Penalty"])
    score = max(0, 100 - int(table["Penalty"].sum()))
    reasons = [f"{row.Check}: {row.Finding} (−{row.Penalty})" for row in table.itertuples() if row.Penalty > 0]
    return {"score": score, "checks": table, "reasons": reasons}


def quality_table(frames: dict, fundamentals: dict = None) -> pd.DataFrame:
    """One summary row per holding: Ticker, Score, and the main reasons."""
    rows = []
    for ticker, df in frames.items():
        result = assess_holding(df, (fundamentals or {}).get(ticker))
        rows.append({"Ticker": ticker, "Score": result["score"],
                     "Reasons": "; ".join(result["reasons"]) or "no issues found"})
    return pd.DataFrame(rows, columns=["Ticker", "Score", "Reasons"])
