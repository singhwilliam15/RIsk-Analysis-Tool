"""Data-quality checks on synthetic price histories with known defects."""

import numpy as np
import pandas as pd
import pytest

from data_quality import assess_holding, date_gaps, quality_table, stale_days


def clean_history(n=600, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({"Date": pd.bdate_range("2023-01-02", periods=n), "Close": close,
                       "Volume": rng.integers(10_000, 50_000, n).astype(float)})
    df["Returns"] = df["Close"].pct_change()
    return df


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    df["Returns"] = df["Close"].pct_change()
    return df


def penalty(result: dict, check: str) -> int:
    return int(result["checks"].set_index("Check").loc[check, "Penalty"])


def test_clean_history_scores_100():
    result = assess_holding(clean_history())
    assert result["score"] == 100 and result["reasons"] == []


def test_reversing_spikes_cost_15_each_up_to_30():
    df = clean_history()
    df.loc[100, "Close"] *= 3  # +200%, undone the next day
    result = assess_holding(recompute(df))
    assert penalty(result, "Reversing spikes") == 15 and result["score"] == 85
    for day in (200, 300):
        df.loc[day, "Close"] *= 3
    assert penalty(assess_holding(recompute(df)), "Reversing spikes") == 30


def test_a_real_crash_is_reported_but_not_penalised():
    df = clean_history()
    df.loc[300:, "Close"] *= 0.6  # a lasting −40% move
    result = assess_holding(recompute(df))
    assert penalty(result, "Reversing spikes") == 0 and penalty(result, "Large moves (not reversed)") == 0
    assert result["checks"].set_index("Check").loc["Large moves (not reversed)", "Finding"].startswith("1 move")


def test_stale_day_count_by_hand():
    assert stale_days(pd.Series([1, 1, 1, 2, 3, 3, 4, 4, 4, 4.0])) == 7  # runs of 3 and 4; the pair of 3s is too short
    assert stale_days(pd.Series([1, 2, 3.0])) == 0
    assert stale_days(pd.Series([], dtype=float)) == 0


def test_stale_prices_are_penalised_by_share_of_days():
    df = clean_history()
    df.loc[100:111, "Close"] = df.loc[100, "Close"]  # 12 of 600 days = 2%
    assert penalty(assess_holding(recompute(df)), "Stale prices") == 10
    df.loc[200:239, "Close"] = df.loc[200, "Close"]  # 52 of 600 days > 5%
    assert penalty(assess_holding(recompute(df)), "Stale prices") == 20


def no_trade_days(df: pd.DataFrame, days) -> pd.DataFrame:
    """Rows like Yahoo's holiday fillers: the previous close repeated, with zero volume."""
    for day in days:
        df.loc[day, "Close"] = df.loc[day - 1, "Close"]
        df.loc[day, "Volume"] = 0
    return recompute(df)


@pytest.mark.parametrize("count, expected", [(6, 0), (15, 10), (36, 20)])  # 1%, 2.5%, 6% of 600 days
def test_zero_volume_days_with_an_unchanged_close(count, expected):
    df = no_trade_days(clean_history(), range(10, 10 + 10 * count, 10))
    result = assess_holding(df)
    assert penalty(result, "Zero-volume days") == expected
    assert penalty(result, "Zero volume, price moved") == 0


@pytest.mark.parametrize("count, expected", [(3, 5), (10, 10)])  # 0.5% and 1.7% of 600 days
def test_zero_volume_while_the_price_moved(count, expected):
    df = clean_history()
    df.loc[:count - 1, "Volume"] = 0
    result = assess_holding(df)
    assert penalty(result, "Zero volume, price moved") == expected and penalty(result, "Zero-volume days") == 0


def test_volume_not_available():
    assert penalty(assess_holding(clean_history().drop(columns="Volume")), "Zero-volume days") == 5
    assert penalty(assess_holding(clean_history().assign(Volume=np.nan)), "Zero-volume days") == 5


def test_gaps_longer_than_a_week():
    df = clean_history()
    expected = (df.loc[310, "Date"] - df.loc[299, "Date"]).days  # 11 business days apart: Fri 23 Feb to Mon 11 Mar = 17
    df = df.drop(index=range(300, 310)).reset_index(drop=True)
    gaps = date_gaps(df["Date"])
    assert expected == 17
    assert len(gaps) == 1 and gaps["Calendar Days"].iloc[0] == expected
    assert penalty(assess_holding(recompute(df)), "Gaps in history") == 5
    # An ordinary weekend or a 4-day holiday is not a gap
    assert date_gaps(pd.Series(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-09"]))).empty


@pytest.mark.parametrize("n, expected", [(200, 25), (251, 10), (500, 10), (501, 0)])
def test_short_history(n, expected):
    # n prices give n − 1 returns: 500 returns are one 250-day window plus the 250 test days a 99% backtest needs
    assert penalty(assess_holding(clean_history(n)), "History length") == expected


def test_missing_fundamentals():
    result = assess_holding(clean_history(), {"missing_fields": ["revenue", "ebit", "inventory"]})
    # 2 of the 10 key fields (inventory is not a key field): round(15 × 2/10) = 3
    assert penalty(result, "Missing fundamentals") == 3
    assert penalty(assess_holding(clean_history(), {"missing_fields": []}), "Missing fundamentals") == 0
    assert assess_holding(clean_history())["checks"].set_index("Check").loc["Missing fundamentals", "Finding"] == "not checked"


def test_score_never_below_zero_and_data_unchanged():
    df = clean_history(150).drop(columns="Volume")
    for day in (20, 40, 60):
        df.loc[day, "Close"] *= 3
    df.loc[80:120, "Close"] = df.loc[80, "Close"]
    df = df.drop(index=[*range(125, 132), *range(135, 142), *range(143, 149)]).reset_index(drop=True)  # 3 gaps
    recompute(df)
    before = df.copy()
    result = assess_holding(df, {"missing_fields": ["revenue", "ebit", "total_assets", "total_liabilities", "current_assets",
                                                    "current_liabilities", "retained_earnings", "total_equity",
                                                    "operating_cash_flow", "shares_outstanding"]})
    assert result["checks"]["Penalty"].sum() > 100 and result["score"] == 0
    pd.testing.assert_frame_equal(df, before)


def test_quality_table_has_one_row_per_holding():
    table = quality_table({"A": clean_history(), "B": clean_history(200)})
    assert table["Ticker"].tolist() == ["A", "B"] and table["Score"].tolist() == [100, 75]
    assert table.loc[0, "Reasons"] == "no issues found" and "History length" in table.loc[1, "Reasons"]
