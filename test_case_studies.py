"""Phase 7 engine: no data after the as-of date, realised loss by hand, flag rules, tally and jump frequencies."""

import numpy as np
import pandas as pd
import pytest

from case_studies import engine as K
from data_fetcher import _add_return_columns
from events import load_rules

RULES = load_rules()
FLAGS = {"market_ewma_vol_ratio_min": 1.5, "market_es_95_1d_min": 0.05, "stress_minus20_loss_min": 0.30}


def prices(n=900, seed=0, vol=0.01, start="2016-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0, vol, n))
    df = pd.DataFrame({"Date": dates, "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                       "Volume": rng.integers(100_000, 200_000, n).astype(float)})
    return _add_return_columns(df)


def market_returns(df):
    return pd.Series(df["Returns"].to_numpy(), index=df["Date"]).dropna()


def test_as_of_dates():
    dates = K.as_of_dates("2020-03-05", [12, 6, 3, 1])
    assert [d for _, d in dates] == list(pd.to_datetime(["2019-03-05", "2019-09-05", "2019-12-05", "2020-02-05"]))


def test_window_and_evaluation_use_no_data_after_the_as_of_date():
    df = prices()
    as_of = df["Date"].iloc[700]
    assert K.window_before(df, as_of)["Date"].max() == as_of and len(K.window_before(df, as_of)) == 500
    base = K.evaluate(df, market_returns(prices(seed=9)), as_of, 1e7, FLAGS, RULES)
    future = df.copy()
    future.loc[future.index > 700, "Close"] *= 0.1  # a crash after the as-of date
    future = _add_return_columns(future.drop(columns=["Returns", "Log_Returns", "Rolling_30d_Vol"]))
    after = K.evaluate(future, market_returns(prices(seed=9)), as_of, 1e7, FLAGS, RULES)
    for key in ("es_hist", "ewma_vol_ratio", "stress_loss", "days_to_liquidate", "warning"):
        assert after[key] == pytest.approx(base[key]) if isinstance(base[key], float) else after[key] == base[key], key


def test_realised_loss_by_hand():
    df = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=10), "Close": [100, 90, 95, 80, 85, 60, 70, 75, 50, 90.0]})
    # From the close on 2 Jan (90) to the lowest close up to 2 trading days after the event on 7 Jan (60 on 8 Jan; 50 is later)
    res = K.realised_loss(df, "2020-01-02", "2020-01-07", days_after=2)
    assert res["loss"] == pytest.approx(60 / 90 - 1) and res["trough"] == pd.Timestamp("2020-01-08")
    assert np.isnan(K.realised_loss(df, "2019-12-01", "2020-01-07")["loss"])  # no price on or before the as-of date


def test_too_little_history_is_reported_not_guessed():
    df = prices(n=300)
    res = K.evaluate(df, market_returns(prices(seed=9)), df["Date"].iloc[200], 1e7, FLAGS, RULES)
    assert res["days"] == 201 and "only 201 days" in res["note"] and not any(res["available"].values())


def test_market_flag_fires_on_a_volatility_jump():
    calm = prices(n=800, vol=0.01)
    stormy = calm.copy()
    rng = np.random.default_rng(1)
    shocks = rng.normal(0, 0.04, 40)
    stormy.loc[760:, "Close"] = stormy.loc[759, "Close"] * np.cumprod(1 + shocks)
    stormy = _add_return_columns(stormy.drop(columns=["Returns", "Log_Returns", "Rolling_30d_Vol"]))
    mkt = market_returns(prices(seed=9))
    quiet = K.evaluate(calm, mkt, calm["Date"].iloc[-1], 1e7, FLAGS, RULES)
    loud = K.evaluate(stormy, mkt, stormy["Date"].iloc[-1], 1e7, FLAGS, RULES)
    assert not quiet["flags"]["market"] and loud["flags"]["market"] and loud["ewma_vol_ratio"] >= 1.5


def test_pillars_without_inputs_are_not_available():
    df = prices()
    res = K.evaluate(df, market_returns(prices(seed=9)), df["Date"].iloc[-1], 1e7, FLAGS, RULES, financial=True)
    assert not res["available"]["credit"] and not res["available"]["events"]
    assert res["credit_note"] == "bank / NBFC: not modelled" and "credit" not in res["flags"]


def test_tally_counts_hits_misses_and_false_positives():
    rows = []
    for group, warning, n in (("case", True, 3), ("case", False, 1), ("control", True, 2), ("control", False, 8)):
        for _ in range(n):
            rows.append({"group": group, "days": 500, "warning": warning, **{f"available_{p}": p == "market" for p in K.PILLARS},
                         **{f"flag_{p}": warning and p == "market" for p in K.PILLARS}})
    t = K.tally(pd.DataFrame(rows)).set_index("Pillar")
    assert t.loc["market", "Hits"] == 3 and t.loc["market", "Misses"] == 1 and t.loc["market", "False positives"] == 2
    assert t.loc["market", "False-positive rate"] == pytest.approx(0.2)
    assert t.loc["credit", "Case dates"] == 0 and np.isnan(t.loc["credit", "Hit rate"])
    assert t.loc["any", "Hits"] == 3


def test_jump_frequencies():
    df = prices(n=900)
    df.loc[[705, 710], "Returns"] = [-0.25, -0.12]
    results = pd.DataFrame([{"ticker": "X", "as_of": df["Date"].iloc[700], "days": 500, "warning": True, "group": "case"}])
    out = K.jump_frequencies({"X": df}, results, horizon=63).iloc[0]
    assert out["days"] == 63 and out["falls_20"] == 1 and out["falls_10"] == 2
    assert out["p_fall_10"] == pytest.approx(2 / 63)


def test_case_config_is_consistent():
    config = K.load_cases()
    keys = [c["key"] for c in config["cases"]]
    assert len(keys) == len(set(keys)) and config["as_of_months_before"] == [12, 6, 3, 1]
    for c in config["cases"]:
        assert pd.Timestamp(c["event_date"]) < pd.Timestamp("2026-10-01") and c["event"]
    assert {c["key"] for c in config["cases"] if c["enabled"]} == {"yesbank", "dhfl", "zee", "adani", "jet", "fretail"}
    assert len(config["control_group"]) == 10


def test_committed_results_match_the_documented_tally():
    tally = pd.read_csv(K.ROOT / "results" / "tally.csv").set_index("Pillar")
    assert tally.loc["any", "Hits"] == 15 and tally.loc["any", "Case dates"] == 20
    assert tally.loc["any", "False positives"] == 19 and tally.loc["any", "Control dates"] == 200
    text = (K.ROOT.parent / "docs" / "case_studies.md").read_text(encoding="utf-8")
    assert "**15 (75%)**" in text and "**19 (9.5%)**" in text
