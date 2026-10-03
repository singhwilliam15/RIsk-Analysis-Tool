"""
The wider test (universe.py): bhavcopy parsing in both NSE formats, corporate-action adjustment checked on
Reliance's real 1:1 bonuses, symbol chaining, the shared price flags, the metrics by hand and the jump base rates
on a synthetic panel with a known jump rate.
"""

import io

import numpy as np
import pandas as pd
import pytest

import universe as U
from case_studies import engine as K
from conftest import fake_fetch_stock_data
from events import load_rules

# Real rows (NSE bhavcopies of 6 and 7 Sep 2017, old format; 25 and 28 Oct 2024, UDiFF), trimmed to two securities
OLD_06SEP2017 = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
RELIANCE,EQ,1626.4,1652.5,1622.8,1645.4,1644,1632.6,11142512,18339508179.8,06-SEP-2017,107540,INE002A01018,
SGBNOV25,GB,2900,2900,2900,2900,2900,2900,10,29000,06-SEP-2017,1,IN0020170091,
GOLDBEES,EQ,2900,2900,2900,2900,2900,2900,10,29000,06-SEP-2017,1,INF204KB17I5,
"""
OLD_07SEP2017 = """SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,
RELIANCE,EQ,823,832.5,815,818.1,816.6,1645.4,7408536,6093238362.2,07-SEP-2017,125644,INE002A01018,
"""
UDIFF_HEADER = ("TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,"
                "OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,"
                "ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4\n")
UDIFF_25OCT2024 = UDIFF_HEADER + ("2024-10-25,2024-10-25,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE INDUSTRIES LTD,"
                                  "2687.00,2688.70,2644.00,2655.70,2656.30,2679.60,,2655.70,,,8000000,21245600000.00,300000,F1,1,,,,,\n")
UDIFF_28OCT2024 = UDIFF_HEADER + ("2024-10-28,2024-10-28,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE INDUSTRIES LTD,"
                                  "1337.00,1353.00,1322.10,1334.35,1335.00,2655.70,,1334.35,,,16000000,21349600000.00,400000,F1,1,,,,,\n")
ACTIONS = [[{"symbol": "RELIANCE", "series": "EQ", "exDate": "07-Sep-2017", "subject": " Bonus 1:1"},
            {"symbol": "RELIANCE", "series": "EQ", "exDate": "13-Jul-2017", "subject": " Dividend -  Rs 11/- Per Share"}],
           [{"symbol": "RELIANCE", "series": "EQ", "exDate": "28-Oct-2024", "subject": " Bonus 1:1"},
            {"symbol": "RELIANCE", "series": "EQ", "exDate": "28-Oct-2024", "subject": " Bonus 1:1 (Purpose Revised)"}]]


def _day(text, date):
    return U.normalise_bhavcopy(pd.read_csv(io.StringIO(text), dtype=str), date)


def test_both_formats_parse_to_the_same_columns():
    old = _day(OLD_06SEP2017, "2017-09-06")
    new = _day(UDIFF_25OCT2024, "2024-10-25")
    assert list(old.columns) == list(new.columns) == U.PANEL_COLUMNS
    assert old["symbol"].tolist() == ["RELIANCE"]  # neither the gold bond (series GB) nor an ETF (ISIN INF…) is a company
    assert old.loc[0, "close"] == 1645.4 and old.loc[0, "value"] == pytest.approx(18339508179.8)
    assert new.loc[0, "close"] == 2655.70 and new.loc[0, "prevclose"] == 2679.60


@pytest.mark.parametrize("subject, factor", [
    (" Bonus 1:1", 0.5),
    (" Bonus 2:5", 5 / 7),
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 0.2),
    ("Face Value Split From Rs 10/- Per Share To Re 1/- Per Share", 0.1),
    (" Bonus 1:5/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share", 5 / 6 * 0.2),
    (" Bonus 1:1/Interim Dividend Rs 2/- Per Share (Purpose Revised)", 0.5),
    ("Face Value Split Rs.10/- To Re.1/- Per Share", 0.1),          # SBI, 2014: no "From"
    ("Sub-Division From Rs 10/- Per Share To Rs 2/- Per Share", 0.2),
    ("Demerger", None),
    (" Dividend -  Rs 11/- Per Share", None),
])
def test_action_factor(subject, factor):
    got = U.action_factor(subject)
    assert got == (pytest.approx(factor) if factor is not None else None)


def test_reliance_bonuses_are_not_crashes_after_adjustment():
    panel = pd.concat([_day(OLD_06SEP2017, "2017-09-06"), _day(OLD_07SEP2017, "2017-09-07"),
                       _day(UDIFF_25OCT2024, "2024-10-25"), _day(UDIFF_28OCT2024, "2024-10-28")], ignore_index=True)
    actions = U.parse_corporate_actions(ACTIONS)
    assert len(actions) == 2  # the dividend drops out; the revised record of the same bonus counts once
    raw = U.adjust(panel, None)
    assert raw["ret"].iloc[1] == pytest.approx(818.1 / 1645.4 - 1)  # unadjusted: a false −50% day
    adj = U.adjust(panel, actions)
    # 818.1 / (1645.4 × 0.5) − 1 = −0.56%; 1334.35 / (2655.70 × 0.5) − 1 = +0.49%
    assert adj["ret"].iloc[1] == pytest.approx(818.1 / (1645.4 * 0.5) - 1)
    assert adj["ret"].iloc[3] == pytest.approx(1334.35 / (2655.70 * 0.5) - 1)
    assert adj["ret"].iloc[[1, 3]].abs().max() < 0.01  # both ex-dates are ordinary days, not crashes
    assert adj.loc[0, "close"] == pytest.approx(1645.4 * 0.25)  # before both bonuses: two halvings


def test_symbol_changes_are_chained():
    panel = pd.DataFrame({"symbol": ["A", "B", "C", "A"], "date": pd.to_datetime(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"])})
    changes = U.parse_symbol_changes(pd.DataFrame([["X Ltd", "A", "B", "01-JUN-2015"], ["X Ltd", "B", "C", "01-JUN-2016"]]))
    out = U.chain_symbols(panel, changes)
    # A (2015) → B → C; B (2016) → C; the later, unrelated "A" of 2018 keeps its symbol
    assert out["symbol"].tolist() == ["C", "C", "C", "A"]


def test_price_flags_match_the_case_study_engine():
    df = fake_fetch_stock_data("FLAGS.NS", "max")["df"]
    market = fake_fetch_stock_data("^NSEI", "max")["df"].set_index("Date")["Close"].pct_change().dropna()
    config, rules = K.load_cases(), load_rules()
    as_of = df["Date"].iloc[-300]
    full = K.evaluate(df, market, as_of, 1e7, config["flag_rules"], rules)
    w = K.window_before(df, as_of)
    pf = K.price_flags(w, market[market.index <= as_of], 1e7, config["flag_rules"], rules)
    for p in ("market", "liquidity", "stress"):
        assert pf["flags"][p] == full["flags"][p]
    assert pf["es_hist"] == pytest.approx(full["es_hist"]) and pf["stress_loss"] == pytest.approx(full["stress_loss"])


def test_baselines_by_hand():
    dates = pd.bdate_range("2020-01-01", periods=700)
    close = np.r_[np.full(500, 100.0), np.linspace(100, 50, 200)]  # flat, then a 50% slide
    df = pd.DataFrame({"Date": dates, "Close": close})
    df["Returns"] = df["Close"].pct_change()
    b = K.baselines(df, dates[-1])
    assert b["flags"]["drawdown_30"]  # 126 days ago: 100 − 50 × 73/199 = 81.66, so 50 / 81.66 − 1 = −38.8%
    assert b["flags"]["below_200dma"]
    assert b["available"]["vol_top_decile"]  # 640 rolling values ≥ 500
    early = K.baselines(df, dates[150])
    assert not early["available"]["below_200dma"] and not early["flags"]["below_200dma"]


def test_rule_metrics_by_hand():
    results = pd.DataFrame({"worst_12m": [-0.6, -0.1, -0.7, -0.2, np.nan],
                            "event_observed": [True, False, True, False, False],
                            "event_upper": [True, False, True, False, True]})
    for rule in U.UNIVERSE_RULES:
        results[rule] = [True, True, False, False, True]
    m = U.rule_metrics(results).set_index("Rule").loc["tool"]
    # observed: the NaN-outcome row is dropped; flagged rows 1-2, events rows 1 and 3
    assert m["Dates"] == 4 and m["Precision"] == pytest.approx(0.5) and m["Recall"] == pytest.approx(0.5)
    assert m["Base rate"] == pytest.approx(0.5) and m["Lift"] == pytest.approx(1.0)
    u = U.rule_metrics(results, "event_upper").set_index("Rule").loc["tool"]
    assert u["Dates"] == 5 and u["Precision"] == pytest.approx(2 / 3) and u["Recall"] == pytest.approx(2 / 3)


def test_jump_base_rates_recover_a_known_rate():
    """A synthetic stock with one-day falls of 25% on 1% of days and none in the market: the forward rate ≈ 1%."""
    rng = np.random.default_rng(3)
    n = 2200
    dates = pd.bdate_range("2015-01-01", periods=n)
    jumps = rng.random(n) < 0.01
    returns = rng.normal(0, 0.01, n) + np.where(jumps, -0.25, 0.0)
    close = 100 * np.cumprod(1 + returns)
    rows = pd.DataFrame({"date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": 1e6, "value": 1e9, "series": "EQ"})
    frame = U.stock_frame(rows)
    market = pd.Series(rng.normal(0, 0.005, n), index=dates)
    as_of = U.month_ends(market.index, "2017-01-01", "2022-12-31")
    config, rules = K.load_cases(), load_rules()
    results = pd.DataFrame(U.evaluate_symbol("JMP", frame, market, as_of, config["flag_rules"], rules, dates[-1]))
    table = U.jump_base_rates(results, n_boot=50).set_index(["Group", "Fall"])
    row = table.loc[("All stocks", "≥ 20%")]
    fwd_j, fwd_d = results["fwd_jumps_20"].sum(), results["fwd_days"].sum()
    assert row["Forward rate a day"] == pytest.approx(fwd_j / fwd_d)
    assert row["Forward rate a day"] == pytest.approx(0.01, abs=0.004)
    assert row["Own trailing rate a day"] == pytest.approx(0.01, abs=0.004)
    assert abs(row["Excess p a day"]) < 0.004  # no excess: the stock's own history already has these jumps
    assert row["Mean fall J"] == pytest.approx(-0.25, abs=0.02)
