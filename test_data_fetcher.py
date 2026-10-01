import io
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import data_fetcher
from data_fetcher import SOURCE_DIRECT, SOURCE_YFINANCE, fetch_stock_data, fetch_stock_data_direct


def _history(n=30, tz="Asia/Kolkata"):
    dates = pd.date_range("2024-01-01 09:15", periods=n, freq="B", tz=tz, name="Date")
    return pd.DataFrame({"Close": [100.0 + i for i in range(n)]}, index=dates)


def _fake_yfinance(history=None, info=None, history_error=None, info_error=None):
    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, auto_adjust=True):
            assert auto_adjust, "prices must be split/dividend adjusted"
            if history_error:
                raise history_error
            return history

        @property
        def info(self):
            if info_error:
                raise info_error
            return info

    return types.SimpleNamespace(Ticker=Ticker)


def _chart_json(timestamps, close, adjclose=None, meta=None):
    indicators = {"quote": [{"close": close}]}
    if adjclose is not None:
        indicators["adjclose"] = [{"adjclose": adjclose}]
    base_meta = {"currency": "INR", "shortName": "Test Ltd", "regularMarketPrice": 123.0}
    base_meta.update(meta or {})
    return {"chart": {"result": [{"meta": base_meta, "timestamp": timestamps, "indicators": indicators}]}}


@pytest.fixture
def mock_http(monkeypatch):
    """Replace the network call with a canned JSON response and record requested URLs."""
    calls = {"urls": [], "payload": None}

    def fake_urlopen(req, timeout=None):
        calls["urls"].append(req.full_url)
        return io.BytesIO(json.dumps(calls["payload"]).encode("utf-8"))

    monkeypatch.setattr(data_fetcher.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_yfinance_path_uses_adjusted_prices_and_metadata(monkeypatch):
    fake = _fake_yfinance(history=_history(), info={"longName": "Reliance Industries", "currency": "INR", "regularMarketPrice": 2900.0})
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    res = fetch_stock_data("reliance.ns")
    assert res["success"] and res["symbol"] == "RELIANCE.NS"
    assert res["data_source"] == SOURCE_YFINANCE and res["price_basis"] == "adjusted"
    assert res["company_name"] == "Reliance Industries" and res["current_price"] == 2900.0
    assert res["df"]["Date"].iloc[0] == pd.Timestamp("2024-01-01")
    assert res["df"]["Returns"].iloc[1] == pytest.approx(0.01)


def test_metadata_failure_keeps_yfinance_prices(monkeypatch, mock_http):
    fake = _fake_yfinance(history=_history(), info_error=RuntimeError("429 Too Many Requests"))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    res = fetch_stock_data("TCS.NS")
    assert mock_http["urls"] == [], "a metadata failure must not trigger a second download"
    assert res["data_source"] == SOURCE_YFINANCE and res["metadata_available"] is False
    assert res["company_name"] == "TCS.NS"
    assert res["currency"] == "INR"
    assert res["current_price"] == pytest.approx(129.0)


def test_metadata_failure_on_us_ticker_defaults_to_usd(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(history=_history(tz="America/New_York"), info_error=RuntimeError()))
    assert fetch_stock_data("AAPL")["currency"] == "USD"


def test_yfinance_price_failure_falls_back_to_direct(monkeypatch, mock_http):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(history_error=RuntimeError("blocked")))
    ts = [1704166200 + 86400 * i for i in range(3)]
    mock_http["payload"] = _chart_json(ts, [100, 101, 102], adjclose=[90, 91, 92], meta={"exchangeTimezoneName": "Asia/Kolkata"})
    res = fetch_stock_data("INFY.NS")
    assert res["data_source"] == SOURCE_DIRECT
    assert "INFY.NS" in mock_http["urls"][0]


def test_direct_prefers_adjusted_close(mock_http):
    ts = [1704166200 + 86400 * i for i in range(3)]
    mock_http["payload"] = _chart_json(ts, close=[100, 110, 121], adjclose=[50, 55, 60.5])
    res = fetch_stock_data_direct("X.NS")
    assert res["price_basis"] == "adjusted"
    assert res["df"]["Close"].tolist() == [50, 55, 60.5]


def test_direct_falls_back_to_unadjusted_close(mock_http):
    ts = [1704166200 + 86400 * i for i in range(3)]
    mock_http["payload"] = _chart_json(ts, close=[100, 110, 121])
    res = fetch_stock_data_direct("X.NS")
    assert res["price_basis"] == "unadjusted"
    assert res["df"]["Close"].tolist() == [100, 110, 121]


def test_direct_dates_use_exchange_timezone_not_host(mock_http):
    # 2024-01-01 22:00 UTC is already 2 Jan in India (03:30 IST)
    mock_http["payload"] = _chart_json([1704146400, 1704232800], close=[100, 101], meta={"exchangeTimezoneName": "Asia/Kolkata"})
    assert fetch_stock_data_direct("X.NS")["df"]["Date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_direct_dates_fall_back_to_gmtoffset(mock_http):
    # Same instant as above; without a timezone name, the +05:30 offset still gives 2 Jan
    mock_http["payload"] = _chart_json([1704146400, 1704232800], close=[100, 101], meta={"gmtoffset": 19800})
    assert fetch_stock_data_direct("X.NS")["df"]["Date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_direct_failure_returns_error(monkeypatch):
    def broken(req, timeout=None):
        raise OSError("network down")
    monkeypatch.setattr(data_fetcher.urllib.request, "urlopen", broken)
    res = fetch_stock_data_direct("X.NS")
    assert not res["success"] and "network down" in res["error"]


def test_suspicious_returns_flags_large_moves_without_changing_data():
    from data_fetcher import suspicious_returns
    df = pd.DataFrame({"Date": pd.bdate_range("2005-07-26", periods=5),
                       "Close": [42.3, 42.8, 187.1, 43.0, 43.3]})
    df["Returns"] = df["Close"].pct_change()
    flagged = suspicious_returns(df)
    assert list(flagged.index) == [2, 3]
    assert flagged.loc[2, "Returns"] == pytest.approx(187.1 / 42.8 - 1)
    assert flagged.loc[2, "Next_Day_Return"] == pytest.approx(43.0 / 187.1 - 1)
    assert flagged.loc[2, "Reversed"]  # +337% then -77%: back near the start, a data error
    assert df["Close"].tolist() == [42.3, 42.8, 187.1, 43.0, 43.3]


def test_suspicious_returns_keeps_real_crashes_unreversed():
    from data_fetcher import suspicious_returns
    df = pd.DataFrame({"Date": pd.bdate_range("2000-09-27", periods=4), "Close": [100.0, 100.0, 48.0, 45.0]})
    df["Returns"] = df["Close"].pct_change()
    flagged = suspicious_returns(df)
    assert list(flagged.index) == [2] and not flagged.loc[2, "Reversed"]


# ---------------------------------------------------------------
# Open/high/low/volume and NSE + BSE volume
# ---------------------------------------------------------------

def test_yfinance_path_keeps_ohlcv(monkeypatch):
    volume = [1000.0, 0.0, 3000.0, np.nan] + [5000.0] * 11
    hist = _history(15).assign(Open=99.0, High=110.0, Low=95.0, Volume=volume)
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(history=hist))
    df = fetch_stock_data("X.NS")["df"]
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)
    assert len(df) == 15  # a day without volume is kept, not dropped
    assert df["Volume"].iloc[:3].tolist() == [1000.0, 0.0, 3000.0] and np.isnan(df["Volume"].iloc[3])
    assert df["High"].iloc[0] == 110.0 and df["Low"].iloc[0] == 95.0


def test_yfinance_path_without_volume_columns(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(history=_history()))
    df = fetch_stock_data("X.NS")["df"]
    assert len(df) == 30 and df["Volume"].isna().all()


def test_direct_scales_open_high_low_by_the_adjustment_factor(mock_http):
    ts = [1704166200 + 86400 * i for i in range(3)]
    payload = _chart_json(ts, close=[100, 110, 120], adjclose=[50, 55, 60])
    payload["chart"]["result"][0]["indicators"]["quote"][0].update(
        open=[98, 108, 118], high=[102, 112, 122], low=[96, 106, None], volume=[500, None, 700])
    mock_http["payload"] = payload
    df = fetch_stock_data_direct("X.NS")["df"]
    assert df["Open"].tolist() == [49, 54, 59] and df["High"].tolist() == [51, 56, 61]
    assert np.isnan(df["Low"].iloc[2]) and df["Low"].iloc[0] == 48
    assert df["Volume"].iloc[0] == 500 and np.isnan(df["Volume"].iloc[1])  # volume is not price-adjusted
    assert len(df) == 3


def _listing(symbol, dates, volume):
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Close": 100.0, "Volume": volume})
    return {"success": True, "symbol": symbol, "df": df, "volume_sources": [symbol]}


def test_other_listing():
    assert data_fetcher.other_listing("RELIANCE.NS") == "RELIANCE.BO"
    assert data_fetcher.other_listing("500325.BO") == "500325.NS"
    assert data_fetcher.other_listing("AAPL") is None


def test_nse_and_bse_volume_are_summed_where_both_exist():
    nse = _listing("X.NS", ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], [1000.0, 2000.0, np.nan, 4000.0])
    bse = _listing("X.BO", ["2026-01-05", "2026-01-07", "2026-01-08", "2026-01-09"], [100.0, 300.0, np.nan, 900.0])
    combined = data_fetcher.combine_exchange_volume(nse, bse)
    df = combined["df"]
    # 5 Jan: both → 1100. 6 Jan: BSE missing → NSE only. 7 Jan: NSE missing → stays missing (prices are NSE's).
    # 8 Jan: BSE has no volume → NSE only. 9 Jan: BSE-only date → ignored.
    assert df["Volume"].tolist()[:2] == [1100.0, 2000.0] and np.isnan(df["Volume"].iloc[2]) and df["Volume"].iloc[3] == 4000.0
    assert len(df) == 4
    assert df["Volume_NS"].tolist()[:2] == [1000.0, 2000.0] and df["Volume_BO"].iloc[0] == 100.0
    assert combined["volume_sources"] == ["X.NS", "X.BO"]
    assert nse["df"]["Volume"].tolist()[0] == 1000.0  # the input is not modified


def test_volume_is_unchanged_when_the_other_listing_fails_or_never_overlaps():
    nse = _listing("X.NS", ["2026-01-05"], [1000.0])
    assert data_fetcher.combine_exchange_volume(nse, {"success": False}) is nse
    assert data_fetcher.combine_exchange_volume(nse, _listing("X.BO", ["2025-01-01"], [5.0])) is nse
