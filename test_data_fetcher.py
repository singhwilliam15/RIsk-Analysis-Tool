import io
import json
import sys
import types

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
