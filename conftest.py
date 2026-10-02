"""
Shared test setup.

- Every test runs with network access blocked, so the suite proves it needs no internet (CI runs offline).
- `fake_fetch_stock_data` generates deterministic synthetic price histories with the same shape as
  data_fetcher.fetch_stock_data, so the full Streamlit app can be exercised without Yahoo Finance.
"""

import socket
import zlib

import numpy as np
import pandas as pd
import pytest

from data_fetcher import SOURCE_YFINANCE, _add_return_columns

PERIOD_DAYS = {"1y": 250, "2y": 500, "5y": 1250, "max": 2600}
LAST_DATE = "2026-09-30"


def fake_fetch_stock_data(ticker: str, period: str = "2y") -> dict:
    """
    Synthetic prices: a common fat-tailed market factor plus a ticker-specific part, seeded by the
    ticker name. Every period ends on the same date, so histories line up across tickers and periods.
    Tickers starting with BAD fail, to exercise error handling.
    """
    symbol = ticker.strip().upper()
    if symbol.startswith("BAD"):
        return {"success": False, "symbol": symbol, "df": pd.DataFrame(),
                "error": f"Failed to fetch data for ticker '{symbol}': not found"}
    n = PERIOD_DAYS.get(period, 500)
    full = PERIOD_DAYS["max"]
    seed = zlib.crc32(symbol.encode())
    market = np.random.default_rng(7).standard_t(5, full) * 0.008
    own = np.random.default_rng(seed).standard_t(4, full) * 0.010
    beta = 0.6 + (seed % 100) / 100
    returns = market if symbol.startswith("^") else beta * market + own
    close = 100 * np.cumprod(1 + returns)[-n:]
    # Open/high/low/volume come from a separate generator, so the closes are the same as before they existed
    extra = np.random.default_rng(seed + 1)
    open_ = close * (1 + extra.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(extra.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(extra.normal(0, 0.005, n)))
    volume = np.round(extra.lognormal(13, 0.4, n))
    df = _add_return_columns(pd.DataFrame({"Date": pd.bdate_range(end=LAST_DATE, periods=n), "Open": open_, "High": high,
                                           "Low": low, "Close": close, "Volume": volume}))
    indian = symbol.endswith((".NS", ".BO")) or symbol == "^NSEI"
    return {
        "success": True, "symbol": symbol, "company_name": f"{symbol} Test Co",
        "currency": "INR" if indian else "USD", "current_price": float(close[-1]), "df": df,
        "price_basis": "adjusted", "data_source": SOURCE_YFINANCE, "metadata_available": True,
        "sector": "Test Sector", "industry": "Test Industry", "volume_sources": [symbol], "error": None,
    }


def fake_fetch_fundamentals(ticker: str) -> dict:
    """
    Fundamentals stub for app tests: three fiscal years with every field the credit pillar uses, in units that
    fit the synthetic prices (about 100 a share, 1 billion shares). HDFCBANK is labelled a bank, to exercise the
    financial-company path. The figures are made up and only ever used in tests.
    """
    from fundamentals import build_fundamentals
    years = pd.to_datetime(["2024-03-31", "2025-03-31", "2026-03-31"])
    s = 1e9

    def frame(rows):
        return pd.DataFrame({y: [v * (1 + 0.05 * i) for v in rows.values()] for i, y in enumerate(years)}, index=list(rows))

    income = frame({"Total Revenue": 80 * s, "EBITDA": 16 * s, "EBIT": 12 * s, "Interest Expense": 2 * s,
                    "Pretax Income": 10 * s, "Net Income": 7.5 * s})
    balance = frame({"Total Assets": 120 * s, "Total Liabilities Net Minority Interest": 60 * s, "Current Assets": 40 * s,
                     "Current Liabilities": 30 * s, "Inventory": 10 * s, "Cash And Cash Equivalents": 8 * s,
                     "Current Debt": 10 * s, "Long Term Debt": 25 * s, "Total Debt": 35 * s,
                     "Retained Earnings": 45 * s, "Stockholders Equity": 60 * s, "Ordinary Shares Number": s})
    cashflow = frame({"Operating Cash Flow": 13 * s, "Capital Expenditure": -5 * s, "Free Cash Flow": 8 * s})
    bank = ticker.upper().startswith("HDFCBANK")
    info = {"sharesOutstanding": s, "marketCap": 100 * s, "financialCurrency": "INR" if ticker.upper().endswith((".NS", ".BO")) else "USD",
            "sector": "Financial Services" if bank else "Test Sector", "industry": "Banks—Regional" if bank else "Test Industry"}
    return build_fundamentals(ticker.upper(), {"income": income, "balance": balance, "cashflow": cashflow}, info)


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _is_loopback(address) -> bool:
    return isinstance(address, tuple) and str(address[0]) in LOOPBACK


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Refuse every outbound connection. Loopback stays open: asyncio uses it internally on Windows."""
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def guarded_connect(sock, address, *args, **kwargs):
        if not _is_loopback(address):
            raise RuntimeError(f"Network access is disabled in tests (tried {address}); mock the data instead.")
        return real_connect(sock, address, *args, **kwargs)

    def guarded_create(address, *args, **kwargs):
        if not _is_loopback(address):
            raise RuntimeError(f"Network access is disabled in tests (tried {address}); mock the data instead.")
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create)


@pytest.fixture
def fake_market(monkeypatch):
    """Route every price and fundamentals download in the app through the synthetic generators."""
    import data_fetcher
    import fundamentals
    import streamlit as st
    import trust
    monkeypatch.setattr(data_fetcher, "fetch_stock_data", fake_fetch_stock_data)
    monkeypatch.setattr(fundamentals, "fetch_fundamentals", fake_fetch_fundamentals)
    # The app smoke tests check wiring, not the ranges (test_trust.py does), so they use few resamples
    monkeypatch.setattr(trust, "N_BOOT", 60)
    monkeypatch.setattr(trust, "N_BOOT_REFIT", 15)
    monkeypatch.setattr(trust, "N_PARAM_DRAWS", 100)
    import ui.concentration_layer
    monkeypatch.setattr(ui.concentration_layer, "N_BOOT", 30)
    st.cache_data.clear()
    yield fake_fetch_stock_data
    st.cache_data.clear()
