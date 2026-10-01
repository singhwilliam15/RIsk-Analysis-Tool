"""
Data Fetcher Module for Yahoo Finance Ticker Integration
Fetches daily price history and company metadata. Prices are split/dividend-adjusted
closes from both sources, so returns are total returns either way.
"""

import urllib.request
import json
import pandas as pd
import numpy as np

SOURCE_YFINANCE = "yfinance"
SOURCE_DIRECT = "Yahoo chart API (direct HTTP)"


def format_ticker_symbol(ticker: str) -> str:
    """Format ticker symbol to standard Yahoo Finance convention."""
    ticker = ticker.strip().upper()
    return ticker


def currency_from_suffix(symbol: str) -> str:
    """Best guess at the trading currency when metadata is unavailable."""
    return "INR" if symbol.endswith((".NS", ".BO")) else "USD"


def _add_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().reset_index(drop=True)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_30d_Vol'] = df['Returns'].rolling(30).std() * np.sqrt(252)
    return df


def _exchange_dates(timestamps, meta: dict) -> pd.DatetimeIndex:
    """
    Convert Unix timestamps to exchange-local calendar dates, so the date does not
    depend on the timezone of the machine running the app.
    """
    utc = pd.to_datetime(pd.Series(timestamps, dtype="int64"), unit="s", utc=True)
    tz_name = meta.get("exchangeTimezoneName")
    if tz_name:
        try:
            return pd.DatetimeIndex(utc.dt.tz_convert(tz_name).dt.tz_localize(None).dt.normalize())
        except Exception:
            pass
    offset = pd.to_timedelta(meta.get("gmtoffset", 0), unit="s")
    return pd.DatetimeIndex((utc.dt.tz_localize(None) + offset).dt.normalize())


def fetch_stock_data_direct(ticker: str, period: str = "2y") -> dict:
    """
    Fetch historical daily market data from Yahoo Finance's chart API directly.
    Zero-dependency HTTP fallback. Uses adjusted closes when the API provides them.
    """
    symbol = format_ticker_symbol(ticker)

    range_map = {
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "max": "max"
    }
    y_range = range_map.get(period, "2y")

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={y_range}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_json = json.loads(response.read().decode('utf-8'))

        result = res_json['chart']['result'][0]
        meta = result['meta']
        timestamps = result.get('timestamp', [])
        indicators = result['indicators']

        adjusted = (indicators.get('adjclose') or [{}])[0].get('adjclose')
        if adjusted and any(p is not None for p in adjusted):
            close_prices, price_basis = adjusted, "adjusted"
        else:
            close_prices, price_basis = indicators['quote'][0].get('close', []), "unadjusted"

        df = pd.DataFrame({
            "Date": _exchange_dates(timestamps, meta),
            "Close": pd.to_numeric(pd.Series(close_prices, dtype="object"), errors="coerce").to_numpy()
        })
        df = _add_return_columns(df)

        currency = meta.get('currency') or currency_from_suffix(symbol)
        company_name = meta.get('shortName') or meta.get('longName') or symbol
        current_price = meta.get('regularMarketPrice') or (float(df['Close'].iloc[-1]) if len(df) > 0 else 0.0)

        return {
            "success": True,
            "symbol": symbol,
            "company_name": company_name,
            "currency": currency,
            "current_price": current_price,
            "df": df,
            "price_basis": price_basis,
            "data_source": SOURCE_DIRECT,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "symbol": symbol,
            "df": pd.DataFrame(),
            "error": f"Failed to fetch data for ticker '{symbol}': {str(e)}"
        }


SUSPICIOUS_MOVE = 0.25


def suspicious_returns(df: pd.DataFrame, threshold: float = SUSPICIOUS_MOVE) -> pd.DataFrame:
    """
    Daily moves larger than `threshold` in either direction, for the user to check.
    Yahoo's long histories sometimes contain one-day price spikes that reverse the next day
    (e.g. a +337% day followed by −77%). The data are not altered here, only flagged.
    """
    flagged = df.loc[df["Returns"].abs() > threshold, ["Date", "Close", "Returns"]]
    next_day = df["Returns"].shift(-1).loc[flagged.index].to_numpy()
    # "Reversed": the next day undoes at least half of the move, so the price is back near where it started
    two_day = (1 + flagged["Returns"].to_numpy()) * (1 + np.nan_to_num(next_day)) - 1
    reversed_ = np.abs(two_day) < 0.5 * np.abs(flagged["Returns"].to_numpy())
    return flagged.assign(Next_Day_Return=next_day, Reversed=reversed_)


def fetch_stock_data(ticker: str, period: str = "2y") -> dict:
    """
    Primary fetcher using yfinance, with the direct HTTP API as fallback.
    Prices and metadata are fetched separately: a metadata failure (stock.info is slow and
    often rate-limited) keeps the good price history and falls back to simple defaults.
    """
    symbol = format_ticker_symbol(ticker)

    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, auto_adjust=True)
    except Exception:
        return fetch_stock_data_direct(symbol, period)

    if hist is None or hist.empty or len(hist) < 10:
        return fetch_stock_data_direct(symbol, period)

    hist = hist.reset_index()
    df = pd.DataFrame({
        "Date": pd.to_datetime(hist['Date']).dt.tz_localize(None).dt.normalize(),
        "Close": hist['Close'].to_numpy()
    })
    df = _add_return_columns(df)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    return {
        "success": True,
        "symbol": symbol,
        "company_name": info.get('longName') or info.get('shortName') or symbol,
        "currency": info.get('currency') or currency_from_suffix(symbol),
        "current_price": info.get('regularMarketPrice') or float(df['Close'].iloc[-1]),
        "df": df,
        "price_basis": "adjusted",
        "data_source": SOURCE_YFINANCE,
        "metadata_available": bool(info),
        "error": None
    }
