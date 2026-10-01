"""
Data Fetcher Module for Yahoo Finance Ticker Integration
Fetches daily price history (Open, High, Low, Close, Volume) and company metadata. Prices are
split/dividend-adjusted from both sources, so returns are total returns either way. For Indian
stocks, volume on the other exchange (.NS <-> .BO) can be added with combine_exchange_volume.
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


OHLV_COLUMNS = ("Open", "High", "Low", "Volume")


def _add_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Only a missing close drops a day; a missing volume or high/low stays as NaN
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
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

        quote = indicators['quote'][0]

        def column(values):
            return pd.to_numeric(pd.Series(values if values is not None else [None] * len(timestamps), dtype="object"),
                                 errors="coerce").to_numpy(dtype=float)

        raw_close = column(quote.get('close'))
        adjusted = (indicators.get('adjclose') or [{}])[0].get('adjclose')
        if adjusted and any(p is not None for p in adjusted):
            close, price_basis = column(adjusted), "adjusted"
        else:
            close, price_basis = raw_close, "unadjusted"
        # Open/high/low are scaled by the same adjustment factor as the close, so ranges stay consistent
        factor = np.divide(close, raw_close, out=np.full_like(close, np.nan), where=raw_close > 0)

        df = pd.DataFrame({
            "Date": _exchange_dates(timestamps, meta),
            "Open": column(quote.get('open')) * factor,
            "High": column(quote.get('high')) * factor,
            "Low": column(quote.get('low')) * factor,
            "Close": close,
            "Volume": column(quote.get('volume')),
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
            "sector": None,
            "industry": None,
            "volume_sources": [symbol],
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
        **{col: (pd.to_numeric(hist[col], errors="coerce").to_numpy(dtype=float) if col in hist else np.nan)
           for col in ("Open", "High", "Low")},
        "Close": hist['Close'].to_numpy(),
        "Volume": pd.to_numeric(hist["Volume"], errors="coerce").to_numpy(dtype=float) if "Volume" in hist else np.nan,
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
        "sector": info.get('sector'),
        "industry": info.get('industry'),
        "volume_sources": [symbol],
        "error": None
    }


def other_listing(symbol: str):
    """The same Indian stock on the other exchange (RELIANCE.NS <-> RELIANCE.BO), or None for other markets."""
    if symbol.endswith(".NS"):
        return symbol[:-3] + ".BO"
    if symbol.endswith(".BO"):
        return symbol[:-3] + ".NS"
    return None


def combine_exchange_volume(primary: dict, other: dict) -> dict:
    """
    Add the other Indian exchange's daily volume to the primary listing's volume.

    The two volumes are summed on dates where both exist; where the other listing has no volume
    for a date, the primary volume is kept unchanged. Prices always come from the primary listing.
    The per-exchange volumes are kept as Volume_<SUFFIX> columns, and `volume_sources` lists the
    listings that contributed. If the other listing failed to download, `primary` is returned as is.
    """
    if not primary.get("success") or not other or not other.get("success"):
        return primary
    df = primary["df"].copy()
    own_suffix = primary["symbol"].rsplit(".", 1)[-1]
    other_suffix = other["symbol"].rsplit(".", 1)[-1]
    other_volume = other["df"].set_index(other["df"]["Date"].dt.normalize())["Volume"]
    other_volume = other_volume[~other_volume.index.duplicated(keep="last")]
    own_volume = df["Volume"].to_numpy(dtype=float)
    matched = other_volume.reindex(df["Date"].dt.normalize()).to_numpy(dtype=float)
    df[f"Volume_{own_suffix}"] = own_volume
    df[f"Volume_{other_suffix}"] = matched
    both = ~np.isnan(own_volume) & ~np.isnan(matched)
    df["Volume"] = np.where(both, own_volume + np.nan_to_num(matched), own_volume)
    if not both.any():
        return primary
    return {**primary, "df": df, "volume_sources": [primary["symbol"], other["symbol"]]}
