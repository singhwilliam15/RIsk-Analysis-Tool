"""
Data Fetcher Module for Yahoo Finance Ticker Integration
Fetches daily historical price data, company info, and return series.
"""

import urllib.request
import json
import pandas as pd
import numpy as np
from datetime import datetime

def format_ticker_symbol(ticker: str) -> str:
    """Format ticker symbol to standard Yahoo Finance convention."""
    ticker = ticker.strip().upper()
    return ticker

def fetch_stock_data_direct(ticker: str, period: str = "2y") -> dict:
    """
    Fetch historical daily market data from Yahoo Finance API directly.
    Provides robust, zero-dependency HTTP fallback.
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
            quote = result['indicators']['quote'][0]
            
            close_prices = quote.get('close', [])
            dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
            
            df = pd.DataFrame({
                "Date": pd.to_datetime(dates),
                "Close": close_prices
            })
            
            # Clean missing data
            df = df.dropna().reset_index(drop=True)
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Rolling_30d_Vol'] = df['Returns'].rolling(30).std() * np.sqrt(252)
            
            currency = meta.get('currency', 'INR' if symbol.endswith('.NS') or symbol.endswith('.BO') else 'USD')
            company_name = meta.get('shortName', meta.get('longName', symbol))
            current_price = meta.get('regularMarketPrice', df['Close'].iloc[-1] if len(df) > 0 else 0.0)
            
            return {
                "success": True,
                "symbol": symbol,
                "company_name": company_name,
                "currency": currency,
                "current_price": current_price,
                "df": df,
                "error": None
            }
    except Exception as e:
        return {
            "success": False,
            "symbol": symbol,
            "df": pd.DataFrame(),
            "error": f"Failed to fetch data for ticker '{symbol}': {str(e)}"
        }

def fetch_stock_data(ticker: str, period: str = "2y") -> dict:
    """Primary fetcher using yfinance library with direct HTTP fallback."""
    symbol = format_ticker_symbol(ticker)
    
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty or len(df) < 10:
            # Fallback to direct HTTP
            return fetch_stock_data_direct(symbol, period)
            
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        df = df[['Date', 'Close']].dropna().reset_index(drop=True)
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Rolling_30d_Vol'] = df['Returns'].rolling(30).std() * np.sqrt(252)
        
        # Metadata
        info = stock.info if hasattr(stock, 'info') else {}
        company_name = info.get('longName', info.get('shortName', symbol))
        currency = info.get('currency', 'INR' if symbol.endswith('.NS') or symbol.endswith('.BO') else 'USD')
        current_price = info.get('regularMarketPrice', df['Close'].iloc[-1] if len(df) > 0 else 0.0)
        
        return {
            "success": True,
            "symbol": symbol,
            "company_name": company_name,
            "currency": currency,
            "current_price": current_price,
            "df": df,
            "error": None
        }
    except Exception:
        return fetch_stock_data_direct(symbol, period)
