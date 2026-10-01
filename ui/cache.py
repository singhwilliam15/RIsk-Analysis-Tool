"""
Cached wrappers, so Streamlit reruns do not repeat downloads or heavy calculations.

Streamlit reruns the whole script on every widget change. These functions are keyed on their
inputs (return series and parameters), so moving an unrelated slider reuses earlier results.
"""

import pandas as pd
import streamlit as st

import data_fetcher
import fundamentals
from stress import run_scenarios
from var_calculator import backtest_all_methods, calculate_all_var, fit_models, rolling_forecasts, rolling_model_fits


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker: str, period: str = "2y") -> dict:
    # Look the fetcher up at call time, so tests can replace data_fetcher.fetch_stock_data
    return data_fetcher.fetch_stock_data(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(ticker: str, period: str = "2y") -> dict:
    """Prices for one listing; for an Indian stock, volume summed over its NSE and BSE listings."""
    primary = cached_fetch(ticker, period)
    other_symbol = data_fetcher.other_listing(primary.get("symbol", ticker)) if primary["success"] else None
    if other_symbol is None:
        return primary
    return data_fetcher.combine_exchange_volume(primary, cached_fetch(other_symbol, period))


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def cached_fundamentals(ticker: str) -> dict:
    # Looked up at call time, so tests can replace fundamentals.fetch_fundamentals
    return fundamentals.fetch_fundamentals(ticker)


@st.cache_data(show_spinner="Fitting GARCH(1,1)-t and Student-t…", max_entries=32)
def cached_fit_models(returns: pd.Series) -> dict:
    return fit_models(returns)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_all_var(returns: pd.Series, investment: float, confidence_level: float, holding_period: int,
                   num_simulations: int) -> dict:
    return calculate_all_var(returns, investment, confidence_level, holding_period, num_simulations=num_simulations,
                             seed=42, fits=cached_fit_models(returns))


@st.cache_data(show_spinner="Refitting Student-t and GARCH for the backtest…", max_entries=16)
def cached_rolling_fits(returns: pd.Series, window: int) -> dict:
    # Independent of the confidence level, so changing it reuses these fits
    return rolling_model_fits(returns, window)


@st.cache_data(show_spinner="Backtesting every model out of sample…", max_entries=32)
def cached_rolling_forecasts(returns: pd.Series, confidence_level: float, window: int, num_simulations: int) -> dict:
    return rolling_forecasts(returns, confidence_level, window=window, num_simulations=num_simulations,
                             fits=cached_rolling_fits(returns, window))


@st.cache_data(show_spinner=False, max_entries=32)
def cached_backtest(returns: pd.Series, var: pd.DataFrame, confidence_level: float, es: pd.DataFrame,
                    sigma: pd.DataFrame) -> pd.DataFrame:
    return backtest_all_methods(returns, var, confidence_level, es, sigma)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_scenarios(position_history: pd.Series, market_prices: pd.Series, scenarios: pd.DataFrame,
                     proxy_beta: float, investment: float) -> pd.DataFrame:
    return run_scenarios(position_history, market_prices, scenarios, proxy_beta, investment)
