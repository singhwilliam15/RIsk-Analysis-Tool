"""
Cached wrappers, so Streamlit reruns do not repeat downloads or heavy calculations.

Streamlit reruns the whole script on every widget change. These functions are keyed on their
inputs (return series and parameters), so moving an unrelated slider reuses earlier results.
"""

import pandas as pd
import streamlit as st

import data_fetcher
import fundamentals
import trust
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


@st.cache_data(show_spinner="Bootstrapping 90% ranges for every model…", max_entries=32)
def cached_trust_ranges(returns: pd.Series, confidence_level: float, num_simulations: int) -> dict:
    # 1-day ranges as decimal losses: independent of the position size and horizon, so those reuse them
    return trust.bootstrap_ranges(returns, confidence_level, cached_fit_models(returns), num_simulations)


@st.cache_data(show_spinner="Recomputing ES on each lookback…", max_entries=32)
def cached_lookback(history: pd.Series, investment: float, confidence_level: float, holding_period: int) -> pd.DataFrame:
    return trust.lookback_sensitivity(history, investment, confidence_level, holding_period)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_ghost(window: pd.Series, history: pd.Series, confidence_level: float, investment: float) -> dict:
    return trust.ghost_effect(window, history, confidence_level, investment=investment)


@st.cache_data(show_spinner="Measuring liquidity…", max_entries=32)
def cached_liquidity(positions: pd.DataFrame, frames: dict, long_frames: dict, scenarios: pd.DataFrame, official: dict,
                     participation: float, amfi_participation: float, amfi_exclude: float, bangia_k: float,
                     impact_y: float, user_spread_pct: float, freeze_override: int) -> dict:
    # Imported here: ui.liquidity_layer imports this module
    from ui.liquidity_layer import analyse
    return analyse(positions, frames, long_frames, scenarios, official, participation, amfi_participation,
                   amfi_exclude, bangia_k, impact_y, user_spread_pct, freeze_override)


@st.cache_data(show_spinner="Measuring credit risk…", max_entries=64)
def cached_credit_holding(ticker: str, prices: pd.DataFrame, fund: dict, vols: dict, sector, industry,
                          trading_currency: str, ratings: pd.DataFrame, r: float, ltd_weight: float, T: float,
                          config: dict, as_of) -> dict:
    # Imported here: ui.credit_layer imports this module
    from ui.credit_layer import analyse_holding
    return analyse_holding(ticker, prices, fund, vols, sector, industry, trading_currency, ratings, r, ltd_weight, T,
                           config, as_of)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_scenarios(position_history: pd.Series, market_prices: pd.Series, scenarios: pd.DataFrame,
                     proxy_beta: float, investment: float) -> pd.DataFrame:
    return run_scenarios(position_history, market_prices, scenarios, proxy_beta, investment)
