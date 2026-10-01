import numpy as np
import pandas as pd
import pytest

from var_calculator import (
    binomial_cdf,
    calculate_historical_var,
    calculate_monte_carlo_var,
    calculate_parametric_var,
    perform_kupiec_backtest,
    rolling_historical_var,
)


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0005, 0.015, 2000))


def test_parametric_var_matches_closed_form(normal_returns):
    res = calculate_parametric_var(normal_returns, 1_000_000, 0.99)
    mu, sigma = normal_returns.mean(), normal_returns.std(ddof=1)
    assert res["var_daily_pct"] == pytest.approx(2.326348 * sigma - mu, rel=1e-5)


def test_expected_shortfall_exceeds_var(normal_returns):
    for fn in (calculate_historical_var, calculate_parametric_var, calculate_monte_carlo_var):
        res = fn(normal_returns, 1_000_000, 0.95)
        assert res["cvar_daily_pct"] > res["var_daily_pct"] > 0


def test_methods_agree_on_normal_data(normal_returns):
    hist = calculate_historical_var(normal_returns, 1, 0.95)["var_daily_pct"]
    param = calculate_parametric_var(normal_returns, 1, 0.95)["var_daily_pct"]
    mc = calculate_monte_carlo_var(normal_returns, 1, 0.95, num_simulations=50_000, seed=1)["var_daily_pct"]
    assert hist == pytest.approx(param, rel=0.1)
    assert mc == pytest.approx(param, rel=0.05)


def test_holding_period_uses_square_root_of_time(normal_returns):
    one_day = calculate_historical_var(normal_returns, 1_000_000, 0.95, 1)
    ten_day = calculate_historical_var(normal_returns, 1_000_000, 0.95, 10)
    assert ten_day["var_scaled_amount"] == pytest.approx(one_day["var_daily_amount"] * np.sqrt(10))


def test_rolling_var_is_out_of_sample(normal_returns):
    rolling = rolling_historical_var(normal_returns, 0.95, window=250)
    assert rolling.iloc[:250].isna().all()
    expected = -np.quantile(normal_returns.iloc[:250], 0.05)
    assert rolling.iloc[250] == pytest.approx(expected)


def test_kupiec_zero_breaches_is_not_zero():
    returns = pd.Series(np.full(250, 0.001))
    res = perform_kupiec_backtest(returns, 0.05, 0.99)
    assert res["actual_breaches"] == 0
    assert res["lr_stat"] == pytest.approx(-2 * 250 * np.log(0.99))


def test_kupiec_known_value():
    returns = pd.Series([-0.1] * 10 + [0.0] * 240)
    res = perform_kupiec_backtest(returns, 0.05, 0.99)
    assert res["actual_breaches"] == 10
    assert res["lr_stat"] == pytest.approx(12.955, abs=1e-3)
    assert res["test_result"].startswith("FAIL")


@pytest.mark.parametrize("breaches, zone", [(4, "GREEN"), (5, "YELLOW"), (9, "YELLOW"), (10, "RED")])
def test_basel_zones_reproduce_regulatory_table(breaches, zone):
    returns = pd.Series([-0.1] * breaches + [0.0] * (250 - breaches))
    res = perform_kupiec_backtest(returns, 0.05, 0.99)
    assert zone in res["traffic_light"]


def test_binomial_cdf_matches_scipy_for_large_samples():
    binom = pytest.importorskip("scipy.stats").binom
    assert binomial_cdf(60, 5000, 0.01) == pytest.approx(binom.cdf(60, 5000, 0.01), rel=1e-9)
