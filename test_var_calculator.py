import numpy as np
import pandas as pd
import pytest

from var_calculator import (
    backtest_all_methods,
    binomial_cdf,
    calculate_all_var,
    calculate_cornish_fisher_var,
    calculate_ewma_var,
    calculate_historical_var,
    calculate_monte_carlo_var,
    calculate_parametric_var,
    calculate_student_t_var,
    christoffersen_test,
    estimate_beta,
    ewma_volatility,
    historical_worst_losses,
    perform_kupiec_backtest,
    rolling_historical_var,
    rolling_var_forecasts,
    run_stress_testing,
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


@pytest.fixture
def fat_tailed_returns():
    rng = np.random.default_rng(11)
    return pd.Series(rng.standard_t(4, 3000) * 0.01)


def test_student_t_es_matches_simulation(fat_tailed_returns):
    from scipy.stats import t as student_t
    res = calculate_student_t_var(fat_tailed_returns, 1, 0.99)
    nu, loc, scale = res["degrees_of_freedom"], res["loc"], res["scale"]
    draws = loc + scale * student_t.rvs(nu, size=2_000_000, random_state=3)
    cutoff = np.quantile(draws, 0.01)
    assert res["var_daily_pct"] == pytest.approx(-cutoff, rel=0.01)
    assert res["cvar_daily_pct"] == pytest.approx(-draws[draws <= cutoff].mean(), rel=0.01)


def test_student_t_exceeds_normal_on_fat_tails(fat_tailed_returns):
    t_var = calculate_student_t_var(fat_tailed_returns, 1, 0.99)["var_daily_pct"]
    normal_var = calculate_parametric_var(fat_tailed_returns, 1, 0.99)["var_daily_pct"]
    assert t_var > normal_var


def test_cornish_fisher_tracks_fat_tails(fat_tailed_returns):
    cf = calculate_cornish_fisher_var(fat_tailed_returns, 1, 0.99)
    normal = calculate_parametric_var(fat_tailed_returns, 1, 0.99)
    assert cf["var_daily_pct"] > normal["var_daily_pct"]
    assert cf["cvar_daily_pct"] > cf["var_daily_pct"]


def test_cornish_fisher_equals_normal_without_skew_or_kurtosis(normal_returns, monkeypatch):
    monkeypatch.setattr(pd.Series, "skew", lambda self: 0.0)
    monkeypatch.setattr(pd.Series, "kurtosis", lambda self: 0.0)
    cf = calculate_cornish_fisher_var(normal_returns, 1, 0.95)["var_daily_pct"]
    normal = calculate_parametric_var(normal_returns, 1, 0.95)["var_daily_pct"]
    assert cf == pytest.approx(normal)


def test_ewma_recursion_by_hand():
    returns = pd.Series([0.01, -0.02, 0.03])
    sigma, sigma_next = ewma_volatility(returns, lam=0.94)
    v0 = np.var([0.01, -0.02, 0.03], ddof=1)
    v1 = 0.94 * v0 + 0.06 * 0.01 ** 2
    v2 = 0.94 * v1 + 0.06 * 0.02 ** 2
    v3 = 0.94 * v2 + 0.06 * 0.03 ** 2
    assert sigma.tolist() == pytest.approx(np.sqrt([v0, v1, v2]).tolist())
    assert sigma_next == pytest.approx(np.sqrt(v3))
    assert calculate_ewma_var(returns, 1, 0.99)["var_daily_pct"] == pytest.approx(2.326348 * np.sqrt(v3), rel=1e-5)


def test_ewma_reacts_to_recent_volatility():
    calm, stressed = np.full(500, 0.005), np.full(20, -0.04)
    returns = pd.Series(np.concatenate([calm * np.resize([1, -1], 500), stressed]))
    ewma = calculate_ewma_var(returns, 1, 0.99)["var_daily_pct"]
    normal = calculate_parametric_var(returns, 1, 0.99)["var_daily_pct"]
    assert ewma > normal


def test_all_models_return_positive_var_below_es(fat_tailed_returns):
    for name, res in calculate_all_var(fat_tailed_returns, 1_000_000, 0.99, num_simulations=20_000).items():
        assert 0 < res["var_daily_pct"] < res["cvar_daily_pct"], name


def test_rolling_forecasts_never_use_future_data(fat_tailed_returns):
    fast = ["Historical", "Parametric (Normal)", "Cornish-Fisher", "EWMA (RiskMetrics)", "FHS (EWMA-filtered)"]
    base = rolling_var_forecasts(fat_tailed_returns, 0.99, window=250, models=fast)
    shocked_returns = fat_tailed_returns.copy()
    shocked_returns.iloc[600] = -0.5
    shocked = rolling_var_forecasts(shocked_returns, 0.99, window=250, models=fast)
    pd.testing.assert_frame_equal(base.iloc[:601], shocked.iloc[:601])
    assert not np.allclose(base.iloc[601].to_numpy(), shocked.iloc[601].to_numpy())


def test_christoffersen_detects_clustered_breaches():
    clustered = np.zeros(500, dtype=int)
    clustered[100:110] = 1
    spread = np.zeros(500, dtype=int)
    spread[::50] = 1
    assert christoffersen_test(clustered)["p_value_ind"] < 0.01
    assert christoffersen_test(spread)["p_value_ind"] > 0.05


def test_christoffersen_known_value():
    hits = [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    res = christoffersen_test(hits)
    assert (res["n00"], res["n01"], res["n10"], res["n11"]) == (4, 2, 2, 1)
    pi, pi01, pi11 = 3 / 9, 2 / 6, 1 / 3
    restricted = 6 * np.log(1 - pi) + 3 * np.log(pi)
    markov = 4 * np.log(1 - pi01) + 2 * np.log(pi01) + 2 * np.log(1 - pi11) + 1 * np.log(pi11)
    assert res["lr_ind"] == pytest.approx(max(-2 * (restricted - markov), 0.0), abs=1e-12)


def test_backtest_table_covers_every_model(fat_tailed_returns):
    forecasts = rolling_var_forecasts(fat_tailed_returns, 0.99, window=250)
    table = backtest_all_methods(fat_tailed_returns, forecasts, 0.99)
    assert list(table["Method"]) == list(forecasts.columns)
    assert (table["Test Days"] == len(fat_tailed_returns) - 250).all()
    normal = table.set_index("Method").loc["Parametric (Normal)", "Actual Breaches"]
    student = table.set_index("Method").loc["Student-t", "Actual Breaches"]
    assert student < normal


def test_beta_recovers_known_value():
    rng = np.random.default_rng(5)
    index = pd.Series(rng.normal(0, 0.01, 1000))
    stock = 1.5 * index + pd.Series(rng.normal(0, 0.002, 1000))
    assert estimate_beta(stock, index) == pytest.approx(1.5, abs=0.02)


def test_stress_scales_by_beta_and_caps_at_total_loss():
    df = run_stress_testing(1_000_000, beta=2.0).set_index("Scenario")
    assert df.loc["COVID-19 Crash (Mar 2020)", "Stock_Shock"] == pytest.approx(-0.68)
    assert df.loc["Lehman / GFC Crash (2008)", "Stock_Shock"] == -1.0
    assert df["Post_Shock_Value"].min() >= 0


def test_worst_losses_compound_returns():
    returns = pd.Series([0.0, -0.1, -0.1, 0.05])
    table = historical_worst_losses(returns, 100, horizons=(1, 2)).set_index("Horizon")
    assert table.loc["1 day", "Worst Return"] == pytest.approx(-0.1)
    assert table.loc["2 days", "Worst Return"] == pytest.approx(0.9 * 0.9 - 1)


def test_excel_report_contains_every_model_and_sheet(fat_tailed_returns):
    import io
    import openpyxl
    from excel_exporter import generate_excel_var_report

    dates = pd.bdate_range("2020-01-01", periods=len(fat_tailed_returns))
    prices = 100 * (1 + fat_tailed_returns).cumprod()
    df = pd.DataFrame({"Date": dates, "Close": prices, "Returns": prices.pct_change(),
                       "Log_Returns": np.log(prices).diff(), "Rolling_30d_Vol": prices.pct_change().rolling(30).std()})
    returns = df["Returns"].dropna()
    by_level = {cl: calculate_all_var(returns, 1_000_000, cl, 1) for cl in (0.90, 0.95, 0.99)}
    models = list(by_level[0.99])
    xlsx = generate_excel_var_report(
        "TEST", "Test Co", "USD", 1_000_000, 0.99, 1, df,
        var_by_level={m: {cl: by_level[cl][m] for cl in by_level} for m in models},
        backtest_table=backtest_all_methods(returns, rolling_var_forecasts(returns, 0.99, 250, models=["Historical", "EWMA (RiskMetrics)"]), 0.99),
        backtest_window=250, stress_df=run_stress_testing(1_000_000, 1.2),
        worst_df=historical_worst_losses(returns, 1_000_000), benchmark_name="S&P 500", beta=1.2)

    wb = openpyxl.load_workbook(io.BytesIO(xlsx))
    assert wb.sheetnames == ["Dashboard", "Backtesting", "Stress Testing", "Raw Data"]
    dash = wb["Dashboard"]
    rows = {dash.cell(row=r, column=2).value: [dash.cell(row=r, column=c).value for c in range(3, 7)] for r in range(15, 15 + len(models))}
    assert list(rows) == models
    for m in models:
        assert rows[m][0] < rows[m][1] < rows[m][2]  # 90% < 95% < 99% VaR
        assert rows[m][3] == pytest.approx(by_level[0.99][m]["cvar_scaled_amount"])
    assert wb["Raw Data"].max_row == 4 + len(df)
    back = wb["Backtesting"]
    headers = [back.cell(row=6, column=c).value for c in range(2, 15)]
    assert "Tick Loss" in headers and headers.index("Verdict") == headers.index("Actual Breaches") + 1
    assert back["B4"].value.startswith("Not enough")  # no recommendation passed in


# ---------------------------------------------------------------
# Phase 1: multi-day scaling, performance statistics, tick loss,
# backtest power and Monte Carlo hygiene
# ---------------------------------------------------------------

def test_parametric_multi_day_uses_sqrt_t_for_sigma_and_t_for_mean(normal_returns):
    mu, sigma, z, t = normal_returns.mean(), normal_returns.std(ddof=1), 2.326348, 10
    res = calculate_parametric_var(normal_returns, 1_000_000, 0.99, holding_period=t)
    assert res["var_scaled_pct"] == pytest.approx(z * sigma * np.sqrt(t) - mu * t, rel=1e-5)
    es_1d_factor = 0.026652 / 0.01  # φ(z)/α at 99%
    assert res["cvar_scaled_pct"] == pytest.approx(es_1d_factor * sigma * np.sqrt(t) - mu * t, rel=1e-4)
    assert res["var_scaled_amount"] == pytest.approx(res["var_scaled_pct"] * 1_000_000)


def test_every_parametric_model_scales_the_mean_linearly(fat_tailed_returns):
    from var_calculator import SCALING_GARCH, SCALING_PARAMETRIC, SCALING_SIMULATED, SCALING_SQRT_TIME
    t = 10
    for name, res in calculate_all_var(fat_tailed_returns, 1, 0.99, holding_period=t).items():
        if res["scaling_rule"] == SCALING_PARAMETRIC:
            mu = res["mu"]
            assert res["var_scaled_pct"] == pytest.approx((res["var_daily_pct"] + mu) * np.sqrt(t) - mu * t), name
        else:
            assert res["scaling_rule"] in (SCALING_SQRT_TIME, SCALING_GARCH, SCALING_SIMULATED), name
            if res["scaling_rule"] == SCALING_SQRT_TIME:
                assert res["var_scaled_pct"] == pytest.approx(res["var_daily_pct"] * np.sqrt(t)), name


def test_overlapping_t_day_var_by_hand():
    from var_calculator import calculate_overlapping_historical_var
    returns = pd.Series([0.10, -0.20, 0.05, -0.10, 0.00])
    res = calculate_overlapping_historical_var(returns, 100, 0.5, holding_period=2)
    windows = [1.1 * 0.8 - 1, 0.8 * 1.05 - 1, 1.05 * 0.9 - 1, 0.9 * 1.0 - 1]  # -0.12, -0.16, -0.055, -0.10
    assert res["observations"] == 4
    assert res["var_pct"] == pytest.approx(-np.percentile(windows, 50))
    assert res["cvar_amount"] == pytest.approx(100 * (0.16 + 0.12) / 2)


def test_historical_reports_overlapping_check_only_for_multi_day(normal_returns):
    assert "overlapping_var_amount" not in calculate_historical_var(normal_returns, 1, 0.95, 1)
    assert calculate_historical_var(normal_returns, 1, 0.95, 10)["overlapping_observations"] == len(normal_returns) - 9


def test_cagr_comes_from_the_compounded_path():
    from var_calculator import calculate_portfolio_statistics
    returns = pd.Series([0.05, -0.05] * 126)  # 252 days of +5% / -5%
    stats = calculate_portfolio_statistics(returns)
    assert stats["cagr"] == pytest.approx((1.05 * 0.95) ** 126 - 1)
    assert stats["ann_mean_return"] == pytest.approx(0.0)
    assert stats["cagr"] < stats["ann_mean_return"]


def test_sortino_uses_downside_deviation_over_all_days():
    from var_calculator import calculate_portfolio_statistics
    returns = pd.Series([0.02, -0.01, 0.0, -0.03])
    stats = calculate_portfolio_statistics(returns, risk_free_rate=0.0)
    downside = np.sqrt((0.01 ** 2 + 0.03 ** 2) / 4)
    assert stats["downside_deviation"] == pytest.approx(downside)
    assert stats["sortino_ratio"] == pytest.approx(returns.mean() / downside * np.sqrt(252))
    assert "coef_variation" in stats and "cov" not in stats


def test_risk_free_rate_input_changes_sharpe(normal_returns):
    from var_calculator import calculate_portfolio_statistics
    low = calculate_portfolio_statistics(normal_returns, risk_free_rate=0.04)
    high = calculate_portfolio_statistics(normal_returns, risk_free_rate=0.065)
    assert high["sharpe_ratio"] < low["sharpe_ratio"]
    assert high["risk_free_rate"] == 0.065


def test_tick_loss_by_hand():
    from var_calculator import tick_loss
    returns = pd.Series([-0.03, 0.01, -0.01])
    var = pd.Series([0.02, 0.02, 0.02])
    # (0.05 - 1)(-0.03 + 0.02) + 0.05(0.01 + 0.02) + 0.05(-0.01 + 0.02), averaged
    assert tick_loss(returns, var, 0.95) == pytest.approx((0.0095 + 0.0015 + 0.0005) / 3)


def test_tick_loss_is_lowest_at_the_true_quantile():
    from var_calculator import tick_loss
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, 200_000))
    true_var = 2.326348 * 0.01
    losses = {k: tick_loss(returns, pd.Series(np.full(len(returns), true_var * k)), 0.99) for k in (0.8, 1.0, 1.2)}
    assert losses[1.0] < losses[0.8] and losses[1.0] < losses[1.2]


def test_recommended_model_is_lowest_loss_among_passing():
    from var_calculator import recommend_model
    table = pd.DataFrame({
        "Method": ["A", "B", "C"],
        "Verdict": ["FAIL", "PASS", "PASS"],
        "Tick Loss": [0.001, 0.003, 0.002],
    })
    assert recommend_model(table) == {"model": "C", "status": "recommended"}
    table["Verdict"] = "FAIL"
    assert recommend_model(table) == {"model": "A", "status": "none_pass"}
    table["Verdict"] = "LOW POWER"
    assert recommend_model(table) == {"model": None, "status": "low_power"}


@pytest.mark.parametrize("confidence, days, expect_low_power", [(0.99, 249, True), (0.99, 250, False), (0.95, 99, True), (0.95, 100, False)])
def test_backtest_flags_low_power(confidence, days, expect_low_power):
    rng = np.random.default_rng(8)
    returns = pd.Series(rng.normal(0, 0.01, 250 + days))
    table = backtest_all_methods(returns, rolling_var_forecasts(returns, confidence, window=250, models=["Historical", "EWMA (RiskMetrics)"]), confidence)
    assert (table["Test Days"] == days).all()
    assert ((table["Verdict"] == "LOW POWER").all()) == expect_low_power


def test_monte_carlo_is_reproducible_without_touching_global_seed(normal_returns):
    state = np.random.get_state()[1].copy()
    a = calculate_monte_carlo_var(normal_returns, 1, 0.99, seed=7)
    b = calculate_monte_carlo_var(normal_returns, 1, 0.99, seed=7)
    assert a["var_daily_pct"] == b["var_daily_pct"]
    assert (np.random.get_state()[1] == state).all()


def test_monte_carlo_flags_too_few_tail_draws(normal_returns):
    few = calculate_monte_carlo_var(normal_returns, 1, 0.99, num_simulations=1000, seed=1)
    enough = calculate_monte_carlo_var(normal_returns, 1, 0.99, num_simulations=10_000, seed=1)
    assert few["tail_draws"] == 10 and few["few_tail_draws"]
    assert enough["tail_draws"] == 100 and not enough["few_tail_draws"]


def test_kupiec_exact_calibration_passes():
    # 5 breaches in 100 days at 95% is exactly the expected rate; LR must be 0, not NaN
    returns = pd.Series([-0.1] * 5 + [0.0] * 95)
    res = perform_kupiec_backtest(returns, 0.05, 0.95)
    assert res["lr_stat"] == pytest.approx(0.0, abs=1e-12)
    assert res["p_value"] == pytest.approx(1.0)
    assert res["test_result"].startswith("PASS")


def test_recommendation_can_exclude_a_model():
    from var_calculator import recommend_model
    table = pd.DataFrame({"Method": ["A", "MC"], "Verdict": ["PASS", "PASS"], "Tick Loss": [0.002, 0.001]})
    assert recommend_model(table)["model"] == "MC"
    assert recommend_model(table, exclude=["MC"]) == {"model": "A", "status": "recommended"}
