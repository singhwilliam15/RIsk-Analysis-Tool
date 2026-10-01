"""Phase 2 model tests: GARCH(1,1)-t, FHS, GARCH Monte Carlo, Student-t MLE, Cornish-Fisher validity, ES backtest."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t as student_t

import var_calculator
from garch import (
    GarchParams,
    fhs_var_es,
    fit_garch_t,
    garch_filter,
    garch_variance_term_structure,
    simulate_garch_paths,
    standardized_t_es,
    standardized_t_quantile,
)
from var_calculator import (
    MONTE_CARLO_BACKTEST_NAME,
    backtest_all_methods,
    calculate_fhs_var,
    calculate_garch_t_var,
    calculate_historical_var,
    calculate_monte_carlo_var,
    calculate_student_t_var,
    cornish_fisher_is_valid,
    cornish_fisher_quantile,
    ewma_volatility,
    fit_student_t,
    mcneil_frey_test,
    rolling_forecasts,
)

TRUE = GarchParams(mu=0.05, omega=0.05, alpha=0.08, beta=0.90, nu=6.0)


def simulate_garch_series(params: GarchParams, n: int, seed: int) -> pd.Series:
    """Independent GARCH(1,1)-t simulator used as the reference data generator."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(params.nu, n) * np.sqrt((params.nu - 2) / params.nu)
    var = params.omega / (1 - params.alpha - params.beta)
    out = np.empty(n)
    for i in range(n):
        eps = np.sqrt(var) * z[i]
        out[i] = (params.mu + eps) / 100
        var = params.omega + params.alpha * eps ** 2 + params.beta * var
    return pd.Series(out)


@pytest.fixture(scope="module")
def garch_returns():
    return simulate_garch_series(TRUE, 4000, seed=12)


# ---------- unit-variance Student-t helpers ----------

@pytest.mark.parametrize("nu", [3.5, 6.0, 30.0])
def test_standardized_t_quantile_and_es_match_simulation(nu):
    draws = student_t.rvs(nu, size=3_000_000, random_state=1) * np.sqrt((nu - 2) / nu)
    assert draws.std() == pytest.approx(1.0, rel=0.02)
    q = np.quantile(draws, 0.01)
    assert standardized_t_quantile(0.01, nu) == pytest.approx(q, rel=0.01)
    assert standardized_t_es(0.01, nu) == pytest.approx(-draws[draws <= q].mean(), rel=0.02)


# ---------- GARCH(1,1)-t ----------

def test_garch_fit_recovers_true_parameters(garch_returns):
    fit = fit_garch_t(garch_returns)
    assert fit["converged"]
    p = fit["params"]
    assert p.alpha == pytest.approx(TRUE.alpha, abs=0.03)
    assert p.beta == pytest.approx(TRUE.beta, abs=0.04)
    assert p.persistence == pytest.approx(TRUE.persistence, abs=0.02)
    assert p.nu == pytest.approx(TRUE.nu, abs=2.0)
    assert p.mu == pytest.approx(TRUE.mu, abs=0.04)


def test_garch_filter_recursion_by_hand():
    p = GarchParams(mu=0.1, omega=0.2, alpha=0.1, beta=0.8, nu=5)
    r = np.array([0.01, -0.02, 0.005])
    sigma = garch_filter(r, p, initial_variance=1.5)
    v, expected = 1.5, [1.5]
    for x in r:
        e = x * 100 - 0.1
        v = 0.2 + 0.1 * e * e + 0.8 * v
        expected.append(v)
    assert sigma == pytest.approx(np.sqrt(expected))


def test_variance_term_structure_reverts_to_long_run():
    p = GarchParams(mu=0, omega=0.05, alpha=0.08, beta=0.9, nu=6)
    v = garch_variance_term_structure(p, next_variance=9.0, horizon=2000)
    assert v[0] == 9.0
    assert v[1] == pytest.approx(0.05 + 0.98 * 9.0)
    assert v[-1] == pytest.approx(0.05 / 0.02, rel=1e-3)


def test_garch_var_and_es_from_formula(garch_returns):
    fit = fit_garch_t(garch_returns)
    p = fit["params"]
    res = calculate_garch_t_var(garch_returns, 1_000_000, 0.99, fit=fit)
    s = garch_filter(garch_returns, p)[-1]
    assert res["var_daily_pct"] == pytest.approx(-(p.mu + s * standardized_t_quantile(0.01, p.nu)) / 100)
    assert 0 < res["var_daily_pct"] < res["cvar_daily_pct"]
    assert res["var_scaled_pct"] == pytest.approx(res["var_daily_pct"])  # 1-day horizon
    assert not res["fallback"]


def test_garch_multi_day_uses_variance_term_structure(garch_returns):
    fit = fit_garch_t(garch_returns)
    p = fit["params"]
    res = calculate_garch_t_var(garch_returns, 1, 0.99, holding_period=10, fit=fit)
    s = garch_filter(garch_returns, p)[-1]
    total_sd = np.sqrt(garch_variance_term_structure(p, s ** 2, 10).sum())
    assert res["var_scaled_pct"] == pytest.approx(-(10 * p.mu + total_sd * standardized_t_quantile(0.01, p.nu)) / 100)


def test_garch_falls_back_to_ewma_when_fit_fails(garch_returns):
    failed = {"params": None, "converged": False, "error": "did not converge"}
    res = calculate_garch_t_var(garch_returns, 1, 0.99, fit=failed)
    ewma = var_calculator.calculate_ewma_var(garch_returns, 1, 0.99)
    assert res["fallback"] and res["var_daily_pct"] == pytest.approx(ewma["var_daily_pct"])


def test_rolling_garch_never_uses_future_data(garch_returns):
    r = garch_returns.iloc[:700]
    models = ["GARCH(1,1)-t", MONTE_CARLO_BACKTEST_NAME]
    base = rolling_forecasts(r, 0.99, window=250, models=models)
    shocked = r.copy()
    shocked.iloc[500] = -0.3
    after = rolling_forecasts(shocked, 0.99, window=250, models=models)
    for key in ("var", "es", "sigma"):
        pd.testing.assert_frame_equal(base[key].iloc[:501], after[key].iloc[:501])
    assert not np.allclose(base["var"].iloc[501], after["var"].iloc[501])
    assert base["garch_refits"] == int(np.ceil((700 - 250) / 20))


def test_rolling_garch_counts_fallbacks(monkeypatch, garch_returns):
    monkeypatch.setattr(var_calculator, "fit_garch_t", lambda r: {"params": None, "converged": False, "error": "x"})
    out = rolling_forecasts(garch_returns.iloc[:400], 0.99, window=250, models=["GARCH(1,1)-t", "EWMA (RiskMetrics)"])
    assert out["garch_fallbacks"] == out["garch_refits"] == 8
    pd.testing.assert_series_equal(out["var"]["GARCH(1,1)-t"], out["var"]["EWMA (RiskMetrics)"], check_names=False)


def test_garch_backtest_coverage_on_garch_data(garch_returns):
    fc = rolling_forecasts(garch_returns, 0.99, window=250, models=["GARCH(1,1)-t", "Parametric (Normal)"])
    table = backtest_all_methods(garch_returns, fc["var"], 0.99).set_index("Method")
    # The correctly specified model should be close to 1% breaches; the static normal model should not beat it
    assert table.loc["GARCH(1,1)-t", "Breach Rate"] == pytest.approx(0.01, abs=0.004)
    assert table.loc["GARCH(1,1)-t", "Tick Loss"] < table.loc["Parametric (Normal)", "Tick Loss"]


# ---------- Monte Carlo from GARCH-t ----------

def test_simulated_paths_follow_the_garch_recursion():
    p = GarchParams(mu=0.02, omega=0.1, alpha=0.1, beta=0.85, nu=5)
    paths = simulate_garch_paths(p, next_variance=2.0, horizon=3, num_simulations=4, seed=9)
    z = np.random.default_rng(9).standard_t(5, size=(4, 3)) * np.sqrt(3 / 5)
    var = np.full(4, 2.0)
    for h in range(3):
        eps = np.sqrt(var) * z[:, h]
        assert paths[:, h] == pytest.approx((0.02 + eps) / 100)
        var = 0.1 + 0.1 * eps ** 2 + 0.85 * var


def test_monte_carlo_matches_garch_and_simulates_paths(garch_returns):
    from var_calculator import SCALING_SIMULATED
    fit = fit_garch_t(garch_returns)
    mc = calculate_monte_carlo_var(garch_returns, 1, 0.99, holding_period=10, num_simulations=200_000, seed=3, fit=fit)
    analytic = calculate_garch_t_var(garch_returns, 1, 0.99, fit=fit)
    assert mc["process"] == "GARCH(1,1)-t" and mc["scaling_rule"] == SCALING_SIMULATED
    assert mc["var_daily_pct"] == pytest.approx(analytic["var_daily_pct"], rel=0.02)
    assert mc["cvar_daily_pct"] == pytest.approx(analytic["cvar_daily_pct"], rel=0.03)
    assert mc["var_daily_pct"] < mc["var_scaled_pct"] < mc["var_daily_pct"] * 10


def test_monte_carlo_falls_back_to_normal(garch_returns):
    failed = {"params": None, "converged": False, "error": "x"}
    mc = calculate_monte_carlo_var(garch_returns, 1, 0.99, num_simulations=200_000, seed=1, fit=failed)
    mu, sd = garch_returns.mean(), garch_returns.std()
    assert mc["fallback"] and mc["var_daily_pct"] == pytest.approx(2.326348 * sd - mu, rel=0.02)


# ---------- Filtered Historical Simulation ----------

def test_fhs_var_es_by_hand():
    var, es = fhs_var_es([-2.0, -1.0, 0.0, 1.0, 2.0], sigma_next=0.01, alpha=0.2)
    assert var == pytest.approx(0.012)  # 20th percentile of z is -1.2 (linear interpolation)
    assert es == pytest.approx(0.02)    # only z = -2 lies at or below it


def test_fhs_rescales_standardized_residuals(garch_returns):
    sigma, sigma_next = ewma_volatility(garch_returns)
    res = calculate_fhs_var(garch_returns, 1, 0.99)
    z = garch_returns / sigma
    assert res["var_daily_pct"] == pytest.approx(-np.percentile(z, 1) * sigma_next)


def test_fhs_reacts_to_current_volatility():
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 800)
    stressed = rng.normal(0, 0.03, 40)
    r = pd.Series(np.concatenate([calm, stressed]))
    assert calculate_fhs_var(r, 1, 0.99)["var_daily_pct"] > calculate_historical_var(r, 1, 0.99)["var_daily_pct"]


# ---------- Student-t maximum likelihood ----------

def test_student_t_mle_recovers_parameters():
    r = pd.Series(student_t.rvs(5, loc=0.001, scale=0.01, size=40_000, random_state=4))
    fit = fit_student_t(r)
    assert fit["method"] == "maximum likelihood"
    assert fit["nu"] == pytest.approx(5, abs=0.5)
    assert fit["scale"] == pytest.approx(0.01, rel=0.03)
    assert fit["loc"] == pytest.approx(0.001, abs=2e-4)
    res = calculate_student_t_var(r, 1, 0.99, fit=fit)
    assert res["var_daily_pct"] == pytest.approx(-student_t.ppf(0.01, fit["nu"], fit["loc"], fit["scale"]))


def test_student_t_falls_back_to_moments(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("optimiser failed")
    monkeypatch.setattr(student_t, "fit", broken)
    fit = fit_student_t(pd.Series(np.random.default_rng(0).standard_t(5, 2000) * 0.01))
    assert fit["method"].startswith("method of moments")


# ---------- Cornish-Fisher validity ----------

def _numerically_monotone(skew, kurt):
    z = np.linspace(-8, 8, 4001)
    return bool(np.all(np.diff(cornish_fisher_quantile(z, skew, kurt)) >= -1e-12))


@pytest.mark.parametrize("skew, kurt", [(0, 0), (0, 3), (0, 6), (0, 10), (-0.5, 3), (-1, 2), (1.2, 4), (-0.3, 12), (0.2, -0.5)])
def test_cornish_fisher_validity_matches_monotonicity(skew, kurt):
    assert cornish_fisher_is_valid(skew, kurt) == _numerically_monotone(skew, kurt)


def test_cornish_fisher_validity_known_cases():
    assert cornish_fisher_is_valid(0, 3)
    assert not cornish_fisher_is_valid(0, 10)


# ---------- McNeil-Frey Expected Shortfall test ----------

def _normal_backtest(n=6000, es_factor=1.0, seed=5):
    rng = np.random.default_rng(seed)
    sigma = 0.01 * np.exp(rng.normal(0, 0.3, n))  # time-varying but known volatility
    r = sigma * rng.standard_normal(n)
    z = norm.ppf(0.975)
    var = z * sigma
    es = norm.pdf(z) / 0.025 * sigma * es_factor
    return r, var, es, sigma


def test_mcneil_frey_passes_correct_model():
    res = mcneil_frey_test(*_normal_backtest())
    assert res["breaches"] > 100
    assert res["p_value"] > 0.05
    assert res["mean_residual"] == pytest.approx(0.0, abs=0.1)


def test_mcneil_frey_rejects_es_understated_by_30_percent():
    res = mcneil_frey_test(*_normal_backtest(es_factor=0.7))
    assert res["p_value"] < 0.01 and res["mean_residual"] > 0


def test_mcneil_frey_needs_enough_breaches():
    r, var, es, sigma = _normal_backtest(n=100)
    assert np.isnan(mcneil_frey_test(r, var * 3, es, sigma)["p_value"])


def test_backtest_table_includes_es_test(garch_returns):
    fc = rolling_forecasts(garch_returns.iloc[:1500], 0.975, window=250, models=["GARCH(1,1)-t", "Historical"])
    table = backtest_all_methods(garch_returns.iloc[:1500], fc["var"], 0.975, fc["es"], fc["sigma"])
    assert {"ES p-value", "ES Test"} <= set(table.columns)
    assert set(table["ES Test"]) <= {"PASS", "FAIL", "TOO FEW BREACHES", "LOW POWER"}
