"""
Concentration and factor pillar: factor-file parsers on excerpts of the real IIMA and Kenneth French files, factor
regressions recovering known loadings, Newey-West against a hand-worked sandwich and a simulation, Euler sums,
HHI, PCA and effective bets, crisis correlation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import concentration as K
import factor_data as F

FIXTURES = Path(__file__).parent / "test_data" / "factors"


# ---------------------------------------------------------------
# Factor files
# ---------------------------------------------------------------

def test_parse_iima_excerpt():
    table = F.parse_iima((FIXTURES / "iima_daily_excerpt.csv").read_text())
    assert table.columns.tolist() == F.FILE_COLUMNS
    assert table["Date"].iloc[0] == pd.Timestamp("1993-10-04")  # the first row has NA market and risk-free values
    assert table["MKT_RF"].iloc[0] == pytest.approx(-0.938504124604478 / 100)
    assert table["MOM"].iloc[0] == pytest.approx(-0.652692555458934 / 100)  # WML is the momentum factor
    assert table["Date"].iloc[-1] == pd.Timestamp("2025-12-31") and len(table) == 7


def test_parse_french_excerpts_and_combine():
    ff3 = F.parse_french((FIXTURES / "french_ff3_daily_excerpt.csv").read_text())
    assert ff3.columns.tolist() == ["Date", "Mkt-RF", "SMB", "HML", "RF"]
    assert ff3["Date"].iloc[0] == pd.Timestamp("1926-07-01") and ff3["Mkt-RF"].iloc[0] == pytest.approx(0.0009)
    assert ff3["Date"].iloc[-1] == pd.Timestamp("2026-08-31")  # the copyright footer is not read as data
    mom = F.parse_french((FIXTURES / "french_mom_daily_excerpt.csv").read_text())
    us = F.combine_us(ff3, mom)
    assert us.columns.tolist() == F.FILE_COLUMNS
    assert set(us["Date"]) == set(ff3["Date"]) & set(mom["Date"])


def test_french_missing_codes_become_missing():
    text = "notes\n\n,Mkt-RF,SMB,HML,RF\n20200102, 1.00, -99.99, 0.50, 0.01\n20200103, -999, 0.10, 0.20, 0.01\n\nCopyright\n"
    table = F.parse_french(text)
    assert np.isnan(table.loc[0, "SMB"]) and np.isnan(table.loc[1, "Mkt-RF"]) and table.loc[0, "Mkt-RF"] == 0.01


def test_write_and_load_round_trip(tmp_path):
    table = F.parse_iima((FIXTURES / "iima_daily_excerpt.csv").read_text())
    F.write_factors("india", table, {"source": "test", "citation": "test"}, directory=tmp_path)
    loaded, meta = F.load_factors("india", directory=tmp_path)
    assert loaded.index[0] == pd.Timestamp("1993-10-04") and meta["end_date"] == "2025-12-31" and meta["rows"] == 7
    assert loaded["SMB"].iloc[0] == pytest.approx(table["SMB"].iloc[0])
    assert F.load_factors("us", directory=tmp_path) == (None, None)


def test_shipped_factor_files():
    india, meta_in = F.load_factors("india")
    us, meta_us = F.load_factors("us")
    assert meta_in["end_date"] == f"{india.index.max():%Y-%m-%d}" and meta_in["survivorship_bias_adjusted"]
    assert us.index.max() >= pd.Timestamp("2026-01-01") and list(us.columns) == F.FILE_COLUMNS[1:]
    assert india.abs().max().max() < 0.25 and us.abs().max().max() < 0.25  # decimals, not percent


def test_region_for():
    assert F.region_for(["RELIANCE.NS", "TCS.BO"]) == "india"
    assert F.region_for(["AAPL", "MSFT"]) == "us"
    assert F.region_for(["AAPL", "TCS.NS"]) is None


# ---------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------

@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(0)
    n = 2500
    dates = pd.bdate_range("2015-01-01", periods=n)
    factors = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), columns=F.FACTOR_COLUMNS, index=dates)
    factors["RF"] = 0.0002
    loadings = {"A": [1.0, 0.5, -0.3, 0.2], "B": [0.8, -0.2, 0.6, 0.0], "C": [1.2, 0.0, 0.0, -0.4]}
    noise = {"A": 0.008, "B": 0.012, "C": 0.005}
    returns = pd.DataFrame({t: factors["RF"] + factors[F.FACTOR_COLUMNS].to_numpy() @ np.array(b) + rng.normal(0, noise[t], n)
                            for t, b in loadings.items()}, index=dates)
    return factors, returns, loadings, noise


def test_known_loadings_and_variance_split_are_recovered(simulated):
    factors, returns, loadings, noise = simulated
    for t, b in loadings.items():
        fit = K.factor_regression(returns[t], factors)
        assert fit["betas"].to_numpy() == pytest.approx(b, abs=0.06)
        assert np.sqrt(fit["resid_var"]) == pytest.approx(noise[t], rel=0.05)
        true_factor_var = np.array(b) @ (np.eye(4) * 0.01 ** 2) @ np.array(b)
        assert fit["factor_var"] == pytest.approx(true_factor_var, rel=0.1)


def test_newey_west_without_lags_is_the_white_sandwich():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.normal(size=200)])
    y = X @ [0.1, 2.0] + rng.normal(size=200) * (1 + np.abs(X[:, 1]))
    fit = K.ols(y, X, lags=0)
    u = fit["resid"]
    inv = np.linalg.inv(X.T @ X)
    white = inv @ (X.T * u ** 2) @ X @ inv
    assert fit["se_nw"] == pytest.approx(np.sqrt(np.diag(white)))


def test_newey_west_tracks_autocorrelated_errors_better_than_ols():
    """With AR(1) errors and an AR(1) regressor, OLS standard errors are too small; Newey-West is close to the truth."""
    rng = np.random.default_rng(2)
    n, reps, rho = 400, 300, 0.7
    slopes, se_ols, se_nw = [], [], []
    for _ in range(reps):
        x, e = np.zeros(n), np.zeros(n)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + rng.normal()
            e[t] = rho * e[t - 1] + rng.normal()
        fit = K.ols(e + 0.5 * x, np.column_stack([np.ones(n), x]))
        slopes.append(fit["params"][1])
        se_ols.append(fit["se"][1])
        se_nw.append(fit["se_nw"][1])
    true_sd = np.std(slopes)
    assert abs(np.mean(se_nw) - true_sd) < abs(np.mean(se_ols) - true_sd)
    assert np.mean(se_ols) < 0.8 * true_sd


def test_newey_west_lag_rule():
    assert K.newey_west_lags(100) == 4 and K.newey_west_lags(500) == 5 and K.newey_west_lags(2500) == 8


def test_too_little_overlap_returns_none_and_single_index_works(simulated):
    factors, returns, _, _ = simulated
    assert K.factor_regression(returns["A"].iloc[:100], factors) is None
    market = factors["MKT_RF"] + factors["RF"]
    fit = K.single_index(returns["A"], market)
    assert fit["model"].startswith("Single") and fit["betas"].index.tolist() == ["MKT"]
    assert fit["betas"]["MKT"] == pytest.approx(1.0, abs=0.1)


def test_rolling_betas_use_only_their_window(simulated):
    factors, returns, _, _ = simulated
    rolling = K.rolling_betas(returns["A"], factors)
    assert rolling.index[0] == returns.index[251] and rolling["MKT_RF"].between(0.8, 1.2).all()
    changed = returns["A"].copy()
    changed.iloc[-1] = 5.0
    assert K.rolling_betas(changed, factors).iloc[:-1].equals(rolling.iloc[:-1])


# ---------------------------------------------------------------
# Portfolio factor risk and concentration
# ---------------------------------------------------------------

def test_euler_parts_sum_to_the_total(simulated):
    factors, returns, _, _ = simulated
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    fits = {t: K.factor_regression(returns[t], factors) for t in w.index}
    res = K.portfolio_factor_risk(w, pd.DataFrame({t: f["betas"] for t, f in fits.items()}).T,
                                  pd.Series({t: f["resid_var"] for t, f in fits.items()}), factors[F.FACTOR_COLUMNS].cov(),
                                  investment=1e6, confidence_level=0.99)
    assert res["parts"].sum() == pytest.approx(res["variance"])
    assert res["var_parts"].sum() == pytest.approx(res["var_total"])
    assert res["var_total"] == pytest.approx(2.3263479 * np.sqrt(res["variance"]) * 1e6, rel=1e-6)
    # The factor model's variance is close to the portfolio's sample variance (residuals are independent here)
    assert res["variance"] == pytest.approx(np.var(returns[w.index].to_numpy() @ w.to_numpy(), ddof=1), rel=0.05)


def test_hhi_by_hand():
    assert K.hhi([0.5, 0.3, 0.2]) == pytest.approx(0.25 + 0.09 + 0.04)
    assert K.hhi([1, 1, 1, 1]) == pytest.approx(0.25)  # weights are normalised: effective number of holdings 4


def test_sector_contributions_sum_to_one(simulated):
    _, returns, _, _ = simulated
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    table = K.sector_view(w, returns.cov(), {"A": "Energy", "B": "Energy", "C": "Banks"})
    assert table["Risk Share"].sum() == pytest.approx(1.0) and table["Weight"].sum() == pytest.approx(1.0)
    assert set(table["Sector"]) == {"Energy", "Banks"}


def test_identical_holdings_are_one_bet():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 500)
    returns = pd.DataFrame({"A": r, "B": r, "C": r})
    res = K.pca_bets(pd.Series([0.2, 0.3, 0.5], index=["A", "B", "C"]), returns.cov())
    assert res["first_share"] == pytest.approx(1.0) and res["enb"] == pytest.approx(1.0)


def test_independent_equal_risk_holdings_are_n_bets():
    cov = pd.DataFrame(np.eye(5) * 0.0004, index=list("ABCDE"), columns=list("ABCDE"))
    res = K.pca_bets(pd.Series(0.2, index=cov.index), cov)
    assert res["enb"] == pytest.approx(5.0)


def test_crisis_correlation_on_a_planted_crisis():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2018-01-01", "2021-12-31")
    n = len(dates)
    market = pd.Series(rng.normal(0, 0.01, n), index=dates)
    crisis = (dates >= "2020-02-01") & (dates <= "2020-04-30")
    common = np.where(crisis, 1.0, 0.0)  # in the crisis both stocks follow the market; otherwise independent
    a = common * market.to_numpy() + (1 - common) * rng.normal(0, 0.01, n)
    b = common * market.to_numpy() + (1 - common) * rng.normal(0, 0.01, n) + rng.normal(0, 0.001, n)
    returns = pd.DataFrame({"A": a, "B": b}, index=dates)
    scenarios = pd.DataFrame({"scenario": ["COVID"], "start": ["2020-02-01"], "end": ["2020-04-30"]})
    res = K.crisis_correlation(returns, market, scenarios, pd.Series([0.5, 0.5], index=["A", "B"]), investment=1e6)
    t = res["table"].set_index("Sample")
    assert t.loc["Crisis windows", "Average Correlation"] > 0.95 > t.loc["Full sample", "Average Correlation"]
    assert t.loc["Full sample", "Benefit Kept"] == pytest.approx(1.0)
    assert t.loc["Crisis windows", "Benefit Kept"] < 0.1  # almost no diversification left
    assert t.loc["Crisis windows", "ES"] > t.loc["Full sample", "ES"]


def test_bootstrap_ranges_contain_the_point(simulated):
    factors, returns, _, _ = simulated
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    boot = K.bootstrap_concentration(returns, w, factors, n_boot=60)
    enb = K.pca_bets(w, returns.cov())["enb"]
    assert boot["enb"][0] <= enb <= boot["enb"][1]
    assert 0 < boot["factor_share"][0] < boot["factor_share"][1] < 1


# ---------------------------------------------------------------
# App layer and Excel
# ---------------------------------------------------------------

def test_analyse_falls_back_to_single_index_for_every_holding(simulated):
    from ui.concentration_layer import analyse
    factors, returns, _, _ = simulated
    series = {t: returns[t] for t in ["A", "B"]}
    series["B"] = series["B"].iloc[-150:]
    short_factors = factors.iloc[:-100]  # factor data end 100 days early, so B overlaps them on only 50 days
    market = factors["MKT_RF"] + factors["RF"]
    res = analyse(series, pd.Series([0.6, 0.4], index=["A", "B"]), short_factors, market, None, None,
                  pd.DataFrame(columns=["scenario", "start", "end"]), {}, 1e6, 0.95, n_boot=20)
    assert res["model"].startswith("Single") and all(f["betas"].index.tolist() == ["MKT"] for f in res["fits"].values())
    assert res["factor_risk"]["parts"].index.tolist() == ["MKT", K.SPECIFIC]


def test_excel_concentration_sheet(simulated):
    import io
    import openpyxl
    from excel_exporter import generate_excel_var_report
    from trust import TrustedMetric
    from ui.concentration_layer import analyse
    from var_calculator import backtest_all_methods, calculate_all_var, historical_worst_losses, rolling_var_forecasts

    factors, returns, _, _ = simulated
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    res = analyse({t: returns[t] for t in w.index}, w, factors, factors["MKT_RF"], returns, factors["MKT_RF"],
                  pd.DataFrame({"scenario": ["X"], "start": ["2016-01-01"], "end": ["2016-06-30"]}), {}, 1e6, 0.95, n_boot=20)
    port = returns.to_numpy() @ w.to_numpy()
    df = pd.DataFrame({"Date": returns.index, "Close": 100 * np.cumprod(1 + port), "Returns": port,
                       "Log_Returns": np.log1p(port), "Rolling_30d_Vol": pd.Series(port).rolling(30).std()})
    r = df["Returns"]
    points = calculate_all_var(r, 1e6, 0.95, 1, num_simulations=1000)
    xlsx = generate_excel_var_report(
        "PORTFOLIO", "Test", "USD", 1e6, 0.95, 1, df, var_by_level={m: {0.95: v} for m, v in points.items()},
        backtest_table=backtest_all_methods(r, rolling_var_forecasts(r, 0.95, 250, models=["Historical"]), 0.95),
        backtest_window=250, stress_table=None, worst_df=historical_worst_losses(r, 1e6), benchmark_name="S&P 500", beta=1.0,
        concentration={"result": res, "metrics": {"enb": TrustedMetric("Effective number of bets", res["pca"]["enb"], 1.1, 1.3, "B")},
                       "factor_meta": {"source": "test", "end_date": "2024-12-31"}})
    sheet = openpyxl.load_workbook(io.BytesIO(xlsx))["Concentration"]
    values = [c.value for row in sheet.iter_rows(min_col=2, max_col=6) for c in row]
    assert "Effective number of bets" in values and "Specific" in values and "Full sample" in values
