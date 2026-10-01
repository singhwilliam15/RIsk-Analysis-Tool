import numpy as np
import pandas as pd
import pytest

from portfolio import (
    align_asset_returns,
    build_portfolio_frame,
    component_var,
    diversification_summary,
    normalize_weights,
)
from var_calculator import calculate_parametric_var


@pytest.fixture
def asset_returns():
    rng = np.random.default_rng(21)
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.3], [0.2, 0.3, 1.0]]) * 0.015 ** 2
    data = rng.multivariate_normal([0.0005, 0.0003, 0.0004], cov, 1500)
    return pd.DataFrame(data, columns=["AAA", "BBB", "CCC"], index=pd.bdate_range("2020-01-01", periods=1500))


@pytest.fixture
def weights():
    return pd.Series([0.5, 0.3, 0.2], index=["AAA", "BBB", "CCC"])


def test_normalize_weights_cleans_input():
    holdings = pd.DataFrame({"Ticker": [" aaa ", "BBB", "", "aaa", None], "Weight": [30, 50, 10, 20, 5]})
    w = normalize_weights(holdings)
    assert list(w.index) == ["AAA", "BBB"]
    assert w["AAA"] == pytest.approx(0.5)
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("holdings, message", [
    (pd.DataFrame({"Ticker": ["AAA", "BBB"], "Weight": [50, -10]}), "positive"),
    (pd.DataFrame({"Ticker": ["AAA", "BBB"], "Weight": [50, None]}), "numeric"),
    (pd.DataFrame({"Ticker": ["AAA"], "Weight": [100]}), "at least two"),
])
def test_normalize_weights_rejects_bad_input(holdings, message):
    with pytest.raises(ValueError, match=message):
        normalize_weights(holdings)


def test_align_uses_only_common_dates():
    a = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]), "Close": [100, 101, 102, 103]})
    b = pd.DataFrame({"Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "Close": [50, 55, 50]})
    r = align_asset_returns({"A": a, "B": b})
    assert list(r.index) == list(pd.to_datetime(["2024-01-03", "2024-01-04"]))
    assert r.loc["2024-01-03", "B"] == pytest.approx(0.1)


def test_portfolio_returns_are_weighted_sum(asset_returns, weights):
    frame = build_portfolio_frame(asset_returns, weights)
    expected = asset_returns @ weights.to_numpy()
    assert frame["Returns"].to_numpy() == pytest.approx(expected.to_numpy())
    assert frame["Close"].iloc[-1] == pytest.approx(100 * (1 + expected).prod())


def test_component_var_adds_up_to_portfolio_var(asset_returns, weights):
    table = component_var(asset_returns, weights, 1_000_000, 0.99, holding_period=10)
    port = build_portfolio_frame(asset_returns, weights)["Returns"]
    total = calculate_parametric_var(port, 1_000_000, 0.99, 10)["var_scaled_amount"]
    assert table["Component VaR"].sum() == pytest.approx(total, rel=1e-9)
    assert table["Contribution %"].sum() == pytest.approx(1.0)


def test_component_var_matches_finite_difference(asset_returns, weights):
    # Marginal VaR is the derivative of portfolio VaR with respect to each weight
    def port_var(w):
        r = asset_returns @ w
        return calculate_parametric_var(r, 1.0, 0.95)["var_daily_pct"]
    table = component_var(asset_returns, weights, 1.0, 0.95)
    for i in range(len(weights)):
        bumped = weights.to_numpy().copy()
        bumped[i] += 1e-6
        numeric = (port_var(bumped) - port_var(weights.to_numpy())) / 1e-6
        assert table["Marginal VaR"].iloc[i] == pytest.approx(numeric, rel=1e-3)


def test_no_diversification_when_assets_move_together(asset_returns):
    r = pd.DataFrame({"A": asset_returns["AAA"], "B": 2 * asset_returns["AAA"]})
    summary = diversification_summary(r, pd.Series([0.5, 0.5], index=["A", "B"]), 1_000_000, 0.95)
    assert summary["diversification_benefit"] == pytest.approx(0.0, abs=1e-6)
    assert summary["average_correlation"] == pytest.approx(1.0)


def test_diversification_benefit_with_imperfect_correlation(asset_returns, weights):
    summary = diversification_summary(asset_returns, weights, 1_000_000, 0.95)
    assert summary["diversification_benefit"] > 0
    assert 0 < summary["diversification_ratio"] < 1
    assert summary["average_correlation"] == pytest.approx((0.6 + 0.2 + 0.3) / 3, abs=0.05)


def test_excel_portfolio_sheet(asset_returns, weights):
    import io
    import openpyxl
    from excel_exporter import generate_excel_var_report
    from var_calculator import backtest_all_methods, calculate_all_var, historical_worst_losses, rolling_var_forecasts, run_stress_testing

    frame = build_portfolio_frame(asset_returns, weights)
    r = frame["Returns"]
    by_level = {cl: calculate_all_var(r, 1e6, cl) for cl in (0.90, 0.95, 0.99)}
    comp = component_var(asset_returns, weights, 1e6, 0.95)
    comp.insert(1, "Company", comp["Ticker"])
    xlsx = generate_excel_var_report(
        "PORTFOLIO", "3-stock portfolio", "USD", 1e6, 0.95, 1, frame,
        {m: {cl: by_level[cl][m] for cl in by_level} for m in by_level[0.95]},
        backtest_all_methods(r, rolling_var_forecasts(r, 0.95, 250, models=["Historical", "EWMA (RiskMetrics)"]), 0.95), 250,
        run_stress_testing(1e6), historical_worst_losses(r, 1e6), "S&P 500", 1.0,
        portfolio={"components": comp, "diversification": diversification_summary(asset_returns, weights, 1e6, 0.95),
                   "correlation": asset_returns.corr()})
    ws = openpyxl.load_workbook(io.BytesIO(xlsx))["Portfolio Risk"]
    assert [ws.cell(row=r, column=2).value for r in range(13, 16)] == ["AAA", "BBB", "CCC"]
    assert ws.cell(row=16, column=2).value == "TOTAL"
    assert ws.cell(row=16, column=8).value == "=SUM(H13:H15)"
    assert sum(ws.cell(row=r, column=8).value for r in range(13, 16)) == pytest.approx(comp["Component VaR"].sum())
