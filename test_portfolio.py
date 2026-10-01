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
    from var_calculator import backtest_all_methods, calculate_all_var, historical_worst_losses, rolling_var_forecasts

    frame = build_portfolio_frame(asset_returns, weights)
    r = frame["Returns"]
    by_level = {cl: calculate_all_var(r, 1e6, cl) for cl in (0.90, 0.95, 0.99)}
    decomposition = risk_decomposition(asset_returns, weights, 1e6, 0.95)
    decomposition["table"].insert(1, "Company", decomposition["table"]["Ticker"])
    comp = decomposition["table"]
    xlsx = generate_excel_var_report(
        "PORTFOLIO", "3-stock portfolio", "USD", 1e6, 0.95, 1, frame,
        {m: {cl: by_level[cl][m] for cl in by_level} for m in by_level[0.95]},
        backtest_all_methods(r, rolling_var_forecasts(r, 0.95, 250, models=["Historical", "EWMA (RiskMetrics)"]), 0.95), 250,
        None, historical_worst_losses(r, 1e6), "S&P 500", 1.0,
        portfolio={"decomposition": decomposition, "diversification": diversification_summary(asset_returns, weights, 1e6, 0.95),
                   "correlation": asset_returns.corr()})
    ws = openpyxl.load_workbook(io.BytesIO(xlsx))["Portfolio Risk"]
    assert [ws.cell(row=r, column=2).value for r in range(13, 16)] == ["AAA", "BBB", "CCC"]
    assert ws.cell(row=16, column=2).value == "TOTAL"
    assert ws.cell(row=16, column=7).value == "=SUM(G13:G15)"
    assert sum(ws.cell(row=r, column=7).value for r in range(13, 16)) == pytest.approx(decomposition["total"])


# ---------------------------------------------------------------
# Phase 3: decomposition bases, incremental VaR, what-if, buy-and-hold, alignment
# ---------------------------------------------------------------

from portfolio import (
    BUY_AND_HOLD, DECOMPOSITION_BASES, HISTORICAL_ES, HISTORICAL_VAR, PARAMETRIC_VAR,
    alignment_report, current_weights, drifted_weights, portfolio_risk, risk_decomposition,
    what_if, what_if_weights,
)


@pytest.mark.parametrize("basis", DECOMPOSITION_BASES)
@pytest.mark.parametrize("holding_period", [1, 10])
def test_components_sum_to_the_total_on_every_basis(asset_returns, weights, basis, holding_period):
    res = risk_decomposition(asset_returns, weights, 1_000_000, 0.975, holding_period, basis)
    assert res["table"]["Component"].sum() == pytest.approx(res["total"], rel=1e-9)
    assert res["table"]["Contribution %"].sum() == pytest.approx(1.0)


def test_historical_totals_match_the_headline_models(asset_returns, weights):
    from var_calculator import calculate_historical_var, calculate_parametric_var
    port = build_portfolio_frame(asset_returns, weights)["Returns"]
    hist = calculate_historical_var(port, 1e6, 0.99, 10)
    assert risk_decomposition(asset_returns, weights, 1e6, 0.99, 10, HISTORICAL_ES)["total"] == pytest.approx(hist["cvar_scaled_amount"])
    assert risk_decomposition(asset_returns, weights, 1e6, 0.99, 10, HISTORICAL_VAR)["total"] == pytest.approx(hist["var_scaled_amount"])
    assert risk_decomposition(asset_returns, weights, 1e6, 0.99, 10, PARAMETRIC_VAR)["total"] == pytest.approx(
        calculate_parametric_var(port, 1e6, 0.99, 10)["var_scaled_amount"])


def test_component_es_is_the_tail_average_by_hand():
    r = pd.DataFrame({"A": [-0.10, 0.02, -0.04, 0.01, 0.03], "B": [-0.02, -0.06, 0.01, 0.00, 0.02]})
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    # portfolio returns: -0.06, -0.02, -0.015, 0.005, 0.025; 40th percentile = -0.018 -> tail days 0 and 1
    res = risk_decomposition(r, w, 1.0, 0.60, 1, HISTORICAL_ES)
    comp = res["table"].set_index("Ticker")["Component"]
    assert comp["A"] == pytest.approx(-0.5 * (-0.10 + 0.02) / 2)
    assert comp["B"] == pytest.approx(-0.5 * (-0.02 - 0.06) / 2)
    assert res["total"] == pytest.approx((0.06 + 0.02) / 2)


def test_historical_components_match_parametric_on_normal_data():
    rng = np.random.default_rng(3)
    cov = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 2.0]]) * 1e-4
    r = pd.DataFrame(rng.multivariate_normal([0, 0, 0], cov, 400_000), columns=["A", "B", "C"])
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    param = risk_decomposition(r, w, 1.0, 0.99, 1, PARAMETRIC_VAR)["table"]["Contribution %"]
    hist_var = risk_decomposition(r, w, 1.0, 0.99, 1, HISTORICAL_VAR)["table"]["Contribution %"]
    hist_es = risk_decomposition(r, w, 1.0, 0.99, 1, HISTORICAL_ES)["table"]["Contribution %"]
    # Under normality every Euler allocation gives the same shares: wᵢ(Σw)ᵢ / σₚ²
    assert hist_var.to_numpy() == pytest.approx(param.to_numpy(), abs=0.03)
    assert hist_es.to_numpy() == pytest.approx(param.to_numpy(), abs=0.01)


def test_incremental_var_is_risk_with_minus_without(asset_returns, weights):
    res = risk_decomposition(asset_returns, weights, 1e6, 0.95, 1, HISTORICAL_VAR)
    without_a = portfolio_risk(asset_returns, weights.drop("AAA"), 1e6, 0.95, 1, HISTORICAL_VAR)
    assert res["table"].set_index("Ticker").loc["AAA", "Incremental"] == pytest.approx(res["total"] - without_a)


def test_diversification_benefit_on_every_basis(asset_returns, weights):
    for basis in DECOMPOSITION_BASES:
        res = risk_decomposition(asset_returns, weights, 1e6, 0.95, 1, basis)
        assert res["diversification_benefit"] == pytest.approx(res["standalone_sum"] - res["total"])
        assert res["diversification_benefit"] > 0


def test_what_if_weights_scale_the_others_proportionally():
    w = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    new = what_if_weights(w, {"A": 0.2})
    assert new["A"] == pytest.approx(0.2)
    assert new["B"] / new["C"] == pytest.approx(1.5) and new.sum() == pytest.approx(1.0)
    added = what_if_weights(w, {"D": 0.1})
    assert added["D"] == pytest.approx(0.1) and added["A"] == pytest.approx(0.45)
    with pytest.raises(ValueError):
        what_if_weights(w, {"A": 1.2})


def test_what_if_reports_before_and_after(asset_returns, weights):
    table = what_if(asset_returns, weights, {"AAA": 0.0}, 1e6, 0.95).set_index("Measure")
    expected_after = portfolio_risk(asset_returns, what_if_weights(weights, {"AAA": 0.0}), 1e6, 0.95, 1, HISTORICAL_VAR)
    assert table.loc[HISTORICAL_VAR, "What-if"] == pytest.approx(expected_after)
    assert table.loc[HISTORICAL_VAR, "Current"] == pytest.approx(portfolio_risk(asset_returns, weights, 1e6, 0.95, 1, HISTORICAL_VAR))


def test_buy_and_hold_tracks_the_value_of_shares_bought_once():
    r = pd.DataFrame({"A": [0.10, 0.10, -0.05], "B": [0.00, -0.10, 0.02]})
    w = pd.Series([0.5, 0.5], index=["A", "B"])
    frame = build_portfolio_frame(r, w, rebalance=BUY_AND_HOLD)
    value_a = 0.5 * np.cumprod([1.10, 1.10, 0.95])
    value_b = 0.5 * np.cumprod([1.00, 0.90, 1.02])
    assert frame["Close"].to_numpy() == pytest.approx(100 * (value_a + value_b))
    assert drifted_weights(r, w).iloc[1].to_numpy() == pytest.approx([0.55 / 1.05, 0.5 / 1.05])
    assert current_weights(r, w, BUY_AND_HOLD).to_numpy() == pytest.approx([value_a[-1], value_b[-1]] / (value_a[-1] + value_b[-1]))


def test_alignment_report_names_the_limiting_ticker():
    long = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=300), "Close": np.linspace(100, 130, 300)})
    short = pd.DataFrame({"Date": pd.bdate_range("2020-06-01", periods=150), "Close": np.linspace(50, 60, 150)})
    report = alignment_report({"OLD": long, "NEW": short})
    assert report["limiting_ticker"] == "NEW"
    common = len(set(long["Date"]) & set(short["Date"]))
    assert report["common_days"] == common
    assert report["days_dropped"] == 300 - common
