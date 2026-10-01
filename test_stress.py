"""Phase 3 stress-testing tests: measured drawdowns, recovery, replay, downside beta, scenario config."""

import numpy as np
import pandas as pd
import pytest

from stress import (
    NO_MARKET_DATA, PROXY, REPLAY, SCENARIO_FILE,
    downside_beta, load_scenarios, market_drawdown, replay_return, run_scenarios, volatility_shock,
)


def _prices(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_market_drawdown_and_recovery_by_hand():
    p = _prices([100, 110, 99, 88, 95, 105, 111, 120])
    d = market_drawdown(p, p.index[0], p.index[4])
    assert d["covered"]
    assert d["peak"] == p.index[1] and d["trough"] == p.index[3]
    assert d["drawdown"] == pytest.approx(88 / 110 - 1)
    assert d["recovery_days"] == 3  # trough on day 3, back above 110 on day 6


def test_market_drawdown_without_recovery_and_outside_data():
    p = _prices([100, 90, 80, 85])
    assert market_drawdown(p, p.index[0], p.index[-1])["recovery_days"] is None
    assert not market_drawdown(p, "2010-01-01", "2010-06-30")["covered"]


def test_replay_equals_compounded_actual_return():
    prices = _prices([50, 52, 47, 44, 46])
    returns = prices.pct_change().dropna()
    assert replay_return(returns, prices.index[1], prices.index[3]) == pytest.approx(44 / 52 - 1)
    assert replay_return(returns.iloc[2:], prices.index[1], prices.index[3]) is None  # history starts too late


def test_downside_beta_uses_only_the_worst_market_days():
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, 5000))
    worst = market <= market.quantile(0.10)
    stock = np.where(worst, 2.0 * market, 0.8 * market) + rng.normal(0, 0.001, 5000)
    assert downside_beta(pd.Series(stock), market) == pytest.approx(2.0, abs=0.05)


def test_scenarios_replay_when_data_exist_and_use_downside_beta_otherwise():
    market = _prices(np.r_[np.linspace(100, 120, 20), np.linspace(120, 90, 10), np.linspace(90, 125, 30)])
    scenarios = pd.DataFrame({"scenario": ["Crash"], "start": [market.index[0]], "end": [market.index[35]], "description": ["test"]})
    stock = market.pct_change().dropna() * 1.5
    replay = run_scenarios(stock, market, scenarios, proxy_beta=2.0, investment=1000).iloc[0]
    peak, trough = market.index[19], market.index[29]
    assert replay["Method"] == REPLAY
    assert replay["Position Return"] == pytest.approx(np.prod(1 + stock.loc[(stock.index > peak) & (stock.index <= trough)]) - 1)

    short_history = stock.loc[market.index[25]:]
    proxy = run_scenarios(short_history, market, scenarios, proxy_beta=2.0, investment=1000).iloc[0]
    assert proxy["Method"] == PROXY
    assert proxy["Position Return"] == pytest.approx(2.0 * (90 / 120 - 1))
    assert proxy["P&L"] == pytest.approx(1000 * 2.0 * (90 / 120 - 1))


def test_scenario_outside_market_data_is_flagged():
    market = _prices(np.linspace(100, 110, 30), start="2021-01-01")
    scenarios = pd.DataFrame({"scenario": ["Old"], "start": [pd.Timestamp("2008-01-01")], "end": [pd.Timestamp("2009-01-01")], "description": [""]})
    assert run_scenarios(market.pct_change().dropna(), market, scenarios, 1.0, 1000).iloc[0]["Method"] == NO_MARKET_DATA


def test_scenario_config_is_well_formed():
    table = load_scenarios(SCENARIO_FILE)
    assert set(table["market"]) == {"NIFTY50", "SP500"}
    assert (table["start"] < table["end"]).all()
    assert not table.duplicated(["market", "scenario"]).any()
    assert table["start"].min() >= pd.Timestamp("2007-01-01")  # Yahoo's Nifty 50 history starts Sep 2007
    assert len(load_scenarios(market="NIFTY50")) == 7


def test_volatility_shock_scales_historical_loss_with_k():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    table = volatility_shock(r, 1e6, 0.99).set_index("Volatility")
    base = table.loc["σ × 1", "Historical VaR"]
    mu = r.mean() * 1e6
    # VaR = −(μ + k·(q − μ)), so VaR(k) + μ scales exactly with k
    assert table.loc["σ × 2", "Historical VaR"] + mu == pytest.approx(2 * (base + mu))
    assert table.loc["σ × 3", "Normal VaR"] > table.loc["σ × 2", "Normal VaR"] > table.loc["σ × 1", "Normal VaR"]


def test_downside_beta_is_stable_when_the_market_range_is_narrow():
    # Pure noise around a true crisis beta of 1.2: the estimate should stay close across samples
    estimates = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        market = pd.Series(rng.normal(0, 0.01, 500))
        stock = pd.Series(1.2 * market + rng.normal(0, 0.01, 500))
        estimates.append(downside_beta(stock, market))
    assert np.mean(estimates) == pytest.approx(1.2, abs=0.05)
    assert np.std(estimates) < 0.15


def test_excel_stress_sheet_lists_measured_scenarios():
    import io
    import openpyxl
    from excel_exporter import generate_excel_var_report
    from var_calculator import calculate_all_var, historical_worst_losses

    market = _prices(np.r_[np.linspace(100, 120, 40), np.linspace(120, 90, 20), np.linspace(90, 125, 60)])
    returns = market.pct_change().dropna()
    scenarios = pd.DataFrame({"scenario": ["Crash", "Old"], "start": [market.index[0], pd.Timestamp("2001-01-01")],
                              "end": [market.index[80], pd.Timestamp("2001-12-31")], "description": ["test", "before data"]})
    table = run_scenarios(returns, market, scenarios, 1.0, 1000)
    df = pd.DataFrame({"Date": returns.index, "Close": market.iloc[1:].to_numpy(), "Returns": returns.to_numpy(),
                       "Log_Returns": np.log1p(returns.to_numpy()), "Rolling_30d_Vol": returns.rolling(30).std().to_numpy()})
    df["Returns"] = df["Returns"].astype(float)
    by_level = {cl: calculate_all_var(df["Returns"], 1000, cl) for cl in (0.95, 0.99)}
    xlsx = generate_excel_var_report(
        "X", "X", "USD", 1000, 0.99, 1, df, {m: {cl: by_level[cl][m] for cl in by_level} for m in by_level[0.99]},
        None, 250, table, historical_worst_losses(df["Returns"], 1000), "S&P 500", 1.0,
        vol_shock_table=volatility_shock(df["Returns"], 1000, 0.99), beta_down=1.3)
    ws = openpyxl.load_workbook(io.BytesIO(xlsx))["Stress Testing"]
    values = [[c.value for c in row] for row in ws.iter_rows(min_row=5, max_row=12, min_col=2, max_col=9)]
    assert values[0][:2] == ["Scenario", "Description"]
    assert values[1][0] == "Crash" and values[1][6] == REPLAY
    assert values[1][4] == pytest.approx(90 / 120 - 1)
    assert all(row[0] != "Old" for row in values)  # outside the market data: left out
    assert "VOLATILITY SHOCK" in str(values[3][0])
