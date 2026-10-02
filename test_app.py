"""
Smoke tests for the Streamlit app itself, run with synthetic market data (see conftest.fake_fetch_stock_data).
They load the full app in each mode and drive the main widgets, checking that nothing raises.
"""

import pytest
from streamlit.testing.v1 import AppTest

TIMEOUT = 180


def run_app(mode=None, page="Market", **sidebar):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    if page != "Overview":
        at.radio(key="page").set_value(page)
        at.run()
    if mode:
        at.sidebar.radio[0].set_value(mode)
        at.run()
    selects = {w.label: w for w in at.sidebar.selectbox}
    for label, value in sidebar.items():
        selects[label].set_value(value)
    if sidebar:
        at.run()
    return at


def assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_single_stock_mode(fake_market):
    at = run_app()
    assert_clean(at)
    assert [t.label for t in at.tabs] == ["📊 Model Comparison", "📈 Price & Volatility", "⚡ Stress Testing", "🔬 Backtesting", "📥 Excel Report"]
    models = at.table[0].value
    assert len(models) == 8 and "GARCH(1,1)-t" in models.index


def test_single_stock_long_history_multi_day(fake_market):
    at = run_app(**{"Historical Lookback Window": "5y", "Holding Period (Days)": 10})
    at.sidebar.select_slider[0].set_value(0.975)
    at.run()
    assert_clean(at)
    assert any("10-day check" in c.value for c in at.caption)


def test_portfolio_mode_with_what_if(fake_market):
    at = run_app("Portfolio")
    assert_clean(at)
    assert "🧩 Portfolio Risk" in [t.label for t in at.tabs]
    for basis in ("Historical VaR", "Parametric VaR"):
        at.radio(key="decomposition_basis").set_value(basis)
        at.run()
        assert_clean(at)
    at.number_input(key="what_if_weight_RELIANCE.NS").set_value(10.0)
    at.run()
    assert_clean(at)
    assert any(c.value.startswith("What-if weights") for c in at.caption)


def test_portfolio_buy_and_hold(fake_market):
    at = run_app("Portfolio")
    at.sidebar.radio[1].set_value("Buy-and-hold (weights drift)")
    at.run()
    assert_clean(at)


def test_short_history_flags_low_power(fake_market):
    at = run_app(**{"Historical Lookback Window": "1y"})
    assert_clean(at)
    assert any("Not enough data" in w.value for w in at.warning)


@pytest.mark.parametrize("edits, expected", [
    ({0: {"Ticker": "AAPL"}}, "different currencies"),
    ({0: {"Ticker": "BADTICKER"}}, "Could not load every holding"),
])
def test_portfolio_input_errors_are_reported(fake_market, edits, expected):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio[0].set_value("Portfolio")
    at.session_state["holdings"] = {"edited_rows": edits, "added_rows": [], "deleted_rows": []}
    at.run()
    assert not at.exception
    assert any(expected in e.value for e in at.error)


def test_bad_single_ticker_is_reported(fake_market):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    at.sidebar.text_input[0].set_value("BADX")
    at.run()
    assert not at.exception
    assert at.error and "BADX" in at.error[0].value


def test_overview_is_the_landing_page(fake_market):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    assert_clean(at)
    assert at.title[0].value == "⚡ Risk Analysis Tool"
    assert at.radio(key="page").value == "Overview"
    assert not at.tabs  # the market tabs live on the Market page
    headings = [m.value for m in at.markdown]
    assert any("Positions" in h for h in headings) and any("Data quality" in h for h in headings)


@pytest.mark.parametrize("page", ["Concentration & Factors", "Event & Governance", "Integrated Stress", "Decisions"])
def test_unbuilt_pages_say_coming_next(fake_market, page):
    at = run_app(page=page)
    assert_clean(at)
    assert any("Coming next" in i.value for i in at.info)


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_trust_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Trust")
    assert_clean(at)
    headings = [m.value for m in at.markdown]
    for section in ("Every model", "Lookback sensitivity", "Ghost effect", "What this metric misses"):
        assert any(section in h for h in headings), section
    assert at.metric[2].label == "Trust grade" and at.metric[2].value in ("A", "B", "C", "D")
    models = at.dataframe[0].value
    assert len(models) == 8 and set(models["Grade"]) <= {"A", "B", "C", "D"}


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_liquidity_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Liquidity")
    assert_clean(at)
    headings = [m.value for m in at.markdown]
    for section in ("Trading capacity", "SEBI/AMFI", "Volume in past crises", "liquidity-adjusted VaR",
                    "Circuit-lock risk", "Amihud", "What this metric misses"):
        assert any(section in h for h in headings), section
    assert [m.label for m in at.metric] == ["Days to liquidate 50% (AMFI)", "Sellable in 5 days", "Liquidity-adjusted VaR",
                                            "Circuit-lock loss"]
    assert any("grade **" in c.value for c in at.caption)
    # The assumptions are sidebar inputs, so every page (and the Excel export) sees the same values
    at.number_input(key="liq_participation").set_value(5.0)
    at.run()
    assert_clean(at)
    assert "Participation rate 5%" in at.caption[1].value or any("Participation rate 5%" in c.value for c in at.caption)


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_credit_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Credit")
    assert_clean(at)
    assert at.metric[0].label == "Weighted PD" and at.metric[1].label == "Credit-implied expected loss"
    assert any("not an agency PD" in c.value for c in at.caption)
    by_holding = at.dataframe[0].value
    assert by_holding["DD"].iloc[0] != "not available"
    if mode == "Portfolio":
        bank = by_holding.set_index("Ticker").loc["HDFCBANK.NS"]
        assert bank["DD"] == "not available" and bank["Altman"] == "not applicable"
        assert any("75% of the portfolio value is modelled" in c.value for c in at.caption)  # HDFC Bank is 25%
    at.selectbox(key="credit_vol").set_value("EWMA")
    at.run()
    assert_clean(at)


def test_headline_numbers_carry_range_and_grade(fake_market):
    at = run_app(page="Market")
    assert_clean(at)
    cards = " ".join(m.value for m in at.markdown if "metric-card" in m.value)
    assert cards.count("90% range") == 3 and cards.count("grade ") == 3
    table = at.table[0].value
    assert "Grade" in table.columns and any("90% range" in c for c in table.columns)


def test_inputs_are_shared_across_pages(fake_market):
    at = run_app("Portfolio", page="Market")
    at.radio(key="page").set_value("Overview")
    at.run()
    assert_clean(at)
    assert at.sidebar.radio[0].value == "Portfolio"  # the sidebar keeps its state when the page changes
    assert "5-stock portfolio" in at.subheader[0].value


@pytest.mark.parametrize("mode, column, amounts", [
    ("Shares", "Shares", [100.0, 150.0, 50.0, 100.0, 20.0]),
    ("Value", "Value", [300000.0, 250000.0, 200000.0, 150000.0, 100000.0]),
])
def test_portfolio_entered_as_shares_or_value(fake_market, mode, column, amounts):
    at = run_app("Portfolio", page="Overview")
    {w.label: w for w in at.sidebar.selectbox}["Holdings entered as"].set_value(mode)
    at.run()
    assert_clean(at)
    positions = at.dataframe[0].value
    assert list(positions.columns[:2]) == ["Ticker", "Quantity"]
    assert positions["Weight"].sum() == pytest.approx(1.0)
    if mode == "Value":
        assert positions.filter(like="Value").iloc[:, 0].tolist() == pytest.approx(amounts)
    else:
        assert positions["Quantity"].tolist() == pytest.approx(amounts)
    # The Market page then runs on the total value of the holdings
    at.radio(key="page").set_value("Market")
    at.run()
    assert_clean(at)


def test_network_guard_blocks_outbound_connections():
    import socket
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=1)
