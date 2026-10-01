"""
Smoke tests for the Streamlit app itself, run with synthetic market data (see conftest.fake_fetch_stock_data).
They load the full app in each mode and drive the main widgets, checking that nothing raises.
"""

import pytest
from streamlit.testing.v1 import AppTest

TIMEOUT = 180


def run_app(mode=None, **sidebar):
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
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


def test_network_guard_blocks_outbound_connections():
    import socket
    with pytest.raises(RuntimeError, match="Network access is disabled"):
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=1)
