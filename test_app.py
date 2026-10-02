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
    # The CRO dashboard opens the page: one row per pillar, bottom line, risks and actions
    assert any("CRO dashboard" in h for h in headings) and any(h.startswith("**Bottom line.**") for h in headings)
    dashboard = at.dataframe[0].value
    assert dashboard["Pillar"].tolist()[:3] == ["Market", "Liquidity", "Credit"]
    assert set(dashboard.columns) == {"Pillar", "Headline", "Range", "Grade", "Status"}


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_integrated_stress_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Integrated Stress")
    assert_clean(at)
    labels = [m.label for m in at.metric]
    assert labels[1:] == ["Interaction (cross effect)", "Feedback rounds"]
    table = at.dataframe[0].value
    assert {"Market Loss", "Market", "Liquidity", "Credit", "Events", "Linked Total", "Interaction", "Rounds"} <= set(table.columns)
    assert "Siloed Sum" not in table.columns
    assert any(m.value == "#### Jump to default" for m in at.markdown)
    assert "Market -20%" in table["Scenario"].tolist()
    assert any("Reverse stress test" in m.value for m in at.markdown)
    at.radio(key="reverse_loss").set_value(0.30)
    at.run()
    assert_clean(at)
    assert any("lose 30%" in m.value for m in at.markdown)


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_decisions_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Decisions")
    assert_clean(at)
    limits = at.dataframe[0].value
    # Concentration limits apply to portfolios only
    assert len(limits) == (7 if mode == "Portfolio" else 5) and limits["Status"].str.contains("green|amber|red|⚪").all()
    headings = [m.value for m in at.markdown]
    for section in ("Limits", "What changed in the risk", "Best risk-reducing trades", "One-page CRO memo"):
        assert any(section in h for h in headings), section
    assert any("Exact Shapley split" in c.value for c in at.caption)
    at.selectbox(key="change_months").set_value(12)
    at.run()
    assert_clean(at)


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
    # The page leads with the distance to default and the agencies' default rates; Merton's PD comes third
    labels = [m.label for m in at.metric]
    assert labels[0].startswith("Lowest distance to default")
    assert labels[1] == "Agency 1-year default rate (value-weighted)" and labels[2] == "Merton PD (risk-neutral, model-implied)"
    assert any("risk-neutral probability" in c.value for c in at.caption)
    by_holding = at.dataframe[0].value
    assert by_holding["DD"].iloc[0] != "not available"
    assert {"DD percentile", "Agency 1-yr default rate", "Merton PD (risk-neutral)"} <= set(by_holding.columns)
    if mode == "Portfolio":
        bank = by_holding.set_index("Ticker").loc["HDFCBANK.NS"]
        assert bank["DD"] == "not available" and bank["Altman"] == "not applicable"
        assert any("75% of the portfolio value is modelled" in c.value for c in at.caption)  # HDFC Bank is 25%
    at.selectbox(key="credit_vol").set_value("EWMA")
    at.run()
    assert_clean(at)


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_concentration_page(fake_market, mode):
    at = run_app(mode if mode == "Portfolio" else None, page="Concentration & Factors")
    assert_clean(at)
    headings = [m.value for m in at.markdown]
    assert any("Factor exposures (Fama-French 3 + momentum)" in h for h in headings)
    assert any("What this metric misses" in h for h in headings)
    assert any("Factor data:" in c.value and "IIM Ahmedabad" in c.value for c in at.caption)
    labels = [m.label for m in at.metric]
    assert "Factor share of variance" in labels
    if mode == "Portfolio":
        assert "Effective number of bets (Meucci)" in labels and "Diversification kept in a crisis" in labels
        assert any("Correlation in a crisis" in h for h in headings)
    else:
        assert any("100% concentrated" in i.value for i in at.info)


@pytest.mark.parametrize("mode", ["Single Stock", "Portfolio"])
def test_events_page_without_disclosures(fake_market, mode, monkeypatch, tmp_path):
    import disclosures
    import ui.foundations
    # An empty disclosure folder (the repo ships a dated NSE snapshot, see the next tests)
    monkeypatch.setattr(ui.foundations, "load_disclosures", lambda: disclosures.load_disclosures(tmp_path))
    at = run_app(mode if mode == "Portfolio" else None, page="Event & Governance")
    assert_clean(at)
    assert any("Disclosure data not loaded" in w.value for w in at.warning)
    panel = at.dataframe[0].value
    assert (panel["Promoter pledge"] == "not available").all() and panel["Basis"].str.contains("signals").all()
    assert [m.label for m in at.metric][:2] == ["Event-adjusted ES (95%, 1-day)", "ES added by event risk"]


def _sample_disclosures(tmp_path):
    """Made-up disclosure files for synthetic tickers, dated relative to the synthetic price history (ends 30 Sep 2026)."""
    import json
    (tmp_path / "pledges.csv").write_text(
        "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter,disclosure_date\n"
        "RELIANCE,2025-06-30,50,5,2025-07-15\nRELIANCE,2026-06-30,50,60,2026-07-15\n"
        "TCS,2026-06-30,70,0,2026-07-15\nRELIANCE,2026-09-30,50,90,2026-10-15\n")  # the last row is not public yet
    (tmp_path / "ratings.csv").write_text(
        "symbol,agency,instrument,rating,outlook,action,action_date\n"
        "HDFCBANK,CRISIL,Bonds,CRISIL AA,Negative,downgraded,2026-05-01\n")
    (tmp_path / "ban.csv").write_text("Securities in Ban For Trade Date 25-SEP-2026:\n1,TCS\n")
    entries = [{"file": f, "dataset": d, "source_url": "https://example.org/test", "downloaded_on": "2026-09-30",
                "coverage_start": "2025-01-01", "coverage_end": "2026-09-30"}
               for f, d in (("pledges.csv", "pledges"), ("ratings.csv", "ratings"), ("ban.csv", "fo_ban"))]
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}))


def test_events_page_with_sample_disclosures(fake_market, monkeypatch, tmp_path):
    import disclosures
    import ui.foundations
    _sample_disclosures(tmp_path)
    monkeypatch.setattr(ui.foundations, "load_disclosures", lambda: disclosures.load_disclosures(tmp_path))
    at = run_app("Portfolio", page="Event & Governance")
    assert_clean(at)
    panel = at.dataframe[0].value.set_index("Ticker")
    assert panel.loc["RELIANCE.NS", "Tier"] == "🔴 High"  # 60% pledged; the 90% row is disclosed after the as-of date
    assert "60.0% of promoter holding" in panel.loc["RELIANCE.NS", "Promoter pledge"]
    assert "+55.0 pp" in panel.loc["RELIANCE.NS", "Promoter pledge"]
    assert panel.loc["HDFCBANK.NS", "Tier"] == "🟠 Elevated" and panel.loc["TCS.NS", "Tier"] == "🟠 Elevated"
    gap = at.metric[1].value
    assert gap not in ("₹0", "not available")


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
    positions = next(d.value for d in at.dataframe if list(d.value.columns[:2]) == ["Ticker", "Quantity"])
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
