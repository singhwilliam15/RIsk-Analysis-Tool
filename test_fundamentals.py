"""
Fundamentals: field mapping on saved yfinance frames, the fetcher's failure handling, CSV overrides and the
point-in-time date rule. The frames in test_data/yfinance/ were saved from yfinance 1.7.0 on 2 Oct 2026
(RELIANCE.NS and AAPL annual statements), so the mapping is tested on real Yahoo row names and gaps.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fundamentals as F

FIXTURES = Path(__file__).parent / "test_data" / "yfinance"


def saved(symbol: str, statement: str) -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / f"{symbol}_{statement}.csv", index_col=0)
    frame.columns = pd.to_datetime(frame.columns)
    return frame


def statements(symbol: str) -> dict:
    return {s: saved(symbol, s) for s in F.STATEMENTS}


def value(fund: dict, field: str, period: str) -> float:
    table = fund["table"]
    rows = table[(table["Field"] == field) & (table["Period End"] == pd.Timestamp(period))]
    assert len(rows) == 1, f"{field} {period}: {len(rows)} rows"
    return rows["Value"].iloc[0]


@pytest.mark.parametrize("symbol", ["RELIANCE.NS", "AAPL"])
def test_every_mapped_value_equals_the_yahoo_row_it_names(symbol):
    raw = statements(symbol)
    fund = F.build_fundamentals(symbol, raw, {})
    assert len(fund["table"]) > 50
    for row in fund["table"].itertuples(index=False):
        yahoo_row = row.Source.split(": ", 1)[1]
        assert raw[row.Statement].loc[yahoo_row, row._2] == pytest.approx(row.Value)
        assert yahoo_row in F.FIELDS_BY_NAME[row.Field].aliases


def test_indian_frame_maps_the_expected_rows():
    fund = F.build_fundamentals("RELIANCE.NS", statements("RELIANCE.NS"), {})
    raw = saved("RELIANCE.NS", "balance")
    assert value(fund, "total_liabilities", "2026-03-31") == raw.loc["Total Liabilities Net Minority Interest", pd.Timestamp("2026-03-31")]
    assert value(fund, "short_term_debt", "2025-03-31") == raw.loc["Current Debt", pd.Timestamp("2025-03-31")]
    assert value(fund, "revenue", "2024-03-31") == saved("RELIANCE.NS", "income").loc["Total Revenue", pd.Timestamp("2024-03-31")]
    # Yahoo's oldest column (FY2022) is empty for Reliance, so four fiscal years remain
    assert sorted(fund["table"]["Period End"].unique()) == list(pd.to_datetime(["2023-03-31", "2024-03-31", "2025-03-31", "2026-03-31"]))


def test_us_frame_keeps_a_field_reported_in_some_years_only():
    fund = F.build_fundamentals("AAPL", statements("AAPL"), {})
    years = fund["table"].loc[fund["table"]["Field"] == "interest_expense", "Period End"]
    assert len(years) == 2  # Apple stopped reporting interest expense separately
    assert "interest_expense" not in fund["missing_fields"]


def test_missing_rows_are_listed_not_raised():
    raw = statements("RELIANCE.NS")
    raw["balance"] = raw["balance"].drop(index=["Inventory", "Retained Earnings"])
    raw["cashflow"] = raw["cashflow"].iloc[0:0]  # no cash flow statement at all
    fund = F.build_fundamentals("RELIANCE.NS", raw, {"sector": "Energy"})
    for field in ("inventory", "retained_earnings", "operating_cash_flow", "capex", "free_cash_flow"):
        assert field in fund["missing_fields"]
    assert "sector" not in fund["missing_fields"] and "market_cap" in fund["missing_fields"]
    assert fund["profile"]["sector"] == {"value": "Energy", "source": F.SOURCE_YAHOO}
    assert fund["profile"]["market_cap"] == {"value": None, "source": F.NOT_AVAILABLE}


def test_alias_fallback_and_old_style_row_names():
    raw = statements("AAPL")
    renamed = {s: f.rename(index=lambda name: name.replace(" ", "")) for s, f in raw.items()}  # "TotalRevenue"
    assert F.build_fundamentals("AAPL", renamed, {})["table"]["Value"].tolist() == \
        F.build_fundamentals("AAPL", raw, {})["table"]["Value"].tolist()

    raw["balance"] = raw["balance"].drop(index="Total Liabilities Net Minority Interest")
    raw["balance"].loc["Total Liabilities"] = 1.0
    fund = F.build_fundamentals("AAPL", raw, {})
    assert value(fund, "total_liabilities", "2025-09-30") == 1.0
    assert fund["table"].loc[fund["table"]["Field"] == "total_liabilities", "Source"].iloc[0].endswith(": Total Liabilities")


def test_keeps_only_the_latest_five_years():
    columns = pd.to_datetime([f"{y}-12-31" for y in range(2015, 2025)])
    raw = pd.DataFrame([np.arange(10.0)], index=["Total Revenue"], columns=columns)
    table = F.normalise_statement(raw, "income")
    assert table["Period End"].dt.year.tolist() == [2020, 2021, 2022, 2023, 2024]


def _fake_yfinance(fail=()):
    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def _get(self, name, statement):
            if name in fail:
                raise RuntimeError(f"{name} blocked")
            return saved("RELIANCE.NS", statement)

        income_stmt = property(lambda self: self._get("income_stmt", "income"))
        balance_sheet = property(lambda self: self._get("balance_sheet", "balance"))
        cashflow = property(lambda self: self._get("cashflow", "cashflow"))

        @property
        def info(self):
            if "info" in fail:
                raise RuntimeError("429 Too Many Requests")
            return {"sharesOutstanding": 13_532_000_000, "marketCap": 1.9e13, "sector": "Energy",
                    "industry": "Oil & Gas Refining & Marketing", "financialCurrency": "INR"}

    return types.SimpleNamespace(Ticker=Ticker)


def test_fetch_fundamentals_with_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance())
    fund = F.fetch_fundamentals("reliance.ns")
    assert fund["success"] and fund["symbol"] == "RELIANCE.NS" and fund["errors"] == []
    assert fund["profile"]["shares_outstanding"]["value"] == 13_532_000_000
    assert fund["profile"]["financial_currency"]["value"] == "INR"
    assert fund["missing_fields"] == []


def test_fetch_fundamentals_never_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(fail=("balance_sheet", "info")))
    fund = F.fetch_fundamentals("RELIANCE.NS")
    assert any("balance" in e for e in fund["errors"]) and any("profile" in e for e in fund["errors"])
    assert "total_assets" in fund["missing_fields"] and "revenue" not in fund["missing_fields"]

    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(fail=("income_stmt", "balance_sheet", "cashflow", "info")))
    fund = F.fetch_fundamentals("RELIANCE.NS")
    assert not fund["success"] and len(fund["missing_fields"]) == len(F.FIELDS) + len(F.PROFILE_FIELDS)


def test_statement_table_is_wide_with_labels():
    fund = F.build_fundamentals("AAPL", statements("AAPL"), {})
    wide = F.statement_table(fund, "income")
    assert wide.columns.tolist() == ["2022-09-30", "2023-09-30", "2024-09-30", "2025-09-30"]
    assert wide.index[0] == "Revenue"
    assert F.statement_table(F.build_fundamentals("X", {}, {}), "income").empty


# ---------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------

def overrides(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=F.TEMPLATE_COLUMNS).astype(str).replace("None", "")


def test_override_replaces_one_figure_and_adds_a_year():
    fund = F.build_fundamentals("RELIANCE.NS", statements("RELIANCE.NS"), {})
    upload = overrides([
        ["RELIANCE", "balance", "total_liabilities", "2026-03-31", "1,000,000", "2026-05-20", "Annual report FY26 p.212"],
        ["RELIANCE.NS", "income", "revenue", "2022-03-31", "7.0e12", "", "Annual report FY22"],
        ["RELIANCE.BO", "profile", "sector", "", "Conglomerate", "", ""],
        ["TCS", "income", "revenue", "2026-03-31", "5", "", "other company"],
    ])
    new, errors = F.apply_overrides(fund, upload)
    assert errors == []
    assert value(new, "total_liabilities", "2026-03-31") == 1_000_000
    assert value(new, "total_liabilities", "2025-03-31") == value(fund, "total_liabilities", "2025-03-31")
    row = new["table"][(new["table"]["Field"] == "total_liabilities") & (new["table"]["Period End"] == "2026-03-31")].iloc[0]
    assert row["Source"] == "CSV upload: Annual report FY26 p.212" and row["Filing Date"] == pd.Timestamp("2026-05-20")
    assert value(new, "revenue", "2022-03-31") == 7.0e12
    assert value(new, "revenue", "2026-03-31") == value(fund, "revenue", "2026-03-31")  # the TCS row is ignored
    assert new["profile"]["sector"] == {"value": "Conglomerate", "source": "CSV upload"}
    assert len(new["table"]) == len(fund["table"]) + 1
    assert len(fund["table"]) == len(F.build_fundamentals("RELIANCE.NS", statements("RELIANCE.NS"), {})["table"])  # unchanged


def test_override_fills_a_missing_field():
    fund = F.build_fundamentals("X.NS", {}, {})
    assert "market_cap" in fund["missing_fields"]
    new, _ = F.apply_overrides(fund, overrides([["X", "profile", "market_cap", "", "5e10", "", "NSE quote"],
                                                ["X", "balance", "total_assets", "2026-03-31", "100", "", ""]]))
    assert "market_cap" not in new["missing_fields"] and "total_assets" not in new["missing_fields"]
    assert new["profile"]["market_cap"]["value"] == 5e10


@pytest.mark.parametrize("row, message", [
    (["X", "income", "total_assets", "2026-03-31", "1", "", ""], "not a field of the income statement"),
    (["X", "notes", "revenue", "2026-03-31", "1", "", ""], "Statement must be one of"),
    (["X", "income", "revenue", "2026-03-31", "lots", "", ""], "Value must be a number"),
    (["X", "income", "revenue", "", "1", "", ""], "Period End must be a date"),
    (["X", "income", "revenue", "2026-03-31", "1", "yesterday", ""], "Filing Date must be a date"),
    (["X", "income", "revenue", "2026-03-31", "1", "2026-01-01", ""], "Filing Date is before Period End"),
    (["X", "profile", "market_cap", "", "big", "", ""], "market_cap must be a number"),
    (["", "income", "revenue", "2026-03-31", "1", "", ""], "Ticker is blank"),
])
def test_bad_override_rows_are_rejected_with_the_row_number(row, message):
    clean, errors = F.validate_overrides(overrides([["X", "income", "revenue", "2025-03-31", "1", "", ""], row]))
    assert len(clean) == 1
    assert len(errors) == 1 and errors[0].startswith("Row 3:") and message in errors[0]


def test_untouched_template_rows_are_skipped():
    template = F.fundamentals_template("RELIANCE.NS")
    assert len(template) == len(F.FIELDS) + len(F.PROFILE_FIELDS)
    assert set(template["Field"]) == set(F.FIELDS_BY_NAME) | set(F.PROFILE_FIELDS)
    template.loc[0, ["Period End", "Value"]] = ["2026-03-31", "42"]
    clean, errors = F.validate_overrides(template)
    assert errors == [] and len(clean) == 1


def test_upload_without_required_columns():
    _, errors = F.validate_overrides(pd.DataFrame({"Ticker": ["X"], "Value": ["1"]}))
    assert errors and "Statement" in errors[0]


def test_point_in_time_date():
    assert F.public_date("2026-03-31", "2026-05-20") == pd.Timestamp("2026-05-20")
    assert F.public_date("2026-03-31") == pd.Timestamp("2026-05-30")  # year end + 60 days
    assert F.public_date(pd.Timestamp("2025-12-31"), pd.NaT) == pd.Timestamp("2026-03-01")
