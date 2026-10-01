"""
Fundamentals Module
Annual financial statements and company profile from Yahoo Finance (via yfinance), normalised to one
set of field names, with field-by-field overrides from a user CSV upload. Every figure carries its source.

Nothing is estimated or filled in: a field Yahoo does not report stays missing and is listed in
`missing_fields`. The credit pillar (Phase 3) decides what it can compute from what is present.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SOURCE_YAHOO = "Yahoo Finance (yfinance)"
SOURCE_UPLOAD = "CSV upload"
NOT_AVAILABLE = "not available"
STATEMENTS = ("income", "balance", "cashflow")
PROFILE = "profile"
MAX_YEARS = 5
# Point-in-time rule when the filing date is unknown: treat a balance sheet as public 60 days after year end
FILING_LAG_DAYS = 60


@dataclass(frozen=True)
class Field:
    name: str
    statement: str
    label: str
    aliases: tuple  # Yahoo row names, in order of preference


# The mapping table. Yahoo's row names have changed between versions ("TotalRevenue" vs "Total Revenue"),
# so names are compared after lowercasing and removing everything except letters and digits.
FIELDS = (
    Field("revenue", "income", "Revenue", ("Total Revenue", "Operating Revenue")),
    Field("ebitda", "income", "EBITDA", ("EBITDA", "Normalized EBITDA")),
    Field("ebit", "income", "EBIT", ("EBIT", "Operating Income")),
    Field("interest_expense", "income", "Interest expense", ("Interest Expense", "Interest Expense Non Operating")),
    Field("pretax_income", "income", "Pre-tax income", ("Pretax Income",)),
    Field("net_income", "income", "Net income", ("Net Income", "Net Income Common Stockholders")),
    Field("total_assets", "balance", "Total assets", ("Total Assets",)),
    Field("total_liabilities", "balance", "Total liabilities", ("Total Liabilities Net Minority Interest", "Total Liabilities")),
    Field("current_assets", "balance", "Current assets", ("Current Assets", "Total Current Assets")),
    Field("current_liabilities", "balance", "Current liabilities", ("Current Liabilities", "Total Current Liabilities")),
    Field("inventory", "balance", "Inventory", ("Inventory",)),
    Field("cash", "balance", "Cash and equivalents", ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")),
    Field("short_term_debt", "balance", "Short-term debt", ("Current Debt", "Current Debt And Capital Lease Obligation")),
    Field("long_term_debt", "balance", "Long-term debt", ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")),
    Field("total_debt", "balance", "Total debt", ("Total Debt",)),
    Field("retained_earnings", "balance", "Retained earnings", ("Retained Earnings",)),
    Field("total_equity", "balance", "Shareholders' equity", ("Stockholders Equity", "Common Stock Equity")),
    Field("working_capital", "balance", "Working capital", ("Working Capital",)),
    Field("operating_cash_flow", "cashflow", "Cash from operations", ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")),
    Field("capex", "cashflow", "Capital expenditure", ("Capital Expenditure",)),
    Field("free_cash_flow", "cashflow", "Free cash flow", ("Free Cash Flow",)),
)
FIELDS_BY_NAME = {f.name: f for f in FIELDS}

# Profile fields: Yahoo `info` key, label, and whether the value is numeric
PROFILE_FIELDS = {
    "shares_outstanding": ("sharesOutstanding", "Shares outstanding", True),
    "market_cap": ("marketCap", "Market capitalisation", True),
    "sector": ("sector", "Sector", False),
    "industry": ("industry", "Industry", False),
    "financial_currency": ("financialCurrency", "Reporting currency", False),
}

TABLE_COLUMNS = ["Statement", "Field", "Period End", "Value", "Filing Date", "Source"]
TEMPLATE_COLUMNS = ["Ticker", "Statement", "Field", "Period End", "Value", "Filing Date", "Source"]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def base_symbol(ticker: str) -> str:
    """Ticker without the exchange suffix: RELIANCE.NS -> RELIANCE."""
    return str(ticker).strip().upper().split(".")[0]


def normalise_statement(raw: pd.DataFrame, statement: str, max_years: int = MAX_YEARS) -> pd.DataFrame:
    """
    Turn a yfinance-shaped statement (rows = Yahoo line items, columns = fiscal year ends) into long rows
    of canonical fields: Statement, Field, Period End, Value, Filing Date, Source. The latest `max_years`
    years with any data are kept. For each field the first alias Yahoo reports in that year is used.
    """
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=TABLE_COLUMNS)
    by_key = {}
    for row_name in raw.index:
        by_key.setdefault(_key(row_name), row_name)
    columns = [c for c in raw.columns if raw[c].notna().any()]
    columns = sorted(columns, key=pd.Timestamp)[-max_years:]
    rows = []
    for field in (f for f in FIELDS if f.statement == statement):
        for col in columns:
            for alias in field.aliases:
                row_name = by_key.get(_key(alias))
                if row_name is None:
                    continue
                value = pd.to_numeric(raw.loc[row_name, col], errors="coerce")
                if np.isfinite(value):
                    rows.append({"Statement": statement, "Field": field.name, "Period End": pd.Timestamp(col).normalize(),
                                 "Value": float(value), "Filing Date": pd.NaT, "Source": f"{SOURCE_YAHOO}: {row_name}"})
                    break
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def missing_fields(table: pd.DataFrame, profile: dict) -> list:
    """Canonical fields with no value in any year, then profile fields with no value."""
    present = set(table["Field"]) if len(table) else set()
    missing = [f.name for f in FIELDS if f.name not in present]
    missing += [name for name in PROFILE_FIELDS if profile.get(name, {}).get("value") is None]
    return missing


def build_fundamentals(symbol: str, statements: dict, info: dict, errors: list = None) -> dict:
    """Assemble the fundamentals result from raw yfinance-shaped statements and an `info` dict."""
    table = pd.concat([normalise_statement(statements.get(s), s) for s in STATEMENTS], ignore_index=True)
    table = table if len(table) else pd.DataFrame(columns=TABLE_COLUMNS)
    profile = {}
    for name, (info_key, _, numeric) in PROFILE_FIELDS.items():
        value = (info or {}).get(info_key)
        if numeric:
            value = pd.to_numeric(value, errors="coerce")
            value = float(value) if value is not None and np.isfinite(value) else None
        elif value in ("", None):
            value = None
        profile[name] = {"value": value, "source": SOURCE_YAHOO if value is not None else NOT_AVAILABLE}
    return {
        "symbol": symbol,
        "table": table,
        "profile": profile,
        "missing_fields": missing_fields(table, profile),
        "errors": list(errors or []),
        "success": bool(len(table)) or any(p["value"] is not None for p in profile.values()),
    }


def fetch_fundamentals(ticker: str) -> dict:
    """
    Annual income statement, balance sheet and cash flow (latest 4-5 years), plus shares outstanding,
    market cap, sector, industry and reporting currency, from yfinance. Never raises: anything that
    fails to download is recorded in `errors` and its fields appear in `missing_fields`.
    """
    symbol = str(ticker).strip().upper()
    statements, info, errors = {}, {}, []
    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
    except Exception as exc:
        return build_fundamentals(symbol, {}, {}, [f"yfinance unavailable: {exc}"])
    for statement, attribute in (("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
        try:
            statements[statement] = getattr(stock, attribute)
        except Exception as exc:
            errors.append(f"{statement}: {exc}")
    try:
        info = stock.info or {}
    except Exception as exc:
        errors.append(f"profile: {exc}")
    return build_fundamentals(symbol, statements, info, errors)


def statement_table(fund: dict, statement: str) -> pd.DataFrame:
    """One statement as a wide table: one row per canonical field, one column per fiscal year end."""
    table = fund["table"]
    rows = table[table["Statement"] == statement]
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot_table(index="Field", columns="Period End", values="Value", aggfunc="first")
    order = [f.name for f in FIELDS if f.statement == statement and f.name in wide.index]
    wide = wide.loc[order]
    wide.index = [FIELDS_BY_NAME[n].label for n in wide.index]
    wide.columns = [f"{c:%Y-%m-%d}" for c in wide.columns]
    return wide


def public_date(period_end, filing_date=None) -> pd.Timestamp:
    """
    The first date a figure may be used in an "as of" analysis: its filing date if known, otherwise
    the fiscal year end plus FILING_LAG_DAYS (the point-in-time rule in RISK_TOOL_PLAN_V3.md).
    """
    if filing_date is not None and not pd.isna(filing_date):
        return pd.Timestamp(filing_date).normalize()
    return pd.Timestamp(period_end).normalize() + pd.Timedelta(days=FILING_LAG_DAYS)


# ---------------------------------------------------------------
# CSV template and overrides
# ---------------------------------------------------------------

def fundamentals_template(ticker: str = "") -> pd.DataFrame:
    """
    An empty override template: one row per field, values blank. Users fill in only the rows they
    want to override and may add one row per fiscal year. Values are in full currency units (not
    lakhs or crores), in the company's reporting currency.
    """
    rows = [{"Ticker": ticker, "Statement": f.statement, "Field": f.name, "Period End": "", "Value": "",
             "Filing Date": "", "Source": ""} for f in FIELDS]
    rows += [{"Ticker": ticker, "Statement": PROFILE, "Field": name, "Period End": "", "Value": "",
              "Filing Date": "", "Source": ""} for name in PROFILE_FIELDS]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def _number(value):
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if number is not None and np.isfinite(number) else None


def _date(value):
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).normalize()
    except (ValueError, TypeError):
        return "invalid"


def validate_overrides(upload: pd.DataFrame) -> tuple:
    """
    Check an uploaded override table. Returns (clean rows, errors); rows with errors are dropped.
    Blank-value rows (untouched template rows) are skipped silently.
    """
    missing_cols = [c for c in ("Ticker", "Statement", "Field", "Value") if c not in upload.columns]
    if missing_cols:
        return pd.DataFrame(columns=TEMPLATE_COLUMNS), [f"Missing column(s): {', '.join(missing_cols)}"]
    clean, errors = [], []
    for i, row in upload.reset_index(drop=True).iterrows():
        line = i + 2  # spreadsheet row number, after the header
        raw_value = row.get("Value")
        if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()) or (
                not isinstance(raw_value, str) and pd.isna(raw_value)):
            continue
        statement = str(row.get("Statement", "")).strip().lower()
        field = str(row.get("Field", "")).strip().lower()
        ticker = str(row.get("Ticker", "")).strip().upper()
        source = str(row.get("Source", "") if not pd.isna(row.get("Source", "")) else "").strip()
        if not ticker:
            errors.append(f"Row {line}: Ticker is blank.")
            continue
        if statement == PROFILE:
            if field not in PROFILE_FIELDS:
                errors.append(f"Row {line}: unknown profile field '{field}'.")
                continue
            numeric = PROFILE_FIELDS[field][2]
            value = _number(raw_value) if numeric else str(raw_value).strip()
            if value is None:
                errors.append(f"Row {line}: {field} must be a number.")
                continue
            clean.append({"Ticker": ticker, "Statement": PROFILE, "Field": field, "Period End": None,
                          "Value": value, "Filing Date": None, "Source": source})
            continue
        if statement not in STATEMENTS:
            errors.append(f"Row {line}: Statement must be one of {', '.join(STATEMENTS + (PROFILE,))}.")
            continue
        if field not in FIELDS_BY_NAME or FIELDS_BY_NAME[field].statement != statement:
            errors.append(f"Row {line}: '{field}' is not a field of the {statement} statement.")
            continue
        value = _number(raw_value)
        if value is None:
            errors.append(f"Row {line}: Value must be a number.")
            continue
        period_end = _date(row.get("Period End"))
        if period_end is None or period_end == "invalid":
            errors.append(f"Row {line}: Period End must be a date (YYYY-MM-DD).")
            continue
        filing_date = _date(row.get("Filing Date"))
        if filing_date == "invalid":
            errors.append(f"Row {line}: Filing Date must be a date (YYYY-MM-DD) or blank.")
            continue
        if filing_date is not None and filing_date < period_end:
            errors.append(f"Row {line}: Filing Date is before Period End.")
            continue
        clean.append({"Ticker": ticker, "Statement": statement, "Field": field, "Period End": period_end,
                      "Value": value, "Filing Date": filing_date, "Source": source})
    return pd.DataFrame(clean, columns=TEMPLATE_COLUMNS), errors


def apply_overrides(fund: dict, upload: pd.DataFrame) -> tuple:
    """
    Replace Yahoo figures field by field with the uploaded ones for this ticker (matched without the
    exchange suffix). A row for a year Yahoo does not report is added. Returns (new fundamentals, errors).
    """
    clean, errors = validate_overrides(upload)
    mine = clean[clean["Ticker"].map(base_symbol) == base_symbol(fund["symbol"])]
    table = fund["table"].copy()
    profile = {k: dict(v) for k, v in fund["profile"].items()}
    new_rows = []
    for row in mine.to_dict("records"):
        source = f"{SOURCE_UPLOAD}: {row['Source']}" if row["Source"] else SOURCE_UPLOAD
        if row["Statement"] == PROFILE:
            profile[row["Field"]] = {"value": row["Value"], "source": source}
            continue
        table = table[~((table["Field"] == row["Field"]) & (table["Period End"] == row["Period End"]))]
        new_rows.append({"Statement": row["Statement"], "Field": row["Field"], "Period End": row["Period End"],
                         "Value": row["Value"], "Filing Date": row["Filing Date"] if row["Filing Date"] is not None else pd.NaT,
                         "Source": source})
    if new_rows:
        table = pd.concat([table, pd.DataFrame(new_rows, columns=TABLE_COLUMNS)], ignore_index=True)
        table = table.sort_values(["Statement", "Field", "Period End"]).reset_index(drop=True)
    result = {**fund, "table": table, "profile": profile, "missing_fields": missing_fields(table, profile)}
    return result, errors
