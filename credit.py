"""
Credit risk pillar (Pillar 3): how close each company is to default, from market prices and from its accounts.

- Merton (1974) structural model, KMV conventions: asset value and volatility from equity value and volatility;
  distance to default and a model-implied, risk-neutral PD (not an agency PD). Iterative KMV cross-check.
  Rolling DD using only the balance sheets public at each date.
- Altman Z (1968, manufacturers) and Z'' (non-manufacturers / emerging markets), never from partial inputs.
- Credit ratios with 4-5 year trends and red flags (thresholds in config/credit_thresholds.json).
- Rating actions from the disclosure data, as of a date.
- Banks, NBFCs and insurers are not modelled with these tools (their liabilities are not debt in the same sense).

Methods and sources: docs/methodology.md section 10.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, root
from scipy.stats import norm

from fundamentals import public_date

CONFIG_PATH = Path(__file__).parent / "config" / "credit_thresholds.json"
DEFAULT_POINT_LTD_WEIGHT = 0.5   # KMV convention: default point = short-term debt + 0.5 × long-term debt
HORIZON_YEARS = 1.0
TRADING_DAYS = 252

# Altman (1968) Z and Altman's Z'' (Altman, 2000; Altman et al., 2017), without the +3.25 emerging-market constant
Z_COEFFICIENTS = (1.2, 1.4, 3.3, 0.6, 1.0)
Z_ZONES = (1.81, 2.99)
Z2_COEFFICIENTS = (6.56, 3.26, 6.72, 1.05)
Z2_ZONES = (1.10, 2.60)
SAFE, GREY, DISTRESS, NOT_AVAILABLE = "safe", "grey", "distress", "not available"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------
# Point-in-time statements
# ---------------------------------------------------------------

def statement_years(fund: dict, as_of=None) -> pd.DataFrame:
    """
    One row per fiscal year (index = period end) and one column per canonical field, keeping only the years
    whose figures were public by `as_of` (filing date if known, else fiscal year end + 60 days). A year with a
    late-filed override row is public only once all its figures are.
    """
    table = fund["table"]
    if table.empty:
        return pd.DataFrame()
    table = table.assign(Public=[public_date(p, f) for p, f in zip(table["Period End"], table["Filing Date"])])
    if as_of is not None:
        public_by_year = table.groupby("Period End")["Public"].max()
        years = public_by_year[public_by_year <= pd.Timestamp(as_of)].index
        table = table[table["Period End"].isin(years)]
    if table.empty:
        return pd.DataFrame()
    return table.pivot_table(index="Period End", columns="Field", values="Value", aggfunc="first").sort_index()


def latest_year(fund: dict, as_of=None) -> tuple:
    """(period end, {field: value}) of the latest fiscal year public by `as_of`; (None, {}) if there is none."""
    years = statement_years(fund, as_of)
    if years.empty:
        return None, {}
    row = years.iloc[-1]
    return years.index[-1], {k: float(v) for k, v in row.items() if pd.notna(v)}


# ---------------------------------------------------------------
# Merton / KMV
# ---------------------------------------------------------------

def default_point(values: dict, ltd_weight: float = DEFAULT_POINT_LTD_WEIGHT):
    """Short-term debt + weight × long-term debt, or None if either is missing."""
    if "short_term_debt" not in values or "long_term_debt" not in values:
        return None
    return values["short_term_debt"] + ltd_weight * values["long_term_debt"]


def merton_equity(V, sigma_v, D, r, T=HORIZON_YEARS):
    """Equity as a call on the assets: E = V·N(d1) − D·e^(−rT)·N(d2). Returns (E, d1, d2)."""
    d1 = (np.log(V / D) + (r + 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))
    d2 = d1 - sigma_v * np.sqrt(T)
    return V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2), d1, d2


def distance_to_default(V, sigma_v, D, r, T=HORIZON_YEARS) -> float:
    """DD = [ln(V/D) + (r − σ_V²/2)T] / (σ_V √T)."""
    return float((np.log(V / D) + (r - 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T)))


def solve_merton(E: float, sigma_e: float, D: float, r: float, T: float = HORIZON_YEARS) -> dict:
    """
    Solve E = V·N(d1) − D·e^(−rT)·N(d2) and σ_E·E = N(d1)·σ_V·V for V and σ_V, then DD and PD = N(−DD)
    (a model-implied, risk-neutral probability). Solved in logs so V and σ_V stay positive.
    """
    if not (E > 0 and sigma_e > 0 and D > 0):
        return {"V": np.nan, "sigma_v": np.nan, "DD": np.nan, "PD": np.nan, "converged": False}

    def equations(x):
        V, s = np.exp(x)
        e, d1, _ = merton_equity(V, s, D, r, T)
        return [(e - E) / E, (norm.cdf(d1) * s * V - sigma_e * E) / (sigma_e * E)]

    start = np.log([E + D * np.exp(-r * T), sigma_e * E / (E + D)])
    sol = root(equations, start, method="hybr")
    V, s = np.exp(sol.x)
    ok = bool(sol.success and np.max(np.abs(equations(sol.x))) < 1e-8)
    dd = distance_to_default(V, s, D, r, T)
    return {"V": float(V), "sigma_v": float(s), "DD": dd, "PD": float(norm.cdf(-dd)), "converged": ok}


def kmv_iterative(equity: pd.Series, D: float, r: float, T: float = HORIZON_YEARS, tol: float = 1e-4,
                  max_iter: int = 100) -> dict:
    """
    Iterative KMV estimate (Crosbie and Bohn, 2003; Vassalou and Xing, 2004): given a guess of σ_V, back out
    each day's asset value from that day's equity value, re-estimate σ_V from the asset returns, and repeat
    until σ_V settles. DD and PD use the last day.
    """
    e = np.asarray(equity, dtype=float)
    e = e[np.isfinite(e) & (e > 0)]
    if len(e) < 60 or not D > 0:
        return {"V": np.nan, "sigma_v": np.nan, "DD": np.nan, "PD": np.nan, "iterations": 0, "converged": False}
    sigma_e = np.std(np.diff(np.log(e)), ddof=1) * np.sqrt(TRADING_DAYS)
    s = sigma_e * e[-1] / (e[-1] + D)
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        # E(V) − E is negative at V = E (a call is worth less than its underlying) and positive at V = E + 2D + 1
        V = np.array([brentq(lambda v: merton_equity(v, s, D, r, T)[0] - x, x, x + D * 2 + 1.0) for x in e])
        s_new = np.std(np.diff(np.log(V)), ddof=1) * np.sqrt(TRADING_DAYS)
        converged = abs(s_new - s) < tol
        s = s_new
        if converged:
            break
    dd = distance_to_default(V[-1], s, D, r, T)
    return {"V": float(V[-1]), "sigma_v": float(s), "DD": dd, "PD": float(norm.cdf(-dd)), "iterations": iterations,
            "converged": converged}


def rolling_dd(prices: pd.DataFrame, fund: dict, r: float, ltd_weight: float = DEFAULT_POINT_LTD_WEIGHT,
               fallback_shares: float = None, min_returns: int = 126) -> pd.DataFrame:
    """
    Month-end distance to default over the price history. At each month end only the latest balance sheet
    public by then is used (point in time); equity = that day's close × that balance sheet's shares in issue
    (or `fallback_shares`, flagged); σ_E = the last year of daily returns, annualised.
    """
    df = prices.set_index(pd.to_datetime(prices["Date"]))
    month_ends = df.groupby(df.index.to_period("M")).tail(1).index
    rows = []
    for date in month_ends:
        period_end, values = latest_year(fund, as_of=date)
        returns = df.loc[:date, "Returns"].dropna().tail(TRADING_DAYS)
        D = default_point(values, ltd_weight) if values else None
        shares = values.get("shares_issued") if values else None
        shares_source = "balance sheet"
        if shares is None and fallback_shares:
            shares, shares_source = fallback_shares, "today's shares outstanding"
        if D is None or shares is None or len(returns) < min_returns:
            continue
        sigma_e = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
        res = solve_merton(float(df.loc[date, "Close"]) * shares, sigma_e, D, r)
        rows.append({"Date": date, "Balance Sheet": period_end, "Equity": float(df.loc[date, "Close"]) * shares,
                     "Default Point": D, "Equity Vol": sigma_e, "DD": res["DD"], "PD": res["PD"],
                     "Shares From": shares_source})
    return pd.DataFrame(rows, columns=["Date", "Balance Sheet", "Equity", "Default Point", "Equity Vol", "DD", "PD",
                                       "Shares From"])


# ---------------------------------------------------------------
# Altman
# ---------------------------------------------------------------

def altman_ratios(values: dict, market_cap: float = None) -> tuple:
    """
    X1 working capital / total assets, X2 retained earnings / TA, X3 EBIT / TA, X4 market equity / total
    liabilities (Z) and book equity / total liabilities (Z''), X5 revenue / TA. Returns (ratios, missing inputs).
    Working capital is current assets − current liabilities, or the reported working-capital line.
    """
    v = values
    missing = []
    if "current_assets" in v and "current_liabilities" in v:
        wc = v["current_assets"] - v["current_liabilities"]
    elif "working_capital" in v:
        wc = v["working_capital"]
    else:
        wc = None
        missing.append("working capital (current assets and liabilities)")
    for field in ("total_assets", "retained_earnings", "ebit", "total_liabilities", "total_equity", "revenue"):
        if field not in v:
            missing.append(field)
    ta, tl = v.get("total_assets"), v.get("total_liabilities")
    ratios = {}
    if ta:
        ratios.update({"X1": wc / ta if wc is not None else None,
                       "X2": v["retained_earnings"] / ta if "retained_earnings" in v else None,
                       "X3": v["ebit"] / ta if "ebit" in v else None,
                       "X5": v["revenue"] / ta if "revenue" in v else None})
    if tl:
        ratios["X4_market"] = market_cap / tl if market_cap else None
        ratios["X4_book"] = v["total_equity"] / tl if "total_equity" in v else None
    if not market_cap:
        missing.append("market capitalisation")
    return ratios, missing


def _zone(score, cutoffs) -> str:
    if score is None or not np.isfinite(score):
        return NOT_AVAILABLE
    return DISTRESS if score < cutoffs[0] else GREY if score <= cutoffs[1] else SAFE


def altman_z(ratios: dict) -> tuple:
    """Altman (1968) Z with zones; (None, 'not available') if any of the five inputs is missing."""
    keys = ("X1", "X2", "X3", "X4_market", "X5")
    if any(ratios.get(k) is None for k in keys):
        return None, NOT_AVAILABLE
    score = float(sum(c * ratios[k] for c, k in zip(Z_COEFFICIENTS, keys)))
    return score, _zone(score, Z_ZONES)


def altman_z2(ratios: dict) -> tuple:
    """Altman Z'' (four variables, book equity) with zones; (None, 'not available') if any input is missing."""
    keys = ("X1", "X2", "X3", "X4_book")
    if any(ratios.get(k) is None for k in keys):
        return None, NOT_AVAILABLE
    score = float(sum(c * ratios[k] for c, k in zip(Z2_COEFFICIENTS, keys)))
    return score, _zone(score, Z2_ZONES)


def primary_altman(sector, indian: bool, config: dict) -> str:
    """Z'' for Indian firms and non-manufacturers; Z for US firms in the configured manufacturing sectors."""
    return "Z" if not indian and sector in config["manufacturing_sectors"] else "Z''"


# ---------------------------------------------------------------
# Ratios and red flags
# ---------------------------------------------------------------

RATIO_LABELS = {
    "debt_to_equity": "Debt / equity", "net_debt_to_ebitda": "Net debt / EBITDA", "interest_cover": "Interest cover",
    "current_ratio": "Current ratio", "quick_ratio": "Quick ratio", "cfo_to_ebitda": "Cash from operations / EBITDA",
}


def _div(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def credit_ratios(years: pd.DataFrame) -> pd.DataFrame:
    """The six credit ratios for every fiscal year (rows) that has their inputs; missing inputs give NaN."""
    rows = {}
    for period, row in years.iterrows():
        v = {k: float(x) for k, x in row.items() if pd.notna(x)}
        debt = v.get("total_debt")
        net_debt = debt - v["cash"] if debt is not None and "cash" in v else None
        rows[period] = {
            "debt_to_equity": _div(debt, v.get("total_equity")),
            "net_debt_to_ebitda": _div(net_debt, v.get("ebitda")),
            "interest_cover": _div(v.get("ebit"), abs(v["interest_expense"]) if "interest_expense" in v else None),
            "current_ratio": _div(v.get("current_assets"), v.get("current_liabilities")),
            "quick_ratio": _div(v["current_assets"] - v["inventory"] if "current_assets" in v and "inventory" in v else None,
                                v.get("current_liabilities")),
            "cfo_to_ebitda": _div(v.get("operating_cash_flow"), v.get("ebitda")),
            "operating_cash_flow": v.get("operating_cash_flow"),
            "ebitda": v.get("ebitda"), "total_equity": v.get("total_equity"),
        }
    return pd.DataFrame.from_dict(rows, orient="index").astype(float).sort_index()


def negative_cfo_streak(ratios: pd.DataFrame) -> int:
    """Consecutive most recent fiscal years with negative cash from operations."""
    streak = 0
    for value in ratios["operating_cash_flow"].dropna()[::-1] if len(ratios) else []:
        if value < 0:
            streak += 1
        else:
            break
    return streak


def red_flags(ratios: pd.DataFrame, config: dict) -> list:
    """Red flags on the latest fiscal year against the configured thresholds; each flag explains itself."""
    if ratios.empty:
        return []
    t = config["red_flags"]
    last = ratios.iloc[-1]
    flags = []

    def check(name, bad, text):
        if bad:
            flags.append(f"{RATIO_LABELS.get(name, name)}: {text}")

    if np.isfinite(last.get("total_equity", np.nan)) and last["total_equity"] <= 0:
        flags.append("Negative or zero book equity")
    else:
        de = last["debt_to_equity"]
        check("debt_to_equity", np.isfinite(de) and de > t["debt_to_equity_max"], f"{de:.2f} above {t['debt_to_equity_max']:g}")
    if np.isfinite(last.get("ebitda", np.nan)) and last["ebitda"] <= 0:
        flags.append("EBITDA zero or negative")
    else:
        nd = last["net_debt_to_ebitda"]
        check("net_debt_to_ebitda", np.isfinite(nd) and nd > t["net_debt_to_ebitda_max"],
              f"{nd:.2f} above {t['net_debt_to_ebitda_max']:g}")
        cf = last["cfo_to_ebitda"]
        check("cfo_to_ebitda", np.isfinite(cf) and cf < t["cfo_to_ebitda_min"], f"{cf:.2f} below {t['cfo_to_ebitda_min']:g}")
    ic = last["interest_cover"]
    check("interest_cover", np.isfinite(ic) and ic < t["interest_cover_min"], f"{ic:.2f} below {t['interest_cover_min']:g}")
    cr = last["current_ratio"]
    check("current_ratio", np.isfinite(cr) and cr < t["current_ratio_min"], f"{cr:.2f} below {t['current_ratio_min']:g}")
    qr = last["quick_ratio"]
    check("quick_ratio", np.isfinite(qr) and qr < t["quick_ratio_min"], f"{qr:.2f} below {t['quick_ratio_min']:g}")
    streak = negative_cfo_streak(ratios)
    if streak >= t["negative_cfo_years"]:
        flags.append(f"Negative cash from operations for {streak} years running")
    return flags


# ---------------------------------------------------------------
# Ratings and financial companies
# ---------------------------------------------------------------

def rating_status(actions: pd.DataFrame, symbol: str, as_of=None) -> dict:
    """Latest rating action public by `as_of` for `symbol`: current rating, last action, its date and direction."""
    if actions is None or actions.empty:
        return {"rating": None, "action": None, "date": None, "direction": "not available", "agency": None}
    rows = actions[actions["symbol"] == symbol]
    if as_of is not None:
        rows = rows[pd.to_datetime(rows["action_date"]) <= pd.Timestamp(as_of)]
    if rows.empty:
        return {"rating": None, "action": None, "date": None, "direction": "no actions on file", "agency": None}
    last = rows.sort_values("action_date").iloc[-1]
    direction = {"upgraded": "up", "downgraded": "down", "placed on watch": "watch",
                 "withdrawn": "withdrawn", "suspended": "suspended"}.get(last["action"], "stable")
    return {"rating": last["rating"], "action": last["action"], "date": pd.Timestamp(last["action_date"]),
            "direction": direction, "agency": last["agency"]}


def is_financial(sector, industry, config: dict) -> bool:
    """Banks, NBFCs, insurers and similar, by Yahoo sector or industry keywords."""
    if sector in config["financial_sectors"]:
        return True
    return any(k.lower() in str(industry or "").lower() for k in config["financial_industry_keywords"])
