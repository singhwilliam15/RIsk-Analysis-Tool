"""Credit calculations shared by every page (Pillar 3), with ranges and grades on the headline numbers."""

from pathlib import Path

import numpy as np
import pandas as pd

import banks as B
import credit as C
import debt as D
from disclosures import normalise_symbol
from garch import PCT, garch_filter, garch_variance_term_structure
from trust import TrustedMetric, grade
from ui.cache import cached_credit_holding, cached_fit_models
from ui.context import export
from var_calculator import ewma_volatility

HISTORICAL, EWMA, GARCH = "Historical (1 year)", "EWMA", "GARCH(1,1)-t"
EQUITY_VOL_CHOICES = (HISTORICAL, EWMA, GARCH)
INDIAN = (".NS", ".BO")
VOL_RANGE_LABEL = "range across equity-volatility inputs"


DD_UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "credit" / "dd_universe.csv"


def load_dd_universe(path: Path = DD_UNIVERSE_PATH) -> pd.DataFrame:
    """Merton DDs of the reference universe (scripts/build_dd_universe.py); empty if not built."""
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()


def equity_vols(returns: pd.Series, garch_fit: dict) -> dict:
    """Annualised equity volatility three ways: last year's daily returns, EWMA today, GARCH-t over the next year."""
    r = returns.dropna()
    vols = {HISTORICAL: float(r.tail(C.TRADING_DAYS).std(ddof=1) * np.sqrt(C.TRADING_DAYS)),
            EWMA: float(ewma_volatility(r)[1] * np.sqrt(C.TRADING_DAYS)), GARCH: np.nan}
    if garch_fit and garch_fit.get("converged"):
        p = garch_fit["params"]
        sigma_next = garch_filter(r, p)[-1]
        # Average variance over the next year from the GARCH term structure, so it matches a one-year horizon
        vols[GARCH] = float(np.sqrt(garch_variance_term_structure(p, sigma_next ** 2, C.TRADING_DAYS).sum()) / PCT)
    return vols


def analyse_holding(ticker: str, prices: pd.DataFrame, fund: dict, vols: dict, sector, industry, trading_currency: str,
                    ratings: pd.DataFrame, r: float, ltd_weight: float, T: float, config: dict, as_of) -> dict:
    """Every credit figure for one holding, using only statements public by `as_of`."""
    out = {"Ticker": ticker, "notes": [], "vols": vols}
    period_end, values = C.latest_year(fund, as_of)
    years = C.statement_years(fund, as_of)
    out.update(period_end=period_end, years=len(years),
               age_months=(pd.Timestamp(as_of) - period_end).days / 30.44 if period_end is not None else np.nan)
    out["rating"] = C.rating_status(ratings, normalise_symbol(ticker), as_of)
    out["financial"] = C.is_financial(sector, industry, config)
    out["ratios"] = C.credit_ratios(years) if len(years) else pd.DataFrame()
    if out["financial"]:
        out["bank"] = B.assess(normalise_symbol(ticker), as_of) if ticker.upper().endswith(INDIAN) else {"available": False}
        out["notes"].append("Bank, NBFC or insurer: Merton, Altman and leverage ratios are not meaningful when deposits "
                            "and policy liabilities are the business. "
                            + ("Assessed against RBI's Prompt Corrective Action thresholds instead." if out["bank"]["available"]
                               else "No bank metrics file (data/banks/); see the manual panel."))
        out.update(merton={}, kmv={}, flags=[], z=(None, C.NOT_AVAILABLE), z2=(None, C.NOT_AVAILABLE),
                   primary="not applicable", altman_missing=[], rolling=pd.DataFrame())
        return out
    out["flags"] = C.red_flags(out["ratios"], config) if len(out["ratios"]) else []

    reporting = (fund["profile"].get("financial_currency") or {}).get("value")
    currency_ok = reporting is None or reporting == trading_currency
    if not currency_ok:
        out["notes"].append(f"Statements are in {reporting} but the shares trade in {trading_currency}: Merton and "
                            "Altman Z (market equity) are not computed without FX conversion.")
    shares = (fund["profile"].get("shares_outstanding") or {}).get("value") or values.get("shares_issued")
    close = float(prices["Close"].iloc[-1])
    market_cap = close * shares if shares and currency_ok else None
    D = C.default_point(values, ltd_weight)
    if D is None:
        out["notes"].append("Short-term or long-term debt is missing, so the default point and Merton are not available.")
    elif D <= 0:
        out["notes"].append("No debt in the default point: distance to default is not meaningful (no default barrier).")
        D = None
    out.update(default_point=D, market_cap=market_cap, shares=shares)

    merton = {}
    if D and market_cap:
        for name, sigma in vols.items():
            merton[name] = C.solve_merton(market_cap, sigma, D, r, T) if np.isfinite(sigma) else \
                {"V": np.nan, "sigma_v": np.nan, "DD": np.nan, "PD": np.nan, "converged": False}
        equity = prices["Close"].tail(C.TRADING_DAYS) * shares
        out["kmv"] = C.kmv_iterative(equity, D, r, T)
        out["rolling"] = C.rolling_dd(prices, fund, r, ltd_weight, fallback_shares=shares)
    else:
        out["kmv"], out["rolling"] = {}, pd.DataFrame()
    out["merton"] = merton

    ratios, missing = C.altman_ratios(values, market_cap)
    out.update(altman_ratios=ratios, altman_missing=missing, z=C.altman_z(ratios), z2=C.altman_z2(ratios),
               primary=C.primary_altman(sector, ticker.endswith(INDIAN), config))
    return out


def portfolio_view(results: dict, positions: pd.DataFrame, vol_choice: str) -> dict:
    """Per-holding table, value-weighted PD over the modelled holdings and the credit-implied expected loss."""
    values = positions.set_index("Ticker")["Value"]
    rows = []
    for t, res in results.items():
        m = res["merton"].get(vol_choice, {})
        primary = res["z2"] if res["primary"] == "Z''" else res["z"]
        rows.append({"Ticker": t, "Value": values[t], "Financial": res["financial"],
                     "DD": m.get("DD", np.nan), "PD": m.get("PD", np.nan),
                     "Altman Model": res["primary"], "Altman Score": primary[0] if primary[0] is not None else np.nan,
                     "Altman Zone": primary[1], "Red Flags": len(res["flags"]),
                     "Rating": res["rating"]["rating"] or "not available", "Rating Direction": res["rating"]["direction"],
                     "PCA Band": (res.get("bank") or {}).get("pca", {}).get("worst", "-") if res["financial"] else "-",
                     "Balance Sheet": res["period_end"]})
    table = pd.DataFrame(rows)
    modelled = table["PD"].notna()

    def weighted(pds):
        covered = values[table.loc[pds.notna(), "Ticker"]].sum()
        return float((pds.fillna(0).to_numpy() * table["Value"].to_numpy()).sum() / covered) if covered else np.nan

    def expected_loss(pds):
        # Not available rather than zero when no holding is modelled
        return float((pds.fillna(0).to_numpy() * table["Value"].to_numpy()).sum()) if pds.notna().any() else np.nan

    by_vol = {}
    for name in EQUITY_VOL_CHOICES:
        pds = pd.Series([res["merton"].get(name, {}).get("PD", np.nan) for res in results.values()])
        by_vol[name] = {"weighted_pd": weighted(pds), "expected_loss": expected_loss(pds)}
    return {"table": table, "coverage": float(table.loc[modelled, "Value"].sum() / table["Value"].sum()),
            "weighted_pd": weighted(table["PD"]), "expected_loss": expected_loss(table["PD"]), "by_vol": by_vol}


def credit_metrics(results: dict, view: dict, vol_choice: str, data_quality: float, sources: list, config: dict,
                   ltd_weight: float = C.DEFAULT_POINT_LTD_WEIGHT) -> dict:
    """TrustedMetric for weighted PD, expected loss, the weakest DD and the weakest Altman score."""
    modelled = [r for r in results.values() if r["merton"]]
    years = min((r["years"] for r in results.values()), default=0)
    stale = [r["Ticker"] for r in results.values()
             if np.isfinite(r["age_months"]) and r["age_months"] > config["stale_balance_sheet_months"]]
    extra = []
    if stale:
        extra.append(("Balance-sheet age", 1, f"older than {config['stale_balance_sheet_months']} months for {', '.join(stale)}"))
    if view["coverage"] < 1:
        extra.append(("Coverage", 1, f"{1 - view['coverage']:.0%} of the value is not modelled (financials or missing data)"))
    merton_assumptions = [f"default point = short-term + {ltd_weight:g} × long-term debt",
                          "horizon T", "risk-free rate (sidebar)"]

    def make(name, value, values_by_vol, assumptions, n_inputs, has_range=True, to_dd=None):
        finite = [v for v in values_by_vol if np.isfinite(v)]
        lo, hi = (min(finite), max(finite)) if has_range and finite else (np.nan, np.nan)
        if not has_range:
            width = None
        elif to_dd is not None:
            # PDs of strong firms are ~1e-20, where a relative width is meaningless: measure the range on the
            # distance-to-default scale the uncertainty actually lives on, DD = −Φ⁻¹(PD)
            dd_value, dd_lo, dd_hi = to_dd(value), to_dd(hi), to_dd(lo)
            width = (dd_hi - dd_lo) / abs(dd_value) if np.isfinite(dd_value) and dd_value and np.isfinite(dd_lo) else np.nan
        else:
            width = (hi - lo) / abs(value) if value and np.isfinite(lo) else np.nan
        letter, reasons = grade(width, None, None, data_quality, years, len(assumptions) / n_inputs,
                                sample_label="years of statements", sample_limits=(4, 3), extra=tuple(extra))
        return TrustedMetric(name, value, lo, hi, letter, reasons, sources, assumptions,
                             range_label=VOL_RANGE_LABEL if has_range else "range")

    modelled_value = float(view["table"].loc[view["table"]["PD"].notna(), "Value"].sum())

    def pd_to_dd(p):
        return float(-C.norm.ppf(p)) if np.isfinite(p) and 0 < p < 1 else (np.inf if p == 0 else np.nan)

    def loss_to_dd(loss):
        return pd_to_dd(loss / modelled_value) if modelled_value else np.nan

    metrics = {
        "weighted_pd": make("Weighted PD (risk-neutral, model-implied)", view["weighted_pd"],
                            [v["weighted_pd"] for v in view["by_vol"].values()], merton_assumptions, 6, to_dd=pd_to_dd),
        "expected_loss": make("Credit-implied expected loss (LGD 100%)", view["expected_loss"],
                              [v["expected_loss"] for v in view["by_vol"].values()], merton_assumptions, 6,
                              to_dd=loss_to_dd),
    }
    if modelled:
        weakest = min(modelled, key=lambda r: r["merton"].get(vol_choice, {}).get("DD", np.inf))
        metrics["weakest_dd"] = make(f"Lowest distance to default ({weakest['Ticker']})",
                                     weakest["merton"].get(vol_choice, {}).get("DD", np.nan),
                                     [m["DD"] for m in weakest["merton"].values()], merton_assumptions, 6)
    else:
        metrics["weakest_dd"] = make("Lowest distance to default", np.nan, [], merton_assumptions, 6)
    scored = view["table"].dropna(subset=["Altman Score"])
    if len(scored):
        low = scored.loc[scored["Altman Score"].idxmin()]
        metrics["weakest_altman"] = make(f"Lowest Altman {low['Altman Model']} ({low['Ticker']}, {low['Altman Zone']})",
                                         float(low["Altman Score"]), [], [], 5, has_range=False)
    else:
        metrics["weakest_altman"] = make("Lowest Altman score", np.nan, [], [], 5, has_range=False)
    return metrics


def compute_credit(ctx):
    """Merton, Altman, ratios, ratings and the portfolio credit view for every holding, as of the latest price."""
    config = C.load_config()
    ratings = ctx.disclosures["data"]["ratings"]
    results = {}
    for t, prices in ctx.price_frames.items():
        fund = ctx.fundamentals_by_ticker[t]
        returns = prices["Returns"].dropna()
        vols = equity_vols(returns, cached_fit_models(returns)["garch"])
        profile = fund["profile"]
        sector = (profile.get("sector") or {}).get("value") or ctx.fetched[t].get("sector")
        industry = (profile.get("industry") or {}).get("value") or ctx.fetched[t].get("industry")
        results[t] = cached_credit_holding(t, prices, fund, vols, sector, industry, ctx.currency, ratings,
                                           ctx.risk_free_pct / 100, ctx.ltd_weight, ctx.merton_horizon, config,
                                           ctx.prices_as_of)
    view = portfolio_view(results, ctx.positions, ctx.equity_vol_choice)
    debt_book, debt_errors = build_debt_book(ctx)
    # Where each DD sits in the reference universe, and the agency's historical default rate for its rating
    universe = load_dd_universe()
    dd_col = f"DD {ctx.equity_vol_choice}"
    reference = universe[dd_col] if dd_col in universe else pd.Series(dtype=float)
    rates = C.load_default_rates()
    agency = {t: C.rating_implied_pd(r["rating"]["rating"], r["rating"]["agency"], rates) for t, r in results.items()}
    view["table"]["DD Percentile"] = [C.dd_percentile(d, reference) for d in view["table"]["DD"]]
    view["table"]["Agency PD"] = [a["pd"] if a else np.nan for a in (agency[t] for t in view["table"]["Ticker"])]
    for t, res in results.items():
        res["agency_pd"] = agency[t]
    rated = view["table"]["Agency PD"].notna()
    value = view["table"]["Value"]
    view["agency_weighted_pd"] = float((view["table"].loc[rated, "Agency PD"] * value[rated]).sum() / value[rated].sum()) \
        if rated.any() else np.nan
    view["agency_coverage"] = float(value[rated].sum() / value.sum()) if value.sum() else 0.0
    view["dd_universe"] = {"size": int(reference.notna().sum()), "built_on": universe["built_on"].iloc[0] if len(universe) else None}
    data_q = float(min(q["score"] for q in ctx.quality.values())) if ctx.quality else np.nan
    sources = ["Yahoo Finance statements or your CSV overrides (Overview)", ctx.data_note]
    metrics = credit_metrics(results, view, ctx.equity_vol_choice, data_q, sources, config, ctx.ltd_weight)
    export(ctx, {"credit_results": results, "credit_view": view, "credit_metrics": metrics, "credit_config": config,
                 "debt_book": debt_book, "debt_errors": debt_errors})


def build_debt_book(ctx) -> tuple:
    """The sidebar's debt holdings, validated and priced; issuer tickers matched to equity holdings (with or
    without the .NS suffix) so the linked stress can tie each bond to its issuer's shares."""
    raw = getattr(ctx, "debt_input", None)
    if raw is None or raw.empty:
        return pd.DataFrame(), []
    clean, errors = D.validate(raw)
    if clean.empty:
        return pd.DataFrame(), errors
    held = {normalise_symbol(t): t for t in ctx.positions["Ticker"]}
    clean["Issuer Ticker"] = clean["Issuer Ticker"].map(lambda t: held.get(normalise_symbol(t), t) if t else "")
    return D.prepare(clean, ctx.risk_free_pct / 100, ctx.prices_as_of), errors
