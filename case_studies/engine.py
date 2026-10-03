"""
Phase 7 evidence: run the tool "as of" dates before real Indian collapses, using only data public by then, and
compare what it flagged with what happened, against a control group of large stable stocks.

Everything here reuses the pillars' own functions. A pillar whose inputs are not loaded (statements for credit,
disclosure files for events, prices for a delisted stock) is reported as not available; nothing is filled in.
Methods and flag rules: docs/case_studies.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import banks as B
import credit as C
import data_quality as DQ
import events as E
import fundamentals as F
import integration as I
import liquidity as L
from disclosures import load_disclosures
from stress import downside_beta
from var_calculator import calculate_all_var, calculate_historical_var, ewma_volatility

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "cases.json"
DATA_DIR = ROOT / "data"
LOOKBACK_DAYS = 500
CONFIDENCE = 0.95
PARAMS = {"bangia_k": L.BANGIA_K, "impact_y": L.IMPACT_Y, "r": 0.065, "T": 1.0, "initial_cover": E.INITIAL_COVER,
          "trigger_cover": E.TRIGGER_COVER}
PILLARS = ("market", "liquidity", "credit", "events", "stress")


def load_cases(path: Path = CASES_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def as_of_dates(event_date, months_before) -> list:
    """The analysis dates: the event date minus each number of months."""
    event = pd.Timestamp(event_date)
    return [(m, event - pd.DateOffset(months=m)) for m in months_before]


def window_before(df: pd.DataFrame, as_of, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """The last `days` rows dated on or before `as_of`: the only prices the analysis may see."""
    return df[df["Date"] <= pd.Timestamp(as_of)].tail(days).reset_index(drop=True)


def realised_loss(df: pd.DataFrame, as_of, event_date, days_after: int = 63) -> dict:
    """
    From the close on the as-of date to the lowest close between the as-of date and `days_after` trading days after
    the event: the loss a holder who did nothing would have seen at the worst point.
    """
    start = df[df["Date"] <= pd.Timestamp(as_of)]
    if start.empty:
        return {"loss": np.nan, "trough": None}
    p0 = float(start["Close"].iloc[-1])
    after = df[df["Date"] > pd.Timestamp(as_of)]
    event_idx = after.index[after["Date"] >= pd.Timestamp(event_date)]
    if len(event_idx) == 0:
        return {"loss": np.nan, "trough": None}
    end = min(event_idx[0] + days_after, df.index[-1])
    path = df.loc[after.index[0]:end]
    trough = path.loc[path["Close"].idxmin()]
    return {"loss": float(trough["Close"] / p0 - 1), "trough": trough["Date"]}


# ---------------------------------------------------------------
# Price-based flags (market, liquidity, integrated stress)
# ---------------------------------------------------------------

def price_flags(w: pd.DataFrame, mkt: pd.Series, position_value: float, flag_rules: dict, rules: dict,
                financial: bool = False) -> dict:
    """
    The flags that need prices only, from the window `w` (already cut at the as-of date) and the benchmark's returns
    up to that date. Used by evaluate() and by the wider test (universe.py), so both apply exactly the same rules:
    market = EWMA / 1-year volatility ≥ the ratio, or historical ES95 ≥ the floor; liquidity = lower-circuit days ≥
    the Elevated rule; stress = loss from a −20% market move (downside-β proxy) through the linked engine ≥ the floor.
    """
    returns = w["Returns"].dropna()
    dated = pd.Series(returns.to_numpy(), index=w.loc[returns.index, "Date"])
    es_hist = calculate_historical_var(returns, 1.0, CONFIDENCE, 1)["cvar_daily_pct"]
    ewma_now = ewma_volatility(returns)[1]
    vol_1y = float(returns.tail(252).std(ddof=1))
    out = {"es_hist": es_hist, "ewma_vol_ratio": ewma_now / vol_1y, "vol_1y": vol_1y, "available": {}, "flags": {}}
    out["available"]["market"] = True
    out["flags"]["market"] = bool(ewma_now / vol_1y >= flag_rules["market_ewma_vol_ratio_min"]
                                  or es_hist >= flag_rules["market_es_95_1d_min"])

    price = float(w["Close"].iloc[-1])
    adv60, _ = L.average_volume(w, 60)
    lock = L.circuit_lock(w)
    out.update(price=price, adv60=adv60, lock=lock, band=lock["band"], lower_circuit_days=lock["circuit_days"],
               longest_run=lock["longest_run"])
    out["available"]["liquidity"] = bool(w["Volume"].notna().any()) if "Volume" in w else False
    out["flags"]["liquidity"] = lock["circuit_days"] >= rules["elevated"]["lower_circuit_days_min"]

    # Integrated stress: a −20% market move through every pillar, with today's (as-of) inputs
    beta = downside_beta(dated, mkt)
    beta = beta if np.isfinite(beta) else 1.0
    spread = L.spread_profile(w)
    holding = pd.DataFrame([{"Ticker": "X", "Value": position_value, "Quantity": position_value / price, "Price": price,
                             "ADV": adv60, "Spread Mean": spread["mean"], "Spread Std": spread["std"], "Band": lock["band"],
                             "Freeze Days": lock["freeze_days"], "Equity": np.nan, "Default Point": np.nan, "PD": np.nan,
                             "Financial": financial, "Pledged Shares": np.nan, "Volume Ratio": 1.0}])
    scenario = {"name": "Market -20%", "kind": "custom", "market": -0.2,
                "holdings": {"X": {"return": max(-0.2 * beta, -1.0), "method": "β-proxy", "sigma": vol_1y * 2}}}
    linked = I.linked_stress(holding, scenario, PARAMS)["totals"]
    out.update(downside_beta=beta, stress_loss=linked["Total"] / position_value)
    out["available"]["stress"] = True
    out["flags"]["stress"] = bool(linked["Total"] / position_value >= flag_rules["stress_minus20_loss_min"])
    return out


# ---------------------------------------------------------------
# Naive baselines (review M1)
# ---------------------------------------------------------------

BASELINES = ("drawdown_30", "vol_top_decile", "below_200dma")
BASELINE_LABELS = {"drawdown_30": "Down ≥ 30% over 6 months", "vol_top_decile": "60-day vol in own 5-year top decile",
                   "below_200dma": "Below 200-day average", "any_baseline": "Any baseline"}
VOL_HISTORY_DAYS, VOL_HISTORY_MIN = 1260, 500


def baselines(df: pd.DataFrame, as_of) -> dict:
    """
    Three naive rules from prices up to `as_of` only, fixed before the test as the review states them:
    (a) close down ≥ 30% from 126 trading days earlier; (b) today's 60-day volatility at or above the 90th percentile
    of the stock's own rolling 60-day volatility over the last 1,260 days (at least 500 values needed); (c) close
    below its 200-day average. A rule without enough history is "not available", never "no warning".
    """
    h = df[df["Date"] <= pd.Timestamp(as_of)].reset_index(drop=True)
    close = h["Close"]
    out = {"available": {}, "flags": {}}
    out["available"]["drawdown_30"] = len(h) >= 127
    out["flags"]["drawdown_30"] = bool(len(h) >= 127 and close.iloc[-1] / close.iloc[-127] - 1 <= -0.30)
    vol = h["Returns"].rolling(60).std(ddof=1).tail(VOL_HISTORY_DAYS).dropna()
    out["available"]["vol_top_decile"] = len(vol) >= VOL_HISTORY_MIN
    out["flags"]["vol_top_decile"] = bool(len(vol) >= VOL_HISTORY_MIN and vol.iloc[-1] >= vol.quantile(0.90))
    out["available"]["below_200dma"] = len(h) >= 200
    out["flags"]["below_200dma"] = bool(len(h) >= 200 and close.iloc[-1] < close.tail(200).mean())
    return out


# ---------------------------------------------------------------
# One stock as of one date
# ---------------------------------------------------------------

def evaluate(df: pd.DataFrame, market: pd.Series, as_of, position_value: float, flag_rules: dict, rules: dict,
             financial: bool = False, fund: dict = None, disclosures: dict = None, symbol: str = None,
             with_grade: bool = False) -> dict:
    """
    Every pillar's view of one stock as of `as_of`, from data up to that date only. `market` is the benchmark's
    daily returns (dated); `fund` is the stock's statements (fundamentals format) if provided; `disclosures` is
    disclosures.load_disclosures(...)["data"] if provided. Returns figures, flags and which pillars were available.
    """
    w = window_before(df, as_of)
    out = {"as_of": pd.Timestamp(as_of), "days": len(w), "available": {p: False for p in PILLARS}, "flags": {},
           "baseline": baselines(df, as_of)}
    if len(w) < 250:
        out["note"] = f"only {len(w)} days of prices before the as-of date"
        return out
    returns = w["Returns"].dropna()
    mkt = market[market.index <= pd.Timestamp(as_of)]

    # Market, liquidity and integrated stress: the price-based flags (shared with the wider test in universe.py)
    pf = price_flags(w, mkt, position_value, flag_rules, rules, financial)
    vol_1y, lock, price, adv60 = pf.pop("vol_1y"), pf.pop("lock"), pf.pop("price"), pf.pop("adv60")
    for p in ("market", "liquidity", "stress"):
        out["available"][p] = pf["available"][p]
        out["flags"][p] = pf["flags"][p]
    out.update({k: v for k, v in pf.items() if k not in ("available", "flags")}, adv60=adv60)
    points = calculate_all_var(returns, 1.0, CONFIDENCE, 1, num_simulations=2000)
    out["es_garch"] = points["GARCH(1,1)-t"]["cvar_daily_pct"]
    out["data_quality"] = DQ.assess_holding(w)["score"]
    if with_grade:
        out["grade"] = market_grade(returns, points, out["data_quality"])
    out.update(days_to_liquidate=float(L.days_to_liquidate(position_value / price, adv60, L.DEFAULT_PARTICIPATION)),
               amihud=float(L.amihud(w).dropna().iloc[-1]) if L.amihud(w).notna().any() else np.nan)

    # Credit (needs statements; banks and NBFCs are not modelled)
    merton = None
    if fund is not None and not financial:
        period, values = C.latest_year(fund, as_of)
        D = C.default_point(values) if values else None
        shares = values.get("shares_issued") if values else None
        ratios, missing = C.altman_ratios(values, price * shares if shares else None) if values else ({}, ["statements"])
        z2 = C.altman_z2(ratios)
        if D and shares:
            merton = C.solve_merton(price * shares, vol_1y * np.sqrt(252), D, PARAMS["r"], PARAMS["T"])
        out.update(statement_year=period, dd=merton["DD"] if merton else np.nan, z2=z2[0], z2_zone=z2[1])
        out["available"]["credit"] = merton is not None or z2[0] is not None
        out["flags"]["credit"] = bool((merton is not None and merton["DD"] < rules["elevated"]["merton_dd_below"])
                                      or z2[1] == C.DISTRESS)
    out["credit_note"] = "bank / NBFC: not modelled" if financial else ("no statements loaded" if fund is None else "")
    if financial and symbol is not None:
        bank = B.assess(symbol, as_of)
        if bank["available"]:
            # Banks: RBI PCA bands (today's 2021 thresholds, applied as a benchmark) and the early-warning rules
            out.update(pca_band=bank["pca"]["worst"], pca_headroom=bank["pca"]["min_headroom"],
                       bank_warnings="; ".join(bank["warnings"]))
            out["available"]["credit"] = True
            out["flags"]["credit"] = bool(bank["pca"]["worst"] in ("RT1", "RT2", "RT3") or bank["warnings"])
            out["credit_note"] = "bank: RBI PCA thresholds of 2021 applied as a benchmark"

    # Events (needs disclosure files)
    if disclosures is not None and symbol is not None:
        signals = {"pledge": E.pledge_signal(disclosures["pledges"], symbol, as_of),
                   "surveillance": E.surveillance_signal(disclosures["surveillance"], symbol, as_of),
                   "fo_ban": E.fo_ban_signal(disclosures["fo_ban"], symbol, as_of),
                   "rating": E.rating_signal(disclosures["ratings"], symbol, as_of),
                   "auditor": E.auditor_signal(disclosures["auditor_events"], symbol, as_of)}
        disclosed = [k for k, v in signals.items() if v is not None]
        signals["circuit"] = {"lower_circuit_days": lock["circuit_days"], "longest_run": lock["longest_run"]}
        if merton is not None:
            signals["merton"] = {"dd": merton["DD"], "dd_6m_ago": np.nan, "fall_6m": np.nan}
        tier = E.classify(signals, rules)
        out.update(tier=tier["tier"], tier_reasons="; ".join(t for _, t in tier["fired"]), disclosed=disclosed)
        out["available"]["events"] = bool(disclosed)
        out["flags"]["events"] = bool(disclosed) and tier["tier"] != E.LOW

    out["warning"] = any(out["flags"].values())
    return out


def market_grade(returns: pd.Series, points: dict, data_quality: float) -> dict:
    """The Trust layer's grade for the recommended model, from an out-of-sample backtest within the window."""
    import trust as T
    from var_calculator import (BACKTEST_WINDOW, MONTE_CARLO_BACKTEST_NAME, backtest_all_methods, fit_models,
                                recommend_model, rolling_forecasts)
    rolling = rolling_forecasts(returns, CONFIDENCE, window=BACKTEST_WINDOW, num_simulations=1000)
    table = backtest_all_methods(returns, rolling["var"], CONFIDENCE, rolling["es"], rolling["sigma"])
    rec = recommend_model(table, exclude=[MONTE_CARLO_BACKTEST_NAME])
    names = list(points)

    def point_name(name):
        return names[-1] if name == MONTE_CARLO_BACKTEST_NAME else name

    fits = fit_models(returns)
    ranges = T.bootstrap_ranges(returns, CONFIDENCE, fits, 2000, n_boot=200, n_boot_refit=40, n_draws=300)
    ranged = T.scale_ranges(ranges, points, 1.0)
    trusted = T.market_trust(points, ranged, table, rec["model"], point_name, len(returns), data_quality, 1, [])
    headline = trusted["headline_model"]
    es = trusted["metrics"][headline]["ES"]
    return {"model": headline, "status": rec["status"], "es": es.value, "low": es.low, "high": es.high, "grade": es.grade}


# ---------------------------------------------------------------
# Tally
# ---------------------------------------------------------------

def _rule_columns(results: pd.DataFrame, rule: str) -> tuple:
    """(available, fired) boolean columns for a pillar, "any", a baseline or "any_baseline"."""
    if rule == "any":
        return results["days"] >= 250, results["warning"].fillna(False).astype(bool)
    if rule == "any_baseline":
        avail = pd.concat([results[f"available_{b}"].fillna(False).astype(bool) for b in BASELINES], axis=1).any(axis=1)
        fired = pd.concat([results[f"flag_{b}"].fillna(False).astype(bool) for b in BASELINES], axis=1).any(axis=1)
        return avail, fired
    return results[f"available_{rule}"].fillna(False).astype(bool), results[f"flag_{rule}"].fillna(False).astype(bool)


RULES = PILLARS + ("any",) + BASELINES + ("any_baseline",)


def tally(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each pillar, any warning, each naive baseline and any baseline: hits = case dates that warned, misses = case
    dates that did not, false positives = control dates that warned; rates only over dates where the rule was available.
    """
    rows = []
    cases = results["group"] == "case"
    for rule in RULES:
        if rule in BASELINES + ("any_baseline",) and f"available_{BASELINES[0]}" not in results:
            continue
        avail, fired = _rule_columns(results, rule)
        n_case, n_ctrl = int((avail & cases).sum()), int((avail & ~cases).sum())
        hits, fps = int((avail & cases & fired).sum()), int((avail & ~cases & fired).sum())
        rows.append({"Pillar": rule, "Case dates": n_case, "Hits": hits, "Misses": n_case - hits,
                     "Hit rate": hits / n_case if n_case else np.nan, "Control dates": n_ctrl, "False positives": fps,
                     "False-positive rate": fps / n_ctrl if n_ctrl else np.nan})
    return pd.DataFrame(rows)


def lead_times(results: pd.DataFrame) -> pd.DataFrame:
    """
    Lead time per rule: for each case, the earliest as-of date (most months before the event) on which the rule
    warned; the mean over the cases it warned on at all, and how many cases it never warned on.
    """
    cases = results[(results["group"] == "case") & (results["days"] >= 250)]  # cases with prices only
    names = list(dict.fromkeys(cases["case"]))
    rows = []
    for rule in RULES:
        if rule in BASELINES + ("any_baseline",) and f"available_{BASELINES[0]}" not in results:
            continue
        avail, fired = _rule_columns(cases, rule)
        earliest = {}
        for name in names:
            sel = (cases["case"] == name) & avail & fired
            earliest[name] = int(cases.loc[sel, "months_before"].max()) if sel.any() else None
        warned = [m for m in earliest.values() if m is not None]
        rows.append({"Rule": rule, "Cases warned": len(warned), "Cases": len(names),
                     "Mean earliest warning (months before)": float(np.mean(warned)) if warned else np.nan,
                     **{name: earliest[name] for name in names}})
    return pd.DataFrame(rows)


def jump_frequencies(frames: dict, results: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    """
    Observed daily frequency of one-day falls of 10% and 20% or more over the `horizon` trading days after each
    as-of date, for dates with and without a warning: the evidence for (or against) the Phase 5 jump defaults.
    """
    rows = []
    for r in results.to_dict("records"):
        df = frames.get(r["ticker"])
        if df is None or r["days"] < 250:
            continue
        after = df[df["Date"] > r["as_of"]].head(horizon)["Returns"].dropna()
        rows.append({"warning": bool(r["warning"]), "group": r["group"], "days": len(after),
                     "falls_10": int((after <= -0.10).sum()), "falls_20": int((after <= -0.20).sum())})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    out = table.groupby(["group", "warning"])[["days", "falls_10", "falls_20"]].sum().reset_index()
    out["p_fall_10"] = out["falls_10"] / out["days"]
    out["p_fall_20"] = out["falls_20"] / out["days"]
    return out


def load_case_fundamentals(ticker: str):
    """Statements uploaded for a case (case_studies/data/fundamentals/<TICKER>.csv, the app's override template)."""
    path = DATA_DIR / "fundamentals" / f"{ticker.upper()}.csv"
    if not path.exists():
        return None
    upload = pd.read_csv(path, dtype=str)
    fund, errors = F.apply_overrides(F.build_fundamentals(ticker.upper(), {}, {}), upload)
    if errors:
        raise ValueError(f"{path.name}: " + "; ".join(errors))
    return fund


def load_case_disclosures():
    """Disclosure files for the case studies (case_studies/data/disclosures/ with its own manifest), or None."""
    directory = DATA_DIR / "disclosures"
    result = load_disclosures(directory)
    loaded = any(not frame.empty for frame in result["data"].values())
    return (result["data"] if loaded else None), result["issues"]
