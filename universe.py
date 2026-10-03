"""
The wider test (review M1/M2): a point-in-time universe of every NSE equity in the daily bhavcopies, including stocks
that were later delisted, so it is not limited to today's survivors.

- Data: scripts/fetch_bhavcopy.py downloads the files; this module parses them, chains renamed symbols and
  back-adjusts prices for bonuses and splits (NSE's previous close is not adjusted).
- Test: on the last trading day of each month, the tool's price-based flags and the naive baselines, against a
  ≥ 50% drawdown over the next 252 trading days (docs/case_studies.md, "Wider test").
- Jumps: base rates of one-day stock-specific falls, for the jump overlay (docs/methodology.md §12.4).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
PANEL_PATH = CACHE / "panel.parquet"
SERIES = ("EQ", "BE", "BZ")  # rolling settlement, trade-for-trade (surveillance stages), non-compliant companies
TRADE_FOR_TRADE = ("BE", "BZ")
PANEL_COLUMNS = ["date", "symbol", "series", "isin", "open", "high", "low", "close", "prevclose", "volume", "value"]

# Old format (to July 2024) and UDiFF format (from 8 July 2024) → panel columns
OLD = {"SYMBOL": "symbol", "SERIES": "series", "ISIN": "isin", "OPEN": "open", "HIGH": "high", "LOW": "low",
       "CLOSE": "close", "PREVCLOSE": "prevclose", "TOTTRDQTY": "volume", "TOTTRDVAL": "value"}
UDIFF = {"TckrSymb": "symbol", "SctySrs": "series", "ISIN": "isin", "OpnPric": "open", "HghPric": "high",
         "LwPric": "low", "ClsPric": "close", "PrvsClsgPric": "prevclose", "TtlTradgVol": "volume", "TtlTrfVal": "value"}


# ---------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------

def normalise_bhavcopy(raw: pd.DataFrame, day) -> pd.DataFrame:
    """One day's bhavcopy (either format) → panel rows for company shares (ISIN INE…) in series EQ, BE and BZ."""
    raw = raw.rename(columns=lambda c: str(c).strip())
    mapping = UDIFF if "TckrSymb" in raw.columns else OLD
    out = raw[list(mapping)].rename(columns=mapping)
    out["symbol"] = out["symbol"].str.strip()
    out["series"] = out["series"].fillna("").str.strip()
    # Company shares only: ISINs of companies start INE; ETFs and fund units (INF) also trade in series EQ
    out = out[out["series"].isin(SERIES) & out["isin"].fillna("").str.strip().str.startswith("INE")].copy()
    for col in ("open", "high", "low", "close", "prevclose", "volume", "value"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.Timestamp(day)
    return out[PANEL_COLUMNS].reset_index(drop=True)


BONUS = re.compile(r"bonus\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.I)
SPLIT = re.compile(r"(?:split|sub-?division).*?r[se]\.?\s*(\d+(?:\.\d+)?).*?to\s*r[se]\.?\s*(\d+(?:\.\d+)?)", re.I)


def action_factor(subject: str):
    """
    Price multiplier of one corporate-action subject, or None if it changes no share count. "Bonus a:b" (a new
    shares for every b held) multiplies the price by b/(a+b); "Face Value Split From Rs F1 To Rs F2" by F2/F1.
    Both in one subject (e.g. "Bonus 1:5/Face Value Split ... From Rs 10 ... To Rs 2") multiply.
    """
    factor = 1.0
    for part in re.split(r"/(?!-)", str(subject)):  # "Rs 10/- Per Share" is not a separator
        b = BONUS.search(part)
        s = SPLIT.search(part)
        if b:
            new, held = float(b.group(1)), float(b.group(2))
            if new > 0 and held > 0:
                factor *= held / (new + held)
        elif s:
            old, new = float(s.group(1)), float(s.group(2))
            if old > 0 and new > 0:
                factor *= new / old
    return factor if factor != 1.0 else None


def parse_corporate_actions(years: list) -> pd.DataFrame:
    """
    NSE corporate-action JSON (one list per year) → one row per (symbol, ex-date) with the price multiplier.
    Repeated records of the same action (e.g. "Purpose Revised") count once.
    """
    rows = []
    for records in years:
        for rec in records:
            f = action_factor(rec.get("subject", ""))
            if f is None or rec.get("series") not in SERIES:
                continue
            rows.append({"symbol": rec["symbol"].strip(), "ex_date": pd.to_datetime(rec["exDate"], format="%d-%b-%Y"),
                         "factor": round(f, 10), "subject": rec["subject"].strip()})
    frame = pd.DataFrame(rows, columns=["symbol", "ex_date", "factor", "subject"])
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(["symbol", "ex_date", "factor"])
    return (frame.groupby(["symbol", "ex_date"], as_index=False)
            .agg(factor=("factor", "prod"), subject=("subject", " | ".join)))


def parse_symbol_changes(raw: pd.DataFrame) -> list:
    """NSE symbolchange.csv (company, old, new, date) → [(old, new, date)] in date order."""
    raw = raw.iloc[:, :4].copy()
    raw.columns = ["company", "old", "new", "date"]
    raw["date"] = pd.to_datetime(raw["date"].str.strip(), format="%d-%b-%Y", errors="coerce")
    raw = raw.dropna(subset=["date"])
    return [(o.strip(), n.strip(), d) for o, n, d in raw[["old", "new", "date"]].sort_values("date").itertuples(index=False)]


def chain_symbols(panel: pd.DataFrame, changes: list) -> pd.DataFrame:
    """Rows traded under an old symbol before its change date are relabelled with the later symbol (chains followed)."""
    if not changes:
        return panel
    panel = panel.copy()
    for old, new, when in changes:  # in date order, so A→B then B→C ends at C
        mask = (panel["symbol"] == old) & (panel["date"] < when)
        panel.loc[mask, "symbol"] = new
    return panel


def adjust(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """
    Back-adjust open/high/low/close for bonuses and splits: every price before an ex-date is multiplied by that
    action's factor, so a 1:1 bonus is a ~0% day, not −50%. Demergers carry no ratio and are not adjusted (they
    show as "unexplained" moves; scripts/run_universe.py reports a run without the symbols that have any). Adds `adj_factor` and the adjusted daily return `ret`
    (from consecutive trading days of the symbol).
    """
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["adj_factor"] = 1.0
    if actions is not None and not actions.empty:
        for symbol, acts in actions.groupby("symbol"):
            idx = panel.index[panel["symbol"] == symbol]
            if len(idx) == 0:
                continue
            dates = panel.loc[idx, "date"].to_numpy()
            factor = np.ones(len(idx))
            for ex, f in zip(acts["ex_date"].to_numpy(), acts["factor"].to_numpy()):
                factor[dates < ex] *= f
            panel.loc[idx, "adj_factor"] = factor
    for col in ("open", "high", "low", "close"):
        panel[col] = panel[col] * panel["adj_factor"]
    panel["ret"] = panel.groupby("symbol")["close"].pct_change()
    return panel


# ---------------------------------------------------------------
# The test: flags and baselines on each month-end, against the next 12 months
# ---------------------------------------------------------------

HORIZON_DAYS = 252           # outcome window, market trading days
DRAWDOWN_EVENT = -0.50       # the outcome: a fall of 50% or more from the as-of close within the window
ACTIVE_WITHIN_DAYS = 5       # a stock counts on an as-of date if it traded in the last 5 market days
LIQUID_VALUE = 1e7           # liquid subset: median 60-day traded value ≥ ₹1 crore
JUMP_THRESHOLDS = (-0.10, -0.20)
MARKET_CRASH = -0.05         # a fall on a day the market fell more than this is market-wide, not a stock jump
JUMP_WINDOW = 63             # forward window for the jump base rates (as in case_studies.jump_frequencies)
POSITION_VALUE = 1e7


def stock_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """One symbol's panel rows → the frame the pillars expect (Date, OHLC, Volume, Returns), adjusted prices."""
    from data_fetcher import _add_return_columns
    frame = rows.sort_values("date").rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low",
                                                     "close": "Close", "volume": "Volume"})
    return _add_return_columns(frame[["Date", "Open", "High", "Low", "Close", "Volume", "value", "series"]].reset_index(drop=True))


def month_ends(market_dates: pd.DatetimeIndex, start: str, end: str) -> list:
    dates = pd.Series(market_dates[(market_dates >= pd.Timestamp(start)) & (market_dates <= pd.Timestamp(end))])
    return list(dates.groupby(dates.dt.to_period("M")).max())


def jump_days(returns: np.ndarray, market: np.ndarray, threshold: float) -> np.ndarray:
    """Stock-specific jump days: a fall of at least |threshold| on a day the market did not crash."""
    return (returns <= threshold) & ~(market <= MARKET_CRASH)


def evaluate_symbol(symbol: str, frame: pd.DataFrame, market: pd.Series, as_of_dates: list, flag_rules: dict,
                    rules: dict, panel_end) -> list:
    """Every month-end row for one stock: flags, baselines, the 12-month outcome and the jump counts."""
    from case_studies import engine as K
    rows = []
    dates = frame["Date"]
    mkt_dates = market.index
    last_trade = dates.iloc[-1]
    for d in as_of_dates:
        upto = frame[dates <= d]
        if len(upto) < 250:
            continue
        recent = mkt_dates[mkt_dates <= d][-ACTIVE_WITHIN_DAYS:]
        if upto["Date"].iloc[-1] < recent[0]:
            continue  # not trading around the as-of date
        after_mkt = mkt_dates[mkt_dates > d]
        if len(after_mkt) < HORIZON_DAYS:
            continue  # the outcome window is not complete
        w = upto.tail(K.LOOKBACK_DAYS).reset_index(drop=True)
        pf = K.price_flags(w, market[market.index <= d], POSITION_VALUE, flag_rules, rules)
        base = K.baselines(frame, d)
        horizon_end = after_mkt[HORIZON_DAYS - 1]
        fut = frame[(dates > d) & (dates <= horizon_end)]
        p0 = float(w["Close"].iloc[-1])
        worst = float(fut["Close"].min() / p0 - 1) if len(fut) else np.nan
        gone = bool(last_trade < horizon_end and last_trade < pd.Timestamp(panel_end) - pd.Timedelta(days=30))
        row = {"symbol": symbol, "as_of": d, "series": w["series"].iloc[-1],
               "liquid": bool(w["value"].tail(60).median() >= LIQUID_VALUE),
               "worst_12m": worst, "future_days": len(fut), "disappeared": gone,
               "event_observed": bool(np.isfinite(worst) and worst <= DRAWDOWN_EVENT),
               "event_upper": bool((np.isfinite(worst) and worst <= DRAWDOWN_EVENT) or gone),
               "es_hist": pf["es_hist"], "ewma_vol_ratio": pf["ewma_vol_ratio"],
               "lower_circuit_days": pf["lower_circuit_days"], "stress_loss": pf["stress_loss"]}
        for p in ("market", "liquidity", "stress"):
            row[f"flag_{p}"] = pf["flags"][p]
        row["flag_tool"] = any(pf["flags"].values())
        for b in K.BASELINES:
            row[f"available_{b}"] = base["available"][b]
            row[f"flag_{b}"] = base["flags"][b]
        row["flag_any_baseline"] = any(base["flags"].values())
        # Jump counts: the next 63 market days, and the trailing window the base model is fitted on
        fwd = frame[(dates > d) & (dates <= after_mkt[JUMP_WINDOW - 1])]
        m_fwd = market.reindex(fwd["Date"]).to_numpy()
        m_trail = market.reindex(w["Date"]).to_numpy()
        row["fwd_days"], row["trail_days"] = len(fwd), int(w["Returns"].notna().sum())
        for thr in JUMP_THRESHOLDS:
            tag = f"{int(round(-thr * 100))}"
            f_mask = jump_days(fwd["Returns"].to_numpy(), m_fwd, thr)
            row[f"fwd_jumps_{tag}"] = int(f_mask.sum())
            row[f"fwd_jump_size_sum_{tag}"] = float(fwd["Returns"].to_numpy()[f_mask].sum())
            row[f"trail_jumps_{tag}"] = int(jump_days(w["Returns"].to_numpy(), m_trail, thr).sum())
        rows.append(row)
    return rows


def worker(task):
    symbol, frame, market, as_of_dates, flag_rules, rules, panel_end = task
    try:
        return evaluate_symbol(symbol, frame, market, as_of_dates, flag_rules, rules, panel_end)
    except Exception as exc:  # one bad series must not stop the test; it is reported
        return [{"symbol": symbol, "error": repr(exc)}]


# ---------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------

UNIVERSE_RULES = ("flag_market", "flag_liquidity", "flag_stress", "flag_tool", "flag_drawdown_30", "flag_vol_top_decile",
                  "flag_below_200dma", "flag_any_baseline")


def rule_metrics(results: pd.DataFrame, outcome: str = "event_observed") -> pd.DataFrame:
    """Precision, recall, flag rate, base rate and lift of every rule against `outcome`."""
    data = results.dropna(subset=["worst_12m"]) if outcome == "event_observed" else results
    y = data[outcome].astype(bool)
    rows = []
    for rule in UNIVERSE_RULES:
        flag = data[rule].fillna(False).astype(bool)
        tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        base = float(y.mean()) if len(y) else np.nan
        rows.append({"Rule": rule.replace("flag_", ""), "Dates": len(data), "Flagged": int(flag.sum()),
                     "Flag rate": float(flag.mean()) if len(flag) else np.nan, "Events": int(y.sum()), "Base rate": base,
                     "Precision": precision, "Recall": recall, "Lift": precision / base if base else np.nan,
                     "F1": 2 * precision * recall / (precision + recall) if precision and recall else np.nan})
    return pd.DataFrame(rows)


def yearly_metrics(results: pd.DataFrame, outcome: str = "event_observed") -> pd.DataFrame:
    out = []
    for year, part in results.groupby(results["as_of"].dt.year):
        m = rule_metrics(part, outcome)
        m.insert(0, "Year", year)
        out.append(m)
    return pd.concat(out, ignore_index=True)


def jump_base_rates(results: pd.DataFrame, n_boot: int = 1000, seed: int = 7) -> pd.DataFrame:
    """
    Jump calibration (methodology §12.4). For each proxy group of as-of dates: the forward daily rate of
    stock-specific falls (63 market days after the date), the stock's own trailing rate (the window the base model is
    fitted on), the excess p = forward − trailing, the mean fall size J, and a 90% range for p from resampling stocks
    (a cluster bootstrap, since the monthly windows of one stock overlap). The groups are fixed in advance: all
    dates; Elevated proxy = the liquidity flag (≥ 3 lower-circuit days in the window, the tool's Elevated circuit
    rule); High proxy = trading in series BE or BZ (trade-for-trade surveillance or non-compliance) on the date.
    """
    groups = {"All stocks": pd.Series(True, index=results.index),
              "Elevated proxy: ≥ 3 lower-circuit days": results["flag_liquidity"].astype(bool),
              "High proxy: series BE/BZ": results["series"].isin(TRADE_FOR_TRADE)}
    rng = np.random.default_rng(seed)
    rows = []
    for name, mask in groups.items():
        part = results[mask]
        for thr in JUMP_THRESHOLDS:
            tag = f"{int(round(-thr * 100))}"
            cols = [f"fwd_jumps_{tag}", "fwd_days", f"trail_jumps_{tag}", "trail_days", f"fwd_jump_size_sum_{tag}"]
            by = part.groupby("symbol")[cols].sum()
            if by.empty or by["fwd_days"].sum() == 0:
                continue
            fwd = by[cols[0]].sum() / by["fwd_days"].sum()
            trail = by[cols[2]].sum() / by["trail_days"].sum()
            values = by.to_numpy(dtype=float)
            boots = np.empty(n_boot)
            for i in range(n_boot):
                s = values[rng.integers(0, len(values), len(values))].sum(axis=0)
                boots[i] = s[0] / s[1] - s[2] / s[3] if s[1] and s[3] else np.nan
            jumps = int(by[cols[0]].sum())
            rows.append({"Group": name, "Fall": f"≥ {tag}%", "As-of dates": int(len(part)), "Stocks": int(len(by)),
                         "Forward days": int(by["fwd_days"].sum()), "Jumps": jumps,
                         "Forward rate a day": fwd, "Own trailing rate a day": trail, "Excess p a day": fwd - trail,
                         "p low (5%)": float(np.nanpercentile(boots, 5)), "p high (95%)": float(np.nanpercentile(boots, 95)),
                         "Mean fall J": by[cols[4]].sum() / jumps if jumps else np.nan})
    return pd.DataFrame(rows)
