"""
The wider test (docs/case_studies.md, "Wider test"): every NSE equity in the bhavcopies on every month-end from
Jan 2016 to Dec 2024, the tool's price-based flags and the naive baselines against a ≥ 50% drawdown over the next
12 months, plus the jump base rates for the event overlay.

    python scripts/fetch_bhavcopy.py      # once: downloads and builds data/cache/panel.parquet
    python scripts/run_universe.py        # writes case_studies/results/universe_*.csv

The benchmark is the Nifty 50 index from Yahoo Finance (an index has no survivorship problem). The design was fixed
before the run: thresholds, outcome, universe filters and proxy groups are in universe.py.
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data_fetcher  # noqa: E402
import universe as U  # noqa: E402
from case_studies import engine as K  # noqa: E402
from events import load_rules  # noqa: E402
from stress import price_series  # noqa: E402

RESULTS = K.ROOT / "results"
START, END = "2016-01-01", "2024-12-31"
UNEXPLAINED_MOVE = 0.40  # a day beyond ±40% that no bonus/split explains (NSE bands are at most 20%)


def main() -> int:
    started = time.time()
    panel = pd.read_parquet(U.PANEL_PATH)
    nifty = data_fetcher.fetch_stock_data("^NSEI", "max")
    market = price_series(nifty["df"]).pct_change().dropna()
    market = market[market.index >= panel["date"].min()]
    as_of_dates = U.month_ends(market.index, START, END)
    config = K.load_cases()
    rules = load_rules()
    panel_end = panel["date"].max()

    odd = panel.loc[panel["ret"].abs() > UNEXPLAINED_MOVE, ["symbol", "date", "ret"]]
    tasks = []
    for symbol, rows in panel.groupby("symbol", sort=True):
        if len(rows) >= 250:
            tasks.append((symbol, U.stock_frame(rows), market, as_of_dates, config["flag_rules"], rules, panel_end))
    print(f"{len(tasks):,} symbols with 250+ days; {len(as_of_dates)} month-ends; "
          f"{len(odd):,} unexplained moves beyond ±40% in {odd['symbol'].nunique()} symbols", flush=True)

    rows, errors = [], []
    with Pool(7) as pool:
        for i, out in enumerate(pool.imap_unordered(U.worker, tasks, chunksize=4), 1):
            for r in out:
                (errors if "error" in r else rows).append(r)
            if i % 200 == 0:
                print(f"  {i:,}/{len(tasks):,} symbols, {len(rows):,} rows ({time.time() - started:.0f}s)", flush=True)
    results = pd.DataFrame(rows)
    results.to_parquet(U.CACHE / "universe_results.parquet", index=False)

    tables = []
    clean = ~results["symbol"].isin(set(odd["symbol"]))
    for subset, mask in (("all", pd.Series(True, index=results.index)), ("liquid", results["liquid"]),
                         ("all, excluding unexplained ±40% moves", clean)):
        for outcome in ("event_observed", "event_upper"):
            m = U.rule_metrics(results[mask], outcome)
            m.insert(0, "Outcome", {"event_observed": "observed path", "event_upper": "disappearance counted as event"}[outcome])
            m.insert(0, "Universe", subset)
            tables.append(m)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(RESULTS / "universe_metrics.csv", index=False)
    U.yearly_metrics(results).to_csv(RESULTS / "universe_yearly.csv", index=False)
    jumps = U.jump_base_rates(results)
    jumps.to_csv(RESULTS / "universe_jumps.csv", index=False)
    meta = {"as_of_start": str(as_of_dates[0].date()), "as_of_end": str(as_of_dates[-1].date()),
            "month_ends": len(as_of_dates), "symbols_evaluated": int(results["symbol"].nunique()),
            "stock_dates": len(results), "liquid_stock_dates": int(results["liquid"].sum()),
            "events_observed": int(results["event_observed"].sum()),
            "dates_without_future_prices": int(results["worst_12m"].isna().sum()),
            "disappeared_stock_dates": int(results["disappeared"].sum()),
            "symbols_that_disappeared": int(results.loc[results["disappeared"], "symbol"].nunique()),
            "unexplained_moves": len(odd), "symbols_with_unexplained_moves": int(odd["symbol"].nunique()),
            "errors": len(errors), "error_examples": errors[:5], "panel_start": str(panel["date"].min().date()),
            "panel_end": str(panel_end.date()), "panel_symbols": int(panel["symbol"].nunique()),
            "runtime_s": round(time.time() - started)}
    (RESULTS / "universe_meta.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(jumps.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
