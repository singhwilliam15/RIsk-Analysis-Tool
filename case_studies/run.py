"""
Run the Phase 7 case studies and the control group, and write case_studies/results/ and docs/images/case_studies.png.

    python case_studies/run.py

Prices come from Yahoo Finance (or case_studies/data/prices/<TICKER>.csv for delisted stocks); statements from
case_studies/data/fundamentals/; disclosures from case_studies/data/disclosures/ (with its manifest). A case
without prices, and a pillar without its inputs, is reported as not available.
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import banks as B  # noqa: E402
import data_fetcher  # noqa: E402
from case_studies import engine as K  # noqa: E402
from data_fetcher import _add_return_columns  # noqa: E402
from events import load_rules  # noqa: E402
from stress import price_series  # noqa: E402

RESULTS = K.ROOT / "results"
CHART = K.ROOT.parent / "docs" / "images" / "case_studies.png"


def prices_for(ticker: str, price_file: str = None):
    if price_file and Path(price_file).exists():
        raw = pd.read_csv(price_file, parse_dates=["Date"]).sort_values("Date")
        return _add_return_columns(raw[[c for c in ("Date", "Open", "High", "Low", "Close", "Volume") if c in raw]]), "uploaded file"
    res = data_fetcher.fetch_stock_data(ticker, "max")
    return (res["df"], res["data_source"]) if res["success"] else (None, res["error"])


def flatten(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in ("flags", "available", "grade")}
    for p in K.PILLARS:
        out[f"available_{p}"] = r.get("available", {}).get(p, False)
        out[f"flag_{p}"] = r.get("flags", {}).get(p, False)
    for k, v in (r.get("grade") or {}).items():
        out[f"grade_{k}"] = v
    return out


def main() -> int:
    config = K.load_cases()
    rules = load_rules()
    market_df = data_fetcher.fetch_stock_data("^NSEI", "max")["df"]
    market = price_series(market_df).pct_change().dropna()
    disclosures, issues = K.load_case_disclosures()
    for issue in issues:
        print("disclosures:", issue)
    frames, rows, missing = {}, [], []
    started = time.time()
    for case in [c for c in config["cases"] if c["enabled"]]:
        df, source = prices_for(case["ticker"], case.get("price_file"))
        if df is None:
            missing.append({"Case": case["name"], "Ticker": case["ticker"], "Reason": f"no prices ({source})"})
            for months, as_of in K.as_of_dates(case["event_date"], config["as_of_months_before"]):
                rows.append({"group": "case", "case": case["name"], "ticker": case["ticker"], "months_before": months,
                             "as_of": as_of, "days": 0, "note": "prices not available", "warning": None})
            continue
        frames[case["ticker"]] = df
        fund = K.load_case_fundamentals(case["ticker"])
        symbol = case["ticker"].split(".")[0]
        for months, as_of in K.as_of_dates(case["event_date"], config["as_of_months_before"]):
            r = K.evaluate(df, market, as_of, config["position_value"], config["flag_rules"], rules, case["financial"], fund,
                           disclosures, symbol, with_grade=True)
            r.update(group="case", case=case["name"], ticker=case["ticker"], months_before=months, prices=source,
                     **{f"realised_{k}": v for k, v in K.realised_loss(df, as_of, case["event_date"], config["realised_window_days"]).items()})
            rows.append(flatten(r))
            print(f"{case['name']:20s} {months:2d}m  warning={r.get('warning')}  ({time.time() - started:.0f}s)")
        # The control group, at the same as-of dates
        for ticker in config["control_group"]:
            if ticker not in frames:
                cdf, _ = prices_for(ticker)
                if cdf is None:
                    continue
                frames[ticker] = cdf
            for months, as_of in K.as_of_dates(case["event_date"], config["as_of_months_before"]):
                symbol = ticker.split(".")[0]
                is_bank = (B.BANKS_DIR / f"{symbol}.csv").exists()  # a control bank is assessed as a bank
                r = K.evaluate(frames[ticker], market, as_of, config["position_value"], config["flag_rules"], rules,
                               is_bank, K.load_case_fundamentals(ticker), disclosures, symbol)
                r.update(group="control", case=case["name"], ticker=ticker, months_before=months,
                         **{f"realised_{k}": v for k, v in K.realised_loss(frames[ticker], as_of, case["event_date"],
                                                                            config["realised_window_days"]).items()})
                rows.append(flatten(r))
    results = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "results.csv", index=False)
    tally = K.tally(results)
    tally.to_csv(RESULTS / "tally.csv", index=False)
    jumps = K.jump_frequencies(frames, results)
    jumps.to_csv(RESULTS / "jump_frequencies.csv", index=False)
    pd.DataFrame(missing).to_csv(RESULTS / "missing.csv", index=False)
    chart(results)
    print(tally.to_string(index=False))
    print(jumps.to_string(index=False))
    print(f"done in {time.time() - started:.0f}s")
    return 0


def chart(results: pd.DataFrame) -> None:
    """Realised loss after each as-of date, coloured by whether the tool warned; one panel per case."""
    cases = results[(results["group"] == "case") & (results["days"] >= 250)]
    names = list(dict.fromkeys(cases["case"]))
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.0), dpi=150, sharey=True)
    axes = [axes] if len(names) == 1 else axes
    for ax, name in zip(axes, names):
        part = cases[cases["case"] == name].sort_values("months_before", ascending=False)
        colors = ["#E53E3E" if w else "#A0AEC0" for w in part["warning"]]
        ax.bar([f"{m}m" for m in part["months_before"]], part["realised_loss"] * 100, color=colors)
        ax.set_title(name, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Realised loss to trough, %", fontsize=8)
    fig.suptitle("Months before the event: red = the tool warned, grey = no warning", fontsize=9)
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
