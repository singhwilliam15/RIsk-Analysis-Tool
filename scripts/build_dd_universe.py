"""
Merton distance to default of a reference universe, so the Credit page can show where a holding's DD sits
(its percentile) instead of leading with risk-neutral PDs of 10^-30.

    python scripts/build_dd_universe.py

Universe: the symbols in the latest data/disclosures/pledges_*.csv (the app's presets, Jaiprakash Power and the
five most-pledged mid/small caps). Each DD uses the app's own code path (ui.credit_layer.analyse_holding) with the
app's defaults: 2-year prices, risk-free 6.5%, default point = short-term + 0.5 × long-term debt, T = 1 year.
Banks and NBFCs are listed without a DD (Merton does not apply).
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import credit as C  # noqa: E402
from data_fetcher import fetch_stock_data  # noqa: E402
from fundamentals import fetch_fundamentals  # noqa: E402
from ui.credit_layer import EQUITY_VOL_CHOICES, analyse_holding, equity_vols  # noqa: E402
from var_calculator import fit_models  # noqa: E402

OUT = ROOT / "data" / "credit" / "dd_universe.csv"
RISK_FREE, PERIOD = 0.065, "2y"


def main() -> int:
    pledges = sorted((ROOT / "data" / "disclosures").glob("pledges_*.csv"))
    if not pledges:
        print("Run scripts/fetch_disclosures.py first.")
        return 1
    symbols = list(dict.fromkeys(pd.read_csv(pledges[-1])["symbol"]))
    config = C.load_config()
    rows = []
    for sym in symbols:
        ticker = f"{sym}.NS"
        prices = fetch_stock_data(ticker, PERIOD)
        if not prices["success"]:
            rows.append({"symbol": sym, "note": f"no prices: {prices.get('error')}"})
            continue
        df = prices["df"]
        fund = fetch_fundamentals(ticker)
        returns = df["Returns"].dropna()
        vols = equity_vols(returns, fit_models(returns)["garch"])
        profile = fund["profile"]
        sector = (profile.get("sector") or {}).get("value") or prices.get("sector")
        industry = (profile.get("industry") or {}).get("value") or prices.get("industry")
        as_of = pd.Timestamp(df["Date"].iloc[-1])
        res = analyse_holding(ticker, df, fund, vols, sector, industry, "INR", None, RISK_FREE,
                              C.DEFAULT_POINT_LTD_WEIGHT, C.HORIZON_YEARS, config, as_of)
        row = {"symbol": sym, "as_of": f"{as_of:%Y-%m-%d}", "financial": res["financial"],
               "balance_sheet": f"{res['period_end']:%Y-%m-%d}" if res["period_end"] is not None else "",
               "note": "; ".join(res["notes"])}
        for name in EQUITY_VOL_CHOICES:
            row[f"DD {name}"] = res["merton"].get(name, {}).get("DD", np.nan)
        rows.append(row)
        print(sym, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k.startswith("DD")})
    table = pd.DataFrame(rows)
    table.insert(0, "built_on", f"{date.today():%Y-%m-%d}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
