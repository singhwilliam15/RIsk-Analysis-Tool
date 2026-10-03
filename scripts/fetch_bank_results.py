"""
Gross NPA %, net NPA % and ROA of banks from NSE results XBRL, into data/banks/<SYMBOL>.csv.

    python scripts/fetch_bank_results.py HDFCBANK:bank:dsib YESBANK:bank

Each argument is SYMBOL:entity_type[:dsib]. Quarterly and annual standalone results are read; the filing date is
NSE's broadcast date. Rows this script wrote before are replaced; rows transcribed by hand from annual reports
(source not starting with the results URL) are kept. Capital ratios are not in the results XBRL and come from the
annual reports (see data/banks/README.md).
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import banks as B  # noqa: E402
import nse_sources as N  # noqa: E402

API = "https://www.nseindia.com/api/corporates-financial-results"
XBRL_PREFIX = "https://nsearchives.nseindia.com/corporate/xbrl/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Accept": "application/json,*/*", "Referer": "https://www.nseindia.com/"}
DELAY = 1.5


def get(session, url, as_json=False):
    for wait in (0, 10, 30):
        time.sleep(DELAY + wait)
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json() if as_json else r.text
        except requests.RequestException:
            pass
    raise RuntimeError(f"NSE did not answer {url}")


def fetch(session, symbol: str, entity_type: str, d_sib: bool) -> list:
    rows = []
    for period, annual in (("Annual", True), ("Quarterly", False)):
        filings = get(session, f"{API}?index=equities&symbol={symbol}&period={period}", as_json=True) or []
        for f in filings:
            url = f.get("xbrl") or ""
            if f.get("consolidated") != "Non-Consolidated" or not url.endswith(".xml"):
                continue
            ratios, note = N.bank_results_ratios(get(session, url), annual)
            if note:
                print(f"  {symbol} {f.get('toDate')}: {note}")
            end = pd.to_datetime(f.get("toDate"), format="%d-%b-%Y")
            filed = pd.to_datetime((f.get("broadCastDate") or f.get("filingDate"))[:11], format="%d-%b-%Y")
            for metric, value in ratios.items():
                rows.append({"symbol": symbol, "entity_type": entity_type, "d_sib": d_sib, "period_end": f"{end:%Y-%m-%d}",
                             "metric": metric, "value_pct": round(value, 4), "filing_date": f"{filed:%Y-%m-%d}",
                             "source": url + (f" ({note})" if note else "")})
        print(f"{symbol} {period}: {len(filings)} filings")
    return rows


def main() -> int:
    session = requests.Session()
    session.headers.update(HEADERS)
    B.BANKS_DIR.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        parts = arg.split(":")
        symbol, entity_type, d_sib = parts[0], parts[1], len(parts) > 2 and parts[2] == "dsib"
        new = pd.DataFrame(fetch(session, symbol, entity_type, d_sib), columns=B.COLUMNS)
        new = new.drop_duplicates(["period_end", "metric", "filing_date"])
        path = B.BANKS_DIR / f"{symbol}.csv"
        if path.exists():
            old = pd.read_csv(path, dtype=str)
            kept = old[~old["source"].fillna("").str.startswith(XBRL_PREFIX)]
            new = pd.concat([kept, new.astype(str)], ignore_index=True)
        new.sort_values(["period_end", "metric", "filing_date"]).to_csv(path, index=False)
        print(f"wrote {path} ({len(new)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
