"""
Download NSE's daily equity bhavcopies, corporate actions and symbol changes for the wider test
(docs/case_studies.md, "Wider test"), and build one adjusted price panel.

    python scripts/fetch_bhavcopy.py                    # 2014-01-01 to 2025-12-31, then build the panel
    python scripts/fetch_bhavcopy.py --build-only       # rebuild the panel from files already downloaded

Bhavcopies list every security traded that day, including stocks later delisted, so the universe is not limited to
today's survivors (Yahoo drops delisted names). Raw files go to data/cache/bhavcopy/ (gitignored; about 300 MB) and
are never edited; a date that returns 404 in both formats is a holiday and is remembered. NSE's previous close is
NOT adjusted for bonuses and splits (Reliance's 1:1 bonus on 7 Sep 2017 shows 1,645.4 → 818.1), so prices are
back-adjusted with NSE's corporate-action list (universe.py). Nothing is estimated.
"""

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import universe as U  # noqa: E402

RAW = U.CACHE / "bhavcopy"
ARCHIVES = "https://nsearchives.nseindia.com"
API = "https://www.nseindia.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Accept": "application/json,text/csv,*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
UDIFF_FROM = date(2024, 7, 8)  # NSE's first UDiFF-format bhavcopy
ARCHIVE_DELAY, API_DELAY = 0.4, 1.5


def old_url(d: date) -> str:
    return f"{ARCHIVES}/content/historical/EQUITIES/{d.year}/{d.strftime('%b').upper()}/cm{d.strftime('%d%b%Y').upper()}bhav.csv.zip"


def udiff_url(d: date) -> str:
    return f"{ARCHIVES}/content/cm/BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def get(session, url: str, delay: float):
    for wait in (0, 10, 30, 90):
        time.sleep(delay + wait)
        try:
            r = session.get(url, timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            continue
        if r.status_code in (403, 429, 503):
            continue
        return r
    raise RuntimeError(f"NSE did not answer {url} after 4 tries; rerun later (finished files are kept)")


def download(start: date, end: date) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    holidays_path = RAW / "holidays.json"
    holidays = set(json.loads(holidays_path.read_text())) if holidays_path.exists() else set()
    session = requests.Session()
    session.headers.update(HEADERS)
    d, fetched, started = start, 0, time.time()
    while d <= end:
        target = RAW / f"{d.isoformat()}.csv.zip"
        if d.weekday() < 5 and not target.exists() and d.isoformat() not in holidays:
            urls = [udiff_url(d), old_url(d)] if d >= UDIFF_FROM else [old_url(d), udiff_url(d)]
            for url in urls:
                r = get(session, url, ARCHIVE_DELAY)
                if r.status_code == 200 and r.content[:2] == b"PK":
                    target.write_bytes(r.content)
                    fetched += 1
                    break
            else:
                holidays.add(d.isoformat())
                holidays_path.write_text(json.dumps(sorted(holidays)))
            if fetched and fetched % 100 == 0:
                print(f"  {d}: {fetched} files ({time.time() - started:.0f}s)", flush=True)
        d += timedelta(days=1)
    print(f"bhavcopies: {fetched} new files, {len(holidays)} holidays", flush=True)


def corporate_actions(start: date, end: date) -> None:
    """NSE's corporate-action list (equities), one request per year, saved unchanged as JSON."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:  # cookies for the API, best effort: the home page sometimes answers 403 while the API still works
        session.get("https://www.nseindia.com/", timeout=30)
    except requests.RequestException:
        pass
    for year in range(start.year, end.year + 1):
        target = RAW / f"corporate_actions_{year}.json"
        if target.exists():
            continue
        url = f"{API}/corporates-corporateActions?index=equities&from_date=01-01-{year}&to_date=31-12-{year}"
        r = get(session, url, API_DELAY)
        if r.status_code != 200:
            raise RuntimeError(f"corporate actions {year}: HTTP {r.status_code}")
        target.write_text(json.dumps(r.json()))
        print(f"  corporate actions {year}: {len(r.json())} records", flush=True)
    r = get(session, f"{ARCHIVES}/content/equities/symbolchange.csv", ARCHIVE_DELAY)
    if r.status_code == 200:
        (RAW / "symbolchange.csv").write_bytes(r.content)


def read_one(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), dtype=str)
    return U.normalise_bhavcopy(raw, pd.Timestamp(path.name[:10]))


def build() -> None:
    files = sorted(RAW.glob("????-??-??.csv.zip"))
    frames = [read_one(p) for p in files]
    panel = pd.concat(frames, ignore_index=True)
    actions = U.parse_corporate_actions([json.loads(p.read_text()) for p in sorted(RAW.glob("corporate_actions_*.json"))])
    changes = pd.read_csv(RAW / "symbolchange.csv", header=None, dtype=str, encoding="latin-1") \
        if (RAW / "symbolchange.csv").exists() else None
    panel = U.chain_symbols(panel, U.parse_symbol_changes(changes) if changes is not None else {})
    panel = U.adjust(panel, actions)
    U.CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(U.PANEL_PATH, index=False)
    actions.to_csv(U.CACHE / "corporate_actions.csv", index=False)
    print(f"panel: {len(panel):,} rows, {panel['symbol'].nunique():,} symbols, {panel['date'].nunique():,} days, "
          f"{len(actions):,} bonus/split actions; written to {U.PANEL_PATH}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if not args.build_only:
        download(start, end)
        corporate_actions(start, end)
    build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
