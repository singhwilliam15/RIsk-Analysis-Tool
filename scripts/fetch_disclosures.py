"""
Download the official NSE disclosure data into data/disclosures/ and list every file in its manifest.

    python scripts/fetch_disclosures.py                 # covered symbols + scan for the most-pledged mid/small caps
    python scripts/fetch_disclosures.py --no-scan       # covered symbols only (reuses the last scan's picks)

Sources (all public NSE endpoints, read with a browser-like session at about one request every 1.5 seconds):
- price bands: the daily sec_list.csv, saved unchanged;
- F&O ban: fo_secban_DDMMYYYY.csv for recent trade dates, saved unchanged;
- ASM / GSM: the surveillance lists (JSON, kept in raw/), written to the surveillance template;
- pledges: each quarter's shareholding-pattern XBRL (promoter holding and shares pledged), last 8 quarters;
- ratings: NSE credit-rating disclosures, swept month by month, kept for the covered issuers.
Nothing is estimated: a file that cannot be downloaded or parsed is reported and left out, and the script stops
if NSE refuses requests. Parsing lives in nse_sources.py (tested offline).
"""

import argparse
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import disclosures as D  # noqa: E402
import nse_sources as N  # noqa: E402

OUT = D.DISCLOSURE_DIR
RAW = OUT / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Accept": "application/json,text/csv,*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
ARCHIVES = "https://nsearchives.nseindia.com"
API = "https://www.nseindia.com/api"
COVERED = ["RELIANCE", "HDFCBANK", "TCS", "ASIANPAINT", "BRITANNIA", "JPPOWER"]  # the app's presets + the small-cap example
SCAN_INDICES = {"Nifty Midcap 150": f"{ARCHIVES}/content/indices/ind_niftymidcap150list.csv",
                "Nifty Smallcap 250": f"{ARCHIVES}/content/indices/ind_niftysmallcap250list.csv"}
PLEDGED_PICKS = 5
QUARTERS = 8
RATING_MONTHS = 24
DELAY = 1.5


class Refused(RuntimeError):
    pass


class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.calls = 0

    def get(self, url: str, json_expected: bool = False):
        """One request, retried twice (after 10 s and 30 s) on a timeout or dropped connection."""
        for wait in (0, 10, 30):
            time.sleep(DELAY + wait)
            self.calls += 1
            try:
                r = self.s.get(url, timeout=60)
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
        else:
            raise Refused(f"NSE did not answer {url} after 3 tries ({last}); stopping.")
        if r.status_code in (401, 403, 429):
            raise Refused(f"NSE refused {url} (HTTP {r.status_code}); stopping. Try again later.")
        if r.status_code != 200:
            return None
        if json_expected:
            try:
                return r.json()
            except ValueError:
                raise Refused(f"NSE returned a non-JSON page for {url}; stopping.")
        return r.content


def manifest_entry(file, dataset, url, today, start, end, as_of=None, notes="", constants=None):
    entry = {"file": file, "dataset": dataset, "source_url": url, "downloaded_on": today, "coverage_start": start,
             "coverage_end": end, "notes": notes}
    if as_of:
        entry["as_of"] = as_of
    if constants:
        entry["constants"] = constants
    return entry


def isin_map(client) -> pd.DataFrame:
    blob = client.get(f"{ARCHIVES}/content/equities/EQUITY_L.csv")
    eq = pd.read_csv(io.BytesIO(blob))
    eq.columns = [c.strip() for c in eq.columns]
    return eq.rename(columns={"SYMBOL": "symbol", "ISIN NUMBER": "isin"})[["symbol", "isin"]]


def shareholding_rows(client, symbol: str, quarters: int) -> list:
    filings = client.get(f"{API}/corporate-share-holdings-master?index=equities&symbol={symbol}", json_expected=True) or []
    rows, seen = [], set()
    for filing in filings:  # newest first; walk back until `quarters` distinct quarters are read
        if len(rows) >= quarters:
            break
        url = filing.get("xbrl") or ""
        if not url.endswith(".xml") or filing["date"] in seen:
            continue  # no XBRL, or a later revision of a quarter already read
        seen.add(filing["date"])
        blob = client.get(url)
        parsed = N.shareholding_pledge(blob.decode("utf-8", errors="replace")) if blob else None
        if parsed is None:
            print(f"  {symbol} {filing['date']}: XBRL not readable, skipped")
            continue
        rows.append(N.pledge_row(symbol, filing, parsed))
    return rows


def scan_pledged(client, today: str) -> list:
    """Latest shareholding filing of every Nifty Midcap 150 and Smallcap 250 constituent; top non-financials by % pledged."""
    members = []
    for name, url in SCAN_INDICES.items():
        frame = pd.read_csv(io.BytesIO(client.get(url)))
        members.append(frame.assign(Index=name))
    members = pd.concat(members).drop_duplicates("Symbol")
    # Checkpoint: a rerun on the same day resumes where an interrupted scan stopped
    partial = RAW / f"pledge_scan_{today}_partial.csv"
    out = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    out = [r for r in out if not bool(r.get("skipped") == True)]  # noqa: E712  stocks skipped last time are tried again
    done = {r["symbol"] for r in out}
    for i, m in enumerate(members.to_dict("records"), 1):
        if m["Symbol"] in done:
            continue
        try:
            rows = shareholding_rows(client, m["Symbol"], 1)
        except Refused as exc:
            print(f"  {m['Symbol']}: skipped ({exc})")
            rows = []
        out.append({**rows[0], "industry": m["Industry"], "index": m["Index"]} if rows else
                   {"symbol": m["Symbol"], "industry": m["Industry"], "index": m["Index"], "skipped": True})
        if i % 25 == 0:
            pd.DataFrame(out).to_csv(partial, index=False)
            print(f"  scanned {i}/{len(members)}", flush=True)
    scan = pd.DataFrame(out)
    skipped = int(scan.get("skipped", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    scan = scan[~scan.get("skipped", pd.Series(False, index=scan.index)).fillna(False).astype(bool)]
    print(f"  scan: {len(scan)} filings read, {skipped} stocks skipped")
    if partial.exists():
        partial.unlink()
    scan.to_csv(RAW / f"pledge_scan_{today}.csv", index=False)
    picks = scan[scan["industry"] != "Financial Services"].sort_values("pledged_pct_of_total", ascending=False)
    return picks.head(PLEDGED_PICKS)["symbol"].tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-scan", action="store_true", help="reuse the pledged picks from the last scan")
    args = parser.parse_args()
    today = date.today()
    stamp, iso = f"{today:%d%m%Y}", f"{today:%Y-%m-%d}"
    RAW.mkdir(parents=True, exist_ok=True)
    client = Client()
    entries, report = [], []
    try:
        # Price bands: today's file, unchanged
        blob = client.get(f"{ARCHIVES}/content/equities/sec_list.csv")
        if blob:
            (OUT / f"sec_list_{stamp}.csv").write_bytes(blob)
            entries.append(manifest_entry(f"sec_list_{stamp}.csv", "price_bands", f"{ARCHIVES}/content/equities/sec_list.csv",
                                          iso, iso, iso, iso, "NSE price-band file as downloaded; the list date is the download date."))
        # F&O ban: the last five trade dates with a file
        found, day = 0, today
        while found < 5 and day > today - timedelta(days=14):
            name = f"fo_secban_{day:%d%m%Y}.csv"
            blob = client.get(f"{ARCHIVES}/archives/fo/sec_ban/{name}")
            if blob and b"Trade Date" in blob:
                (OUT / name).write_bytes(blob)
                d = f"{day:%Y-%m-%d}"
                entries.append(manifest_entry(name, "fo_ban", f"{ARCHIVES}/archives/fo/sec_ban/{name}", iso, d, d,
                                              notes="NSE securities in ban period, unchanged."))
                found += 1
            day -= timedelta(days=1)
        # Surveillance
        asm = client.get(N.ASM_URL, json_expected=True)
        gsm = client.get(N.GSM_URL, json_expected=True)
        (RAW / f"asm_{stamp}.json").write_text(json.dumps(asm))
        (RAW / f"gsm_{stamp}.json").write_text(json.dumps(gsm))
        surv = N.surveillance_rows(asm, gsm, today)
        surv.to_csv(OUT / f"surveillance_{stamp}.csv", index=False)
        entries.append(manifest_entry(f"surveillance_{stamp}.csv", "surveillance", N.ASM_URL + " ; " + N.GSM_URL, iso, iso, iso,
                                      notes=f"Long- and short-term ASM and GSM lists of {iso} (raw JSON in raw/). The lists show "
                                            "the current stage only, so date_in is the list date."))
        # Pledges: covered symbols and the most-pledged mid/small caps
        if args.no_scan:
            last = sorted(RAW.glob("pledge_scan_*.csv"))
            picks = pd.read_csv(last[-1]).query("industry != 'Financial Services'").sort_values(
                "pledged_pct_of_total", ascending=False).head(PLEDGED_PICKS)["symbol"].tolist() if last else []
        else:
            print("Scanning Nifty Midcap 150 and Smallcap 250 shareholding filings (about 400 requests)...")
            picks = scan_pledged(client, iso)
        symbols = COVERED + [p for p in picks if p not in COVERED]
        pledges = []
        for sym in symbols:
            rows = shareholding_rows(client, sym, QUARTERS)
            pledges += rows
            report.append(f"pledges {sym}: {len(rows)} quarters")
        pledge_frame = pd.DataFrame(pledges)
        pledge_frame.to_csv(OUT / f"pledges_{stamp}.csv", index=False)
        entries.append(manifest_entry(
            f"pledges_{stamp}.csv", "pledges", f"{API}/corporate-share-holdings-master (XBRL per row in 'source')", iso,
            pledge_frame["quarter_end"].min(), pledge_frame["quarter_end"].max(),
            notes=f"Shareholding-pattern XBRL, up to {QUARTERS} quarters in the current taxonomy (from about mid-2025; older filings report pledges combined with other encumbrances, so they are not used). Covered: {', '.join(COVERED)}; plus the "
                  f"{PLEDGED_PICKS} non-financial Nifty Midcap 150 / Smallcap 250 stocks with the highest share of total "
                  f"shares pledged in their latest filing (scan of {iso}): {', '.join(picks)}. disclosure_date = NSE broadcast date."))
        # Ratings: month-by-month sweep, kept for the covered issuers
        isins = isin_map(client)
        issuers = {row["isin"][:7]: row["symbol"] for row in isins[isins["symbol"].isin(symbols)].to_dict("records")}
        records = []
        start = (pd.Timestamp(today) - pd.DateOffset(months=RATING_MONTHS)).replace(day=1)
        for month_start in pd.date_range(start, today, freq="MS"):
            month_end = min(month_start + pd.offsets.MonthEnd(0), pd.Timestamp(today))
            url = (f"{API}/corporate-credit-rating?index=equities&from_date={month_start:%d-%m-%Y}"
                   f"&to_date={month_end:%d-%m-%Y}")
            records += client.get(url, json_expected=True) or []
        kept = [r for r in records if str(r.get("ISIN", ""))[:7] in issuers]
        (RAW / f"ratings_{stamp}.json").write_text(json.dumps(kept))
        ratings, skipped = N.rating_rows(kept, issuers)
        ratings.to_csv(OUT / f"ratings_{stamp}.csv", index=False)
        entries.append(manifest_entry(
            f"ratings_{stamp}.csv", "ratings", f"{API}/corporate-credit-rating (XBRL per row in 'source')", iso,
            f"{start:%Y-%m-%d}", iso,
            notes=f"NSE credit-rating disclosures for the covered issuers ({len(records)} records swept, {len(kept)} for "
                  f"covered issuers, {skipped} skipped because the action could not be read). instrument = the rated ISIN."))
    except Refused as exc:
        print(exc)
        print("Nothing was written to the manifest.")
        return 1
    # Manifest: replace this script's datasets, keep any entry added by hand
    path = OUT / D.MANIFEST_NAME
    content = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"files": []}
    auto = {"price_bands", "fo_ban", "surveillance", "pledges", "ratings"}
    stale = [e["file"] for e in content.get("files", []) if e.get("dataset") in auto]
    for name in stale:
        if (OUT / name).exists() and name not in {e["file"] for e in entries}:
            (OUT / name).unlink()
    content["files"] = [e for e in content.get("files", []) if e.get("dataset") not in auto] + entries
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    # Check that every written file loads
    loaded = D.load_disclosures(OUT)
    print(loaded["files"][["File", "Rows", "Rejected rows"]].to_string(index=False))
    for issue in loaded["issues"][:20]:
        print("issue:", issue)
    print("\n".join(report))
    print(f"{client.calls} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
