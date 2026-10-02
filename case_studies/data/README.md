# Case-study data (files you download)

The case studies use only data that was public before each collapse, and nothing here is filled in. Put the
files from the checklist in `docs/case_studies.md` here, then run `python case_studies/run.py`.

| Folder | What goes in it | Format |
| --- | --- | --- |
| `prices/` | Daily prices for a stock Yahoo no longer has (DHFL) | `<TICKER>.csv` with columns `Date, Open, High, Low, Close, Volume`, e.g. `DHFL.csv` (NSE "security-wise price volume" export, one row per day) |
| `fundamentals/` | Annual statements for the years before each event | `<TICKER>.csv` in the app's fundamentals template (Overview → Fundamentals → Download template), e.g. `ZEEL.NS.csv`; include `shares_issued`, both debt lines, and the filing date if you know it |
| `disclosures/` | Pledges, surveillance lists, F&O ban files, rating actions, auditor events | The same files and templates as `data/disclosures/`, each listed in `disclosures/manifest.json` with its source URL and download date |

A file that is missing simply leaves that pillar "not available" in the results.
