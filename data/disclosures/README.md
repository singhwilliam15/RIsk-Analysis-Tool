# Disclosure data (India)

The event, credit and liquidity pillars use Indian disclosure data that **you download from official sources**. The tool never downloads, estimates or fills in these figures. NSE blocks automated downloads, so there is no refresh script.

## Adding a file

1. Download the file from the official source below.
2. Save it in this folder **unchanged**, or copy its rows into the matching template from `templates/`.
3. Add an entry to `manifest.json`:

```json
{
  "file": "fo_secban_02102026.csv",
  "dataset": "fo_ban",
  "source_url": "https://www.nseindia.com/all-reports-derivatives",
  "downloaded_on": "2026-10-02",
  "coverage_start": "2026-10-02",
  "coverage_end": "2026-10-02",
  "as_of": "2026-10-02",
  "notes": "Securities in ban period for trade date 02-Oct-2026"
}
```

`as_of` is the date of a list that has no date column, such as a price-band file or an ASM/GSM list. If you leave it out, `coverage_end` is used. `constants` fills a column that the file implies but does not contain. For example, NSE's long-term ASM list has no "measure" column, so its entry adds `"constants": {"measure": "ASM-LT"}`. Files missing from the manifest are skipped and reported, because their source is unknown.

The app's Overview page lists every loaded file, the rows rejected and why, and offers each template for download.

## Datasets

| Dataset | Official source | How to download | Date used for point-in-time |
| --- | --- | --- | --- |
| `pledges` | NSE Corporate Filings → Pledged Data, and the quarterly Shareholding Pattern | Filter by symbol and download the CSV. Add a `Symbol` column if the file has company names only. | `disclosure_date`. If it is blank: quarter end + 21 days, the SEBI LODR Reg. 31 filing deadline. |
| `surveillance` | NSE Surveillance → Long-term ASM, Short-term ASM, GSM | Download each list for the date you need. A list shows only the current stage, so set `as_of` to the list date. Take entries and exits from the dated NSE circulars. | `date_in` |
| `price_bands` | NSE daily price-band file (`sec_list`), and price-band revision circulars | All Reports → Equities. The file has no date column, so set `as_of`. | `effective_date` |
| `fo_ban` | NSE `fo_secban.csv`, "Securities in Ban Period" | All Reports → Derivatives, one file per trade date. Save it unchanged; the trade date is read from its first line. | `trade_date` |
| `ratings` | Rating rationales from CRISIL, ICRA, CARE, India Ratings, Acuité and Brickwork | No agency publishes one consolidated file. Enter one row per action, with the rationale's URL in `source`. | `action_date` |
| `auditor_events` | NSE/BSE corporate announcements ("Change in Auditors", "Resignation of Statutory Auditor"), and audit reports in results filings | One row per event, with the announcement URL in `source`. | `event_date` |

**Layouts not yet verified.** The loaders read the templates exactly. They also accept official files whose column headers match the alias table in `disclosures.py`, but those aliases were written without a real download to check against. Only the `fo_secban.csv` layout is parsed structurally. The first time you add a real file of each kind, check the Overview page for rejected rows, and extend the aliases if a header is not recognised.

Allowed values:
- `surveillance.measure`: ASM-LT, ASM-ST, GSM, ESM
- `price_bands.band`: 2, 5, 10, 20, or No Band
- `ratings.action`: assigned, reaffirmed, upgraded, downgraded, placed on watch, withdrawn, suspended
- `auditor_events.event_type`: resignation, qualified opinion, adverse opinion, disclaimer of opinion, emphasis of matter

Percentages run from 0 to 100. Dates use YYYY-MM-DD, or DD-MM-YYYY / DD-MON-YYYY as in NSE files.
