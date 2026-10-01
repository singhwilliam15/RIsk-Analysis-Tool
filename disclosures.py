"""
Disclosures Module
Loaders, CSV templates and validators for Indian disclosure data that the user downloads from official
sources (NSE, BSE, rating agencies). This module never downloads or fills in data.

Datasets:
    pledges         promoter holding and pledged/encumbered %, by quarter
    surveillance    ASM / GSM surveillance stages, with dates in and out
    price_bands     daily price bands (circuit limits)
    fo_ban          securities in the F&O ban period, by trade date
    ratings         credit rating actions
    auditor_events  auditor resignations and modified (qualified, adverse, disclaimer) opinions

Files live in data/disclosures/ and are listed in data/disclosures/manifest.json with their source URL,
download date and coverage. A file without a manifest entry is not loaded, because its source is unknown.
"""

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DISCLOSURE_DIR = Path(__file__).parent / "data" / "disclosures"
MANIFEST_NAME = "manifest.json"
MANIFEST_KEYS = ("file", "dataset", "source_url", "downloaded_on", "coverage_start", "coverage_end")

# SEBI LODR Regulation 31(1)(b): the shareholding pattern (including pledged shares) is filed within
# 21 days of each quarter end. Used as the public date when a pledge row has no disclosure date.
SHAREHOLDING_FILING_DAYS = 21


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "symbol", "date", "pct", "number", "text" or "choice"
    required: bool = True
    choices: tuple = ()


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    columns: tuple
    date_column: str  # the date that decides when a row became public, for as-of filtering
    aliases: dict      # normalised header in official files -> canonical column
    official_source: str
    download_steps: str

    @property
    def column_names(self):
        return [c.name for c in self.columns]


SURVEILLANCE_MEASURES = ("ASM-LT", "ASM-ST", "GSM", "ESM")
RATING_ACTIONS = ("assigned", "reaffirmed", "upgraded", "downgraded", "placed on watch", "withdrawn", "suspended")
AUDITOR_EVENT_TYPES = ("resignation", "qualified opinion", "adverse opinion", "disclaimer of opinion", "emphasis of matter")
PRICE_BANDS = (2.0, 5.0, 10.0, 20.0)
NO_BAND = "no band"

DATASETS = {
    "pledges": Dataset(
        "pledges", "Promoter holding and pledges",
        (Column("symbol", "symbol"), Column("quarter_end", "date"), Column("promoter_holding_pct", "pct"),
         Column("pledged_pct_of_promoter", "pct"), Column("pledged_pct_of_total", "pct", False),
         Column("pledged_shares", "number", False), Column("disclosure_date", "date", False),
         Column("source", "text", False)),
        "disclosure_date",
        {"symbol": "symbol", "quarter": "quarter_end", "quarterend": "quarter_end",
         "promoterholding": "promoter_holding_pct", "promoterholdingoftotalshares": "promoter_holding_pct",
         "promotersharesencumberedofpromotershares": "pledged_pct_of_promoter",
         "ofpromotershares": "pledged_pct_of_promoter",
         "promotersharesencumberedoftotalshares": "pledged_pct_of_total", "oftotalshares": "pledged_pct_of_total",
         "promotersharesencumberednoofshares": "pledged_shares", "broadcastdate": "disclosure_date",
         "disclosuredate": "disclosure_date"},
        "NSE Corporate Filings: Pledged Data, and the quarterly Shareholding Pattern",
        "nseindia.com → Companies → Corporate Filings → Pledged Data (and Shareholding Pattern). Filter by symbol, "
        "download the CSV, add a Symbol column if the file has company names only, save it in data/disclosures/ "
        "and add a manifest entry.",
    ),
    "surveillance": Dataset(
        "surveillance", "ASM / GSM surveillance",
        (Column("symbol", "symbol"), Column("measure", "choice", choices=SURVEILLANCE_MEASURES), Column("stage", "text"),
         Column("date_in", "date"), Column("date_out", "date", False), Column("source", "text", False)),
        "date_in",
        {"symbol": "symbol", "measure": "measure", "survdesc": "measure", "stage": "stage", "gsmstage": "stage",
         "asmstage": "stage", "datein": "date_in", "datefrom": "date_in", "dateout": "date_out", "dateto": "date_out"},
        "NSE Surveillance: Long-term ASM, Short-term ASM and GSM lists, and NSE surveillance circulars",
        "nseindia.com → Regulation → Surveillance → ASM / GSM. Download each list for the date you need. The lists "
        "show today's stage only; record the list date as date_in (or pass it as the file's as-of date), and "
        "take earlier entries and exits from the dated circulars.",
    ),
    "price_bands": Dataset(
        "price_bands", "Price bands",
        (Column("symbol", "symbol"), Column("series", "text", False), Column("band", "text"),
         Column("effective_date", "date"), Column("source", "text", False)),
        "effective_date",
        {"symbol": "symbol", "series": "series", "band": "band", "priceband": "band", "pricebandpercent": "band",
         "effectivedate": "effective_date", "date": "effective_date"},
        "NSE daily price band file (sec_list) and price-band revision circulars",
        "nseindia.com → All Reports → Equities → 'Securities available for trading / price bands' (sec_list CSV) for "
        "the date you need. The file has no date column, so pass the report date as the file's as-of date.",
    ),
    "fo_ban": Dataset(
        "fo_ban", "F&O ban list",
        (Column("trade_date", "date"), Column("symbol", "symbol"), Column("source", "text", False)),
        "trade_date",
        {"tradedate": "trade_date", "date": "trade_date", "symbol": "symbol", "security": "symbol"},
        "NSE daily 'Securities in ban period' file (fo_secban.csv)",
        "nseindia.com → All Reports → Derivatives → 'Securities in Ban Period' (fo_secban.csv), one file per trade "
        "date. Save each file unchanged; the loader reads the trade date from its first line.",
    ),
    "ratings": Dataset(
        "ratings", "Credit rating actions",
        (Column("symbol", "symbol"), Column("agency", "text"), Column("instrument", "text"), Column("rating", "text"),
         Column("outlook", "text", False), Column("action", "choice", choices=RATING_ACTIONS),
         Column("action_date", "date"), Column("source", "text", False)),
        "action_date",
        {"symbol": "symbol", "agency": "agency", "ratingagency": "agency", "instrument": "instrument",
         "instrumenttype": "instrument", "rating": "rating", "outlook": "outlook", "action": "action",
         "ratingaction": "action", "date": "action_date", "actiondate": "action_date"},
        "Rating rationales from CRISIL, ICRA, CARE, India Ratings, Acuité, Brickwork; credit-rating disclosures on NSE/BSE",
        "Search the company on each agency's website, open each rating rationale, and enter one row per action "
        "with the rationale's URL in 'source'. No agency publishes one consolidated CSV.",
    ),
    "auditor_events": Dataset(
        "auditor_events", "Auditor resignations and modified opinions",
        (Column("symbol", "symbol"), Column("event_type", "choice", choices=AUDITOR_EVENT_TYPES),
         Column("auditor", "text", False), Column("event_date", "date"), Column("details", "text", False),
         Column("source", "text", False)),
        "event_date",
        {"symbol": "symbol", "eventtype": "event_type", "event": "event_type", "auditor": "auditor",
         "date": "event_date", "eventdate": "event_date", "broadcastdate": "event_date", "details": "details",
         "subject": "details"},
        "NSE/BSE corporate announcements ('Change in Auditors', 'Resignation of Statutory Auditor'), and audit reports",
        "nseindia.com → Companies → Corporate Filings → Announcements, filter by symbol and subject 'Auditor'. For "
        "modified opinions, read the audit report in the annual or quarterly results filing. One row per event, "
        "with the announcement URL in 'source'.",
    ),
}


def _key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def normalise_symbol(value) -> str:
    """NSE/BSE symbol without the Yahoo suffix: 'reliance.ns' -> 'RELIANCE'."""
    text = str(value).strip().upper()
    return re.sub(r"\.(NS|BO)$", "", text)


def template(dataset: str) -> pd.DataFrame:
    """An empty CSV template with the dataset's canonical columns."""
    return pd.DataFrame(columns=DATASETS[dataset].column_names)


def write_templates(directory: Path) -> list:
    """Write every template to `directory` as <dataset>_template.csv; returns the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in DATASETS:
        path = directory / f"{name}_template.csv"
        template(name).to_csv(path, index=False)
        paths.append(path)
    return paths


def _read(source) -> str:
    """Text of a path, a file-like object, or a string that already holds the file's contents."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and len(source) < 250):
        try:
            if Path(source).is_file():
                return Path(source).read_text(encoding="utf-8-sig")
        except OSError:
            pass
    if hasattr(source, "read"):
        content = source.read()
        return content.decode("utf-8-sig") if isinstance(content, bytes) else content
    return str(source)


FO_BAN_HEADER = re.compile(r"trade\s*date\s*[:\-]?\s*(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{4})", re.I)


def _parse_fo_ban_official(text: str):
    """NSE fo_secban.csv: a 'Securities in Ban For Trade Date DD-MON-YYYY:' line, then 'n,SYMBOL' rows."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = FO_BAN_HEADER.search(lines[0])
    if not match:
        return None
    trade_date = pd.to_datetime(match.group(1), dayfirst=True)
    symbols = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        candidate = parts[-1] if parts else ""
        if candidate and candidate.upper() not in ("NIL", "NONE"):
            symbols.append(candidate)
    return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "symbol": symbols})


def _rename_columns(frame: pd.DataFrame, dataset: Dataset) -> pd.DataFrame:
    canonical = {_key(c): c for c in dataset.column_names}
    renamed = {}
    for col in frame.columns:
        key = _key(col)
        target = canonical.get(key) or dataset.aliases.get(key)
        if target and target not in renamed.values():
            renamed[col] = target
    return frame.rename(columns=renamed)[list(renamed.values())]


def _parse_date(value):
    if value is None or (isinstance(value, str) and not value.strip()) or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
    try:
        return pd.to_datetime(value, dayfirst=not re.match(r"^\d{4}-", str(value).strip()))
    except (ValueError, TypeError):
        return None


def validate(frame: pd.DataFrame, dataset: str) -> tuple:
    """
    Check and type a frame with canonical column names. Returns (clean frame, errors). Rows with an error
    are dropped; each error names the spreadsheet row (header = row 1).
    """
    spec = DATASETS[dataset]
    missing = [c.name for c in spec.columns if c.required and c.name not in frame.columns]
    if missing:
        return pd.DataFrame(columns=spec.column_names), [f"{spec.title}: missing column(s) {', '.join(missing)}"]
    frame = frame.reindex(columns=spec.column_names)
    clean, errors = [], []
    for i, row in frame.reset_index(drop=True).iterrows():
        line, out, bad = i + 2, {}, False
        for col in spec.columns:
            value = row[col.name]
            blank = value is None or (isinstance(value, str) and not value.strip()) or (
                not isinstance(value, str) and pd.isna(value))
            if blank:
                if col.required:
                    errors.append(f"{spec.title}, row {line}: {col.name} is blank.")
                    bad = True
                out[col.name] = pd.NaT if col.kind == "date" else np.nan if col.kind in ("pct", "number") else None
                continue
            if col.kind == "symbol":
                out[col.name] = normalise_symbol(value)
            elif col.kind == "date":
                parsed = _parse_date(value)
                if parsed is None:
                    errors.append(f"{spec.title}, row {line}: {col.name} '{value}' is not a date.")
                    bad = True
                out[col.name] = parsed
            elif col.kind in ("pct", "number"):
                number = pd.to_numeric(str(value).replace(",", "").replace("%", "").strip(), errors="coerce")
                if not np.isfinite(number) or number < 0 or (col.kind == "pct" and number > 100):
                    errors.append(f"{spec.title}, row {line}: {col.name} '{value}' must be "
                                  + ("a percentage between 0 and 100." if col.kind == "pct" else "a non-negative number."))
                    bad = True
                out[col.name] = float(number) if np.isfinite(number) else np.nan
            elif col.kind == "choice":
                text = str(value).strip()
                match = next((c for c in col.choices if c.lower() == text.lower()), None)
                if match is None:
                    errors.append(f"{spec.title}, row {line}: {col.name} '{value}' must be one of {', '.join(col.choices)}.")
                    bad = True
                out[col.name] = match
            else:
                out[col.name] = str(value).strip()
        if dataset == "price_bands" and out.get("band") is not None:
            band = str(out["band"]).strip().lower().replace("%", "")
            if band == NO_BAND:
                out["band"], out["band_pct"] = "No Band", np.nan
            else:
                number = pd.to_numeric(band, errors="coerce")
                if number not in PRICE_BANDS:
                    errors.append(f"{spec.title}, row {line}: band '{row['band']}' must be 2, 5, 10, 20 or 'No Band'.")
                    bad = True
                out["band"], out["band_pct"] = (f"{number:g}%" if np.isfinite(number) else None), float(number)
        if dataset == "pledges" and not bad and np.isfinite(out.get("pledged_pct_of_total", np.nan)):
            if out["pledged_pct_of_total"] > out["promoter_holding_pct"] + 1e-9:
                errors.append(f"{spec.title}, row {line}: pledged % of total shares exceeds the promoter holding.")
                bad = True
        if dataset == "surveillance" and not bad and not pd.isna(out.get("date_out")) and out["date_out"] < out["date_in"]:
            errors.append(f"{spec.title}, row {line}: date_out is before date_in.")
            bad = True
        if not bad:
            clean.append(out)
    columns = spec.column_names + (["band_pct"] if dataset == "price_bands" else [])
    return pd.DataFrame(clean, columns=columns), errors


def load_file(source, dataset: str, as_of=None, constants: dict = None) -> tuple:
    """
    Read one file in either the template layout or the official layout (headers matched through the
    dataset's alias table). `as_of` fills a required date column the official file does not carry
    (e.g. the date of a price-band or ASM list); `constants` fills other columns the file implies but
    does not contain (e.g. {"measure": "ASM-LT"} for NSE's long-term ASM list). Returns (clean frame, errors, notes).
    """
    spec = DATASETS[dataset]
    text = _read(source)
    notes = []
    frame = _parse_fo_ban_official(text) if dataset == "fo_ban" else None
    if frame is not None:
        notes.append("Read as the official NSE fo_secban layout.")
    else:
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
        except Exception as exc:
            return pd.DataFrame(columns=spec.column_names), [f"{spec.title}: could not read the file ({exc})."], notes
        frame = _rename_columns(frame, spec)
    if as_of is not None:
        for col in spec.columns:
            if col.kind == "date" and col.required and col.name not in frame.columns:
                frame[col.name] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
                notes.append(f"{col.name} taken from the file's as-of date {pd.Timestamp(as_of):%d %b %Y}.")
    for name, constant in (constants or {}).items():
        if name in spec.column_names and name not in frame.columns:
            frame[name] = constant
            notes.append(f"{name} = {constant} for every row (from the manifest).")
    clean, errors = validate(frame, dataset)
    return clean, errors, notes


def public_date(frame: pd.DataFrame, dataset: str) -> pd.Series:
    """
    The date each row became public. Pledge rows without a disclosure date use quarter end +
    SHAREHOLDING_FILING_DAYS (the regulatory filing deadline); other datasets use their own date column.
    """
    spec = DATASETS[dataset]
    dates = pd.to_datetime(frame[spec.date_column], errors="coerce")
    if dataset == "pledges":
        fallback = pd.to_datetime(frame["quarter_end"], errors="coerce") + pd.Timedelta(days=SHAREHOLDING_FILING_DAYS)
        dates = dates.fillna(fallback)
    return dates


def as_of(frame: pd.DataFrame, dataset: str, date) -> pd.DataFrame:
    """Only the rows that were public on `date` (point-in-time discipline)."""
    if frame.empty:
        return frame
    return frame[public_date(frame, dataset) <= pd.Timestamp(date)]


# ---------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------

def read_manifest(directory: Path = DISCLOSURE_DIR) -> tuple:
    """Manifest entries and any problems with them. A missing manifest is an empty one."""
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        return [], []
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"{MANIFEST_NAME} is not valid JSON: {exc}"]
    entries, problems = [], []
    for i, entry in enumerate(content.get("files", []), start=1):
        missing = [k for k in MANIFEST_KEYS if not entry.get(k)]
        if missing:
            problems.append(f"Manifest entry {i} ({entry.get('file', '?')}): missing {', '.join(missing)}.")
            continue
        if entry["dataset"] not in DATASETS:
            problems.append(f"Manifest entry {i} ({entry['file']}): unknown dataset '{entry['dataset']}'.")
            continue
        if any(_parse_date(entry[k]) is None or pd.isna(_parse_date(entry[k]))
               for k in ("downloaded_on", "coverage_start", "coverage_end")):
            problems.append(f"Manifest entry {i} ({entry['file']}): dates must be YYYY-MM-DD.")
            continue
        entries.append(entry)
    return entries, problems


def load_disclosures(directory: Path = DISCLOSURE_DIR) -> dict:
    """
    Load every file listed in the manifest. Returns {"data": {dataset: frame}, "files": summary frame,
    "issues": [...]}. Each data row keeps its file, source URL and download date. CSV files in the
    directory that the manifest does not list are reported and skipped.
    """
    directory = Path(directory)
    entries, issues = read_manifest(directory)
    frames = {name: [] for name in DATASETS}
    summary = []
    for entry in entries:
        path = directory / entry["file"]
        if not path.exists():
            issues.append(f"{entry['file']}: listed in the manifest but not found.")
            continue
        clean, errors, notes = load_file(path, entry["dataset"], as_of=entry.get("as_of") or entry["coverage_end"],
                                         constants=entry.get("constants"))
        issues += [f"{entry['file']}: {e}" for e in errors]
        clean = clean.assign(file=entry["file"], source_url=entry["source_url"],
                             downloaded_on=pd.Timestamp(entry["downloaded_on"]))
        frames[entry["dataset"]].append(clean)
        summary.append({"File": entry["file"], "Dataset": DATASETS[entry["dataset"]].title, "Rows": len(clean),
                        "Rejected rows": len(errors), "Coverage": f"{entry['coverage_start']} to {entry['coverage_end']}",
                        "Downloaded": entry["downloaded_on"], "Source": entry["source_url"], "Notes": " ".join(notes)})
    listed = {e["file"] for e in entries}
    if directory.exists():
        for path in sorted(directory.glob("*.csv")):
            if path.name not in listed:
                issues.append(f"{path.name}: not in {MANIFEST_NAME}, so not loaded (its source is unknown).")
    data = {name: (pd.concat(parts, ignore_index=True) if parts else template(name)) for name, parts in frames.items()}
    return {"data": data, "files": pd.DataFrame(summary), "issues": issues}
