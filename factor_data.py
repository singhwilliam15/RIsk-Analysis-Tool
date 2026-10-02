"""
Factor data: parsers for the two public factor libraries, and the loader the app uses.

- India: IIM Ahmedabad Indian Fama-French-Momentum data library (Agarwalla, Jacob and Varma, 2013), daily four
  factors and the risk-free rate, survivorship-bias adjusted.
- US: Kenneth R. French data library, daily Fama-French 3 factors and the momentum factor.

scripts/refresh_factor_data.py downloads the files and writes data/factors/<region>_daily.csv (decimal returns,
columns FACTOR_COLUMNS) and data/factors/metadata.json. The app only reads those files; it never downloads
factor data itself and never fills in a missing day.
"""

import io
import json
from pathlib import Path

import pandas as pd

FACTOR_DIR = Path(__file__).parent / "data" / "factors"
METADATA_NAME = "metadata.json"
FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "MOM"]
FILE_COLUMNS = ["Date"] + FACTOR_COLUMNS + ["RF"]

IIMA_PAGE = "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
FRENCH_PAGE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
FRENCH_FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
SOURCES = {
    "india": {
        "name": "IIM Ahmedabad Indian Fama-French-Momentum data library",
        "page": IIMA_PAGE,
        # The release prefix (YYYY-MM) changes with each update; the refresh script takes it as an argument
        "file_pattern": IIMA_PAGE + "DATA/{release}_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv",
        "citation": "Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013), Four factor model in Indian equities market, "
                    "Working Paper W.P. No. 2013-09-05, Indian Institute of Management, Ahmedabad.",
    },
    "us": {
        "name": "Kenneth R. French data library",
        "page": FRENCH_PAGE,
        "ff3": FRENCH_FTP + "F-F_Research_Data_Factors_daily_CSV.zip",
        "mom": FRENCH_FTP + "F-F_Momentum_Factor_daily_CSV.zip",
        "citation": "Kenneth R. French, Data Library, Tuck School of Business at Dartmouth.",
    },
}
MISSING_CODES = (-99.99, -999.0)


def parse_iima(text: str) -> pd.DataFrame:
    """
    IIMA daily file: columns Date, SMB, HML, WML, MF (market minus risk-free), RF, in percent; 'NA' for missing.
    Returns FILE_COLUMNS in decimals, dropping days with any factor missing.
    """
    raw = pd.read_csv(io.StringIO(text), na_values=["NA", ""])
    raw.columns = [c.strip() for c in raw.columns]
    needed = {"Date", "SMB", "HML", "WML", "MF", "RF"}
    if not needed <= set(raw.columns):
        raise ValueError(f"IIMA file is missing columns: {sorted(needed - set(raw.columns))}")
    out = pd.DataFrame({"Date": pd.to_datetime(raw["Date"]), "MKT_RF": raw["MF"], "SMB": raw["SMB"], "HML": raw["HML"],
                        "MOM": raw["WML"], "RF": raw["RF"]})
    out[FILE_COLUMNS[1:]] = out[FILE_COLUMNS[1:]].astype(float) / 100
    return out.dropna().reset_index(drop=True)


def parse_french(text: str) -> pd.DataFrame:
    """
    A daily Kenneth French CSV: free-text notes, then a header line starting with a comma, then YYYYMMDD rows,
    a blank line and a copyright footer. Values are percent; -99.99 and -999 mean missing.
    Returns Date plus the file's own columns, in decimals.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith(",")), None)
    if start is None:
        raise ValueError("No header line found in the French file.")
    body = [lines[start]]
    for line in lines[start + 1:]:
        if not line.strip() or not line.strip()[:8].isdigit():
            break
        body.append(line)
    raw = pd.read_csv(io.StringIO("\n".join(body)), skipinitialspace=True)
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"].astype(str).str.strip(), format="%Y%m%d")
    values = raw.drop(columns="Date").astype(float)
    values = values.mask(values.isin(MISSING_CODES)) / 100
    return pd.concat([raw[["Date"]], values], axis=1)


def combine_us(ff3: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """Join the 3-factor and momentum files on dates present in both; FILE_COLUMNS in decimals."""
    merged = ff3.merge(mom, on="Date", how="inner")
    out = pd.DataFrame({"Date": merged["Date"], "MKT_RF": merged["Mkt-RF"], "SMB": merged["SMB"], "HML": merged["HML"],
                        "MOM": merged["Mom"], "RF": merged["RF"]})
    return out.dropna().reset_index(drop=True)


def write_factors(region: str, table: pd.DataFrame, meta: dict, directory: Path = FACTOR_DIR) -> None:
    """Write data/factors/<region>_daily.csv and update metadata.json for that region."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table[FILE_COLUMNS].to_csv(directory / f"{region}_daily.csv", index=False, date_format="%Y-%m-%d", float_format="%.8f")
    path = directory / METADATA_NAME
    all_meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    all_meta[region] = {**meta, "first_date": f"{table['Date'].min():%Y-%m-%d}", "end_date": f"{table['Date'].max():%Y-%m-%d}",
                        "rows": int(len(table))}
    path.write_text(json.dumps(all_meta, indent=2), encoding="utf-8")


def load_factors(region: str, directory: Path = FACTOR_DIR) -> tuple:
    """(daily factor table indexed by date, metadata) for 'india' or 'us', or (None, None) if not downloaded."""
    directory = Path(directory)
    path = directory / f"{region}_daily.csv"
    if region is None or not path.exists():
        return None, None
    table = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    meta_path = directory / METADATA_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8")).get(region, {}) if meta_path.exists() else {}
    return table, meta


def region_for(tickers) -> str:
    """'india' when every ticker is NSE/BSE, 'us' when none is; None for a mix (not supported without FX)."""
    indian = [str(t).upper().endswith((".NS", ".BO")) for t in tickers]
    if all(indian):
        return "india"
    if not any(indian):
        return "us"
    return None


def excess_returns(returns: pd.Series, factors: pd.DataFrame) -> pd.Series:
    """Returns minus the factor file's daily risk-free rate, on the dates both have."""
    joined = pd.concat([returns.rename("r"), factors["RF"]], axis=1, join="inner").dropna()
    return joined["r"] - joined["RF"]
