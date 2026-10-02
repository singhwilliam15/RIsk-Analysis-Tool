"""
Download the daily factor files and write data/factors/ (see factor_data.py).

    python scripts/refresh_factor_data.py --iima-release 2025-12

--iima-release is the YYYY-MM prefix of the current IIM Ahmedabad file (shown on its page). If any download or
parse fails, nothing is written for that region and the manual steps are printed: the app never uses partial or
made-up factor data.
"""

import argparse
import io
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import factor_data as F  # noqa: E402

HEADERS = {"User-Agent": "Mozilla/5.0 (Risk Analysis Tool factor refresh)"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return response.read()


def unzip_text(blob: bytes) -> str:
    archive = zipfile.ZipFile(io.BytesIO(blob))
    return archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")


def refresh_india(release: str) -> None:
    url = F.SOURCES["india"]["file_pattern"].format(release=release)
    table = F.parse_iima(fetch(url).decode("utf-8", errors="replace"))
    F.write_factors("india", table, {"source": F.SOURCES["india"]["name"], "url": url, "page": F.IIMA_PAGE,
                                     "release": release, "downloaded_on": f"{date.today():%Y-%m-%d}",
                                     "citation": F.SOURCES["india"]["citation"], "survivorship_bias_adjusted": True})
    print(f"india: {len(table)} days, {table['Date'].min():%Y-%m-%d} to {table['Date'].max():%Y-%m-%d}")


def refresh_us() -> None:
    src = F.SOURCES["us"]
    table = F.combine_us(F.parse_french(unzip_text(fetch(src["ff3"]))), F.parse_french(unzip_text(fetch(src["mom"]))))
    F.write_factors("us", table, {"source": src["name"], "url": f"{src['ff3']} ; {src['mom']}", "page": F.FRENCH_PAGE,
                                  "downloaded_on": f"{date.today():%Y-%m-%d}", "citation": src["citation"]})
    print(f"us: {len(table)} days, {table['Date'].min():%Y-%m-%d} to {table['Date'].max():%Y-%m-%d}")


MANUAL = {
    "india": f"""Download the Indian factors by hand:
  1. Open {F.IIMA_PAGE}
  2. Under the daily four-factor returns, download the SURVIVORSHIP-BIAS-ADJUSTED file
     (named like YYYY-MM_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv).
  3. Run: python -c "import factor_data as F, pandas as pd; t = F.parse_iima(open('<file>').read()); F.write_factors('india', t, {{'source': 'IIMA (manual download)', 'url': '<file URL>', 'downloaded_on': '<YYYY-MM-DD>', 'citation': F.SOURCES['india']['citation']}})" """,
    "us": f"""Download the US factors by hand:
  1. Open {F.FRENCH_PAGE}
  2. Download "Fama/French 3 Factors [Daily]" CSV and "Momentum Factor (Mom) [Daily]" CSV and unzip both.
  3. Run: python -c "import factor_data as F; t = F.combine_us(F.parse_french(open('<ff3 csv>').read()), F.parse_french(open('<mom csv>').read())); F.write_factors('us', t, {{'source': 'Kenneth R. French data library (manual download)', 'url': '<URLs>', 'downloaded_on': '<YYYY-MM-DD>', 'citation': F.SOURCES['us']['citation']}})" """,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iima-release", required=True, help="YYYY-MM prefix of the current IIMA file, e.g. 2025-12")
    parser.add_argument("--only", choices=("india", "us"))
    args = parser.parse_args()
    failed = []
    for region, run in (("india", lambda: refresh_india(args.iima_release)), ("us", refresh_us)):
        if args.only and region != args.only:
            continue
        try:
            run()
        except Exception as exc:
            failed.append(region)
            print(f"\n{region}: FAILED ({exc}). Nothing was written for {region}.\n{MANUAL[region]}\n", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
