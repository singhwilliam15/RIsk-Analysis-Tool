"""
Disclosure loaders and validators. The sample files below follow the layouts of NSE's downloads (headers in
capitals, DD-MON-YYYY dates, fo_secban's one-line header) but contain made-up symbols and figures: they test
the parsers and are never used as data.
"""

import json

import numpy as np
import pandas as pd
import pytest

import disclosures as D

PLEDGES_OFFICIAL = """SYMBOL,QUARTER,PROMOTER HOLDING (%),PROMOTER SHARES ENCUMBERED (% OF PROMOTER SHARES),PROMOTER SHARES ENCUMBERED (% OF TOTAL SHARES),BROADCAST DATE
TESTA,30-Jun-2026,55.20,40.00,22.08,15-Jul-2026
TESTB,30-Jun-2026,70.00,0.00,0.00,
"""

ASM_OFFICIAL = """Sr No,Symbol,Security Name,ISIN,Stage
1,TESTA,Test A Limited,INE000A01011,Stage I
2,TESTC,Test C Limited,INE000C01013,Stage II
"""

PRICE_BANDS_OFFICIAL = """Symbol,Series,Security Name,Band,Remarks
TESTA,EQ,Test A Limited,5,
TESTB,EQ,Test B Limited,No Band,
TESTC,BE,Test C Limited,20,
"""

FO_BAN_OFFICIAL = """Securities in Ban For Trade Date 02-OCT-2026:
1,TESTA
2,TESTD
"""

RATINGS_TEMPLATE = """symbol,agency,instrument,rating,outlook,action,action_date,source
TESTA.NS,CRISIL,Long-term bank facilities,CRISIL A,Negative,Downgraded,2026-08-14,https://example.org/rationale
"""

AUDITOR_TEMPLATE = """symbol,event_type,auditor,event_date,details,source
TESTA,Resignation,Test & Co,2026-09-01,Resigned citing lack of information,https://example.org/announcement
"""


def test_pledges_official_layout():
    frame, errors, _ = D.load_file(PLEDGES_OFFICIAL, "pledges")
    assert errors == []
    assert frame["symbol"].tolist() == ["TESTA", "TESTB"]
    assert frame.loc[0, "quarter_end"] == pd.Timestamp("2026-06-30")
    assert frame.loc[0, "pledged_pct_of_promoter"] == 40.0 and frame.loc[0, "pledged_pct_of_total"] == pytest.approx(22.08)
    assert frame.loc[0, "disclosure_date"] == pd.Timestamp("2026-07-15") and pd.isna(frame.loc[1, "disclosure_date"])


def test_asm_list_takes_its_date_and_measure_from_the_manifest():
    frame, errors, notes = D.load_file(ASM_OFFICIAL, "surveillance", as_of="2026-10-01", constants={"measure": "ASM-LT"})
    assert errors == []
    assert frame["symbol"].tolist() == ["TESTA", "TESTC"] and frame["stage"].tolist() == ["Stage I", "Stage II"]
    assert (frame["date_in"] == pd.Timestamp("2026-10-01")).all() and (frame["measure"] == "ASM-LT").all()
    assert any("as-of date" in n for n in notes) and any("measure" in n for n in notes)
    # Without them the required columns are missing
    _, errors, _ = D.load_file(ASM_OFFICIAL, "surveillance")
    assert errors and "measure" in errors[0] and "date_in" in errors[0]


def test_price_bands_official_layout():
    frame, errors, _ = D.load_file(PRICE_BANDS_OFFICIAL, "price_bands", as_of="2026-10-02")
    assert errors == []
    assert frame["band"].tolist() == ["5%", "No Band", "20%"]
    assert frame["band_pct"].tolist()[0] == 5.0 and np.isnan(frame["band_pct"].tolist()[1])
    assert frame["series"].tolist() == ["EQ", "EQ", "BE"]


def test_fo_ban_official_layout():
    frame, errors, notes = D.load_file(FO_BAN_OFFICIAL, "fo_ban")
    assert errors == [] and "official NSE fo_secban" in notes[0]
    assert frame["symbol"].tolist() == ["TESTA", "TESTD"]
    assert (frame["trade_date"] == pd.Timestamp("2026-10-02")).all()


def test_fo_ban_with_no_securities():
    frame, errors, _ = D.load_file("Securities in Ban For Trade Date 05-OCT-2026: NIL\n", "fo_ban")
    assert errors == [] and frame.empty


def test_ratings_and_auditor_templates():
    ratings, errors, _ = D.load_file(RATINGS_TEMPLATE, "ratings")
    assert errors == [] and ratings.loc[0, "symbol"] == "TESTA" and ratings.loc[0, "action"] == "downgraded"
    auditor, errors, _ = D.load_file(AUDITOR_TEMPLATE, "auditor_events")
    assert errors == [] and auditor.loc[0, "event_type"] == "resignation"


@pytest.mark.parametrize("dataset, text, message", [
    ("pledges", "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter\nX,2026-06-30,120,10\n", "between 0 and 100"),
    ("pledges", "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter,pledged_pct_of_total\nX,2026-06-30,30,50,40\n",
     "exceeds the promoter holding"),
    ("pledges", "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter\nX,soon,30,10\n", "is not a date"),
    ("pledges", "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter\n,2026-06-30,30,10\n", "symbol is blank"),
    ("surveillance", "symbol,measure,stage,date_in\nX,ASM-XX,I,2026-01-01\n", "must be one of"),
    ("surveillance", "symbol,measure,stage,date_in,date_out\nX,GSM,I,2026-02-01,2026-01-01\n", "date_out is before date_in"),
    ("price_bands", "symbol,band,effective_date\nX,15,2026-10-01\n", "must be 2, 5, 10, 20"),
    ("ratings", "symbol,agency,instrument,rating,action,action_date\nX,ICRA,NCD,AA,improved,2026-01-01\n", "must be one of"),
])
def test_invalid_rows_are_rejected_with_the_row_number(dataset, text, message):
    frame, errors, _ = D.load_file(text, dataset)
    assert frame.empty
    assert len(errors) == 1 and "row 2" in errors[0] and message in errors[0]


def test_valid_rows_survive_next_to_invalid_ones():
    text = "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter\nX,2026-06-30,30,10\nY,2026-06-30,30,101\n"
    frame, errors, _ = D.load_file(text, "pledges")
    assert frame["symbol"].tolist() == ["X"] and len(errors) == 1 and "row 3" in errors[0]


def test_symbols_lose_the_yahoo_suffix():
    assert D.normalise_symbol(" reliance.ns ") == "RELIANCE"
    assert D.normalise_symbol("TATAMOTORS.BO") == "TATAMOTORS"
    assert D.normalise_symbol("M&M") == "M&M"


def test_point_in_time_filter():
    frame, _, _ = D.load_file(PLEDGES_OFFICIAL, "pledges")
    # TESTA disclosed on 15 Jul; TESTB has no disclosure date, so it is public at quarter end + 21 days = 21 Jul
    assert D.public_date(frame, "pledges").tolist() == [pd.Timestamp("2026-07-15"), pd.Timestamp("2026-07-21")]
    assert D.as_of(frame, "pledges", "2026-07-14").empty
    assert D.as_of(frame, "pledges", "2026-07-15")["symbol"].tolist() == ["TESTA"]
    assert D.as_of(frame, "pledges", "2026-07-21")["symbol"].tolist() == ["TESTA", "TESTB"]
    ban, _, _ = D.load_file(FO_BAN_OFFICIAL, "fo_ban")
    assert D.as_of(ban, "fo_ban", "2026-10-01").empty and len(D.as_of(ban, "fo_ban", "2026-10-02")) == 2


def _entry(file, dataset, **extra):
    return {"file": file, "dataset": dataset, "source_url": "https://www.nseindia.com/", "downloaded_on": "2026-10-02",
            "coverage_start": "2026-10-02", "coverage_end": "2026-10-02", **extra}


def test_load_disclosures_from_a_manifest(tmp_path):
    (tmp_path / "ban.csv").write_text(FO_BAN_OFFICIAL)
    (tmp_path / "asm.csv").write_text(ASM_OFFICIAL)
    (tmp_path / "stray.csv").write_text(RATINGS_TEMPLATE)
    entries = [_entry("ban.csv", "fo_ban"), _entry("asm.csv", "surveillance", constants={"measure": "ASM-ST"}),
               _entry("gone.csv", "pledges"), {"file": "x.csv", "dataset": "pledges"},
               _entry("y.csv", "dividends")]
    (tmp_path / D.MANIFEST_NAME).write_text(json.dumps({"files": entries}))
    result = D.load_disclosures(tmp_path)
    ban = result["data"]["fo_ban"]
    assert ban["symbol"].tolist() == ["TESTA", "TESTD"]
    assert (ban["source_url"] == "https://www.nseindia.com/").all() and (ban["file"] == "ban.csv").all()
    assert result["data"]["surveillance"]["measure"].tolist() == ["ASM-ST", "ASM-ST"]
    assert result["data"]["pledges"].empty
    assert result["files"]["File"].tolist() == ["ban.csv", "asm.csv"]
    issues = " | ".join(result["issues"])
    assert "gone.csv: listed in the manifest but not found" in issues
    assert "missing source_url" in issues and "unknown dataset 'dividends'" in issues
    assert "stray.csv: not in manifest.json" in issues


def test_bad_manifest_json(tmp_path):
    (tmp_path / D.MANIFEST_NAME).write_text("{not json")
    assert "not valid JSON" in D.load_disclosures(tmp_path)["issues"][0]


def test_shipped_folder_loads_cleanly():
    result = D.load_disclosures()
    assert result["issues"] == [] and result["files"].empty
    assert all(frame.empty for frame in result["data"].values())


def test_templates_match_the_schemas(tmp_path):
    paths = D.write_templates(tmp_path)
    assert len(paths) == len(D.DATASETS)
    for name, spec in D.DATASETS.items():
        assert pd.read_csv(tmp_path / f"{name}_template.csv").columns.tolist() == spec.column_names
        shipped = D.DISCLOSURE_DIR / "templates" / f"{name}_template.csv"
        assert pd.read_csv(shipped).columns.tolist() == spec.column_names
