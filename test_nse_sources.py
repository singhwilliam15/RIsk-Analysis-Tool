"""
NSE parsers on real downloads (trimmed excerpts in test_data/nse/, saved 2 Oct 2026): shareholding XBRL → pledge
rows, ASM/GSM lists → surveillance rows, credit-rating disclosures → rating rows; and the agency default-rate
lookup and DD percentile used by the Credit page.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import credit as C
import disclosures as D
import events as E
import nse_sources as N

FIXTURES = Path(__file__).parent / "test_data" / "nse"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_shareholding_xbrl_gives_jaiprakash_powers_pledge():
    parsed = N.shareholding_pledge(fixture("shp_JPPOWER_2026-06-30_excerpt.xml"))
    # 30 Jun 2026 filing: promoters hold 24%; 1,200,509,465 of their shares are pledged (72.99% of theirs, 17.52% of all)
    assert parsed["promoter_holding_pct"] == pytest.approx(24.0)
    assert parsed["pledged_shares"] == 1_200_509_465
    assert parsed["pledged_pct_of_promoter"] == pytest.approx(72.99)
    assert parsed["pledged_pct_of_total"] == pytest.approx(17.52)
    assert parsed["pledged_shares"] / parsed["total_shares"] == pytest.approx(0.1752, abs=5e-5)
    # Shares under non-disposal undertakings (102,188,566) are not pledges and are left out
    assert parsed["pledged_shares"] != 1_302_698_031


def test_pledge_row_uses_the_broadcast_date_and_loads_cleanly():
    record = json.loads(fixture("shp_master_JPPOWER_record.json"))
    row = N.pledge_row("JPPOWER", record, N.shareholding_pledge(fixture("shp_JPPOWER_2026-06-30_excerpt.xml")))
    assert row["quarter_end"] == "2026-06-30" and row["disclosure_date"] == "2026-07-13"
    clean, errors, _ = D.load_file(pd.DataFrame([row]).to_csv(index=False), "pledges")
    assert not errors and clean.iloc[0]["pledged_pct_of_promoter"] == pytest.approx(72.99)
    # Point in time: not public on 12 Jul, public on 13 Jul
    assert D.as_of(clean, "pledges", "2026-07-12").empty and len(D.as_of(clean, "pledges", "2026-07-13")) == 1


def test_a_filing_stating_percentages_in_percent_reads_the_same():
    # The 30 Jun 2025 filing uses the older form: total shareholding "100.00" and promoters "24.00", not 1 and 0.24
    old = N.shareholding_pledge(fixture("shp_JPPOWER_2025-06-30_percent_form_excerpt.xml"))
    assert old["promoter_holding_pct"] == pytest.approx(24.0)
    assert old["pledged_pct_of_promoter"] == pytest.approx(72.99) and old["pledged_pct_of_total"] == pytest.approx(17.52)


def test_a_filing_without_pledges_reads_as_zero():
    parsed = N.shareholding_pledge(next(fixture(p.name) for p in FIXTURES.glob("shp_TCS_*_excerpt.xml")))
    assert parsed["promoter_holding_pct"] > 50
    assert parsed["pledged_pct_of_promoter"] == 0 and parsed["pledged_shares"] == 0


def test_an_unreadable_filing_is_skipped_not_guessed():
    assert N.shareholding_pledge("<xbrli:xbrl></xbrli:xbrl>") is None


def test_a_company_without_a_promoter_group_reads_as_zero():
    # Only the total-shareholding facts, as in HDFC Bank's or Coforge's filings (no promoter context at all)
    text = ('<in-bse-shp:NumberOfShares contextRef="ShareholdingPattern_ContextI" unitRef="shares">1000</in-bse-shp:NumberOfShares>'
            '<in-bse-shp:DateOfReport contextRef="MainI">2026-06-30</in-bse-shp:DateOfReport>')
    parsed = N.shareholding_pledge(text)
    assert parsed["promoter_holding_pct"] == 0 and parsed["pledged_shares"] == 0 and parsed["total_shares"] == 1000


def test_surveillance_lists_become_template_rows():
    rows = N.surveillance_rows(json.loads(fixture("asm_excerpt.json")), json.loads(fixture("gsm_excerpt.json")), "2026-10-02")
    assert set(rows["measure"]) == {"ASM-LT", "ASM-ST", "GSM"}
    assert (rows["date_in"] == "2026-10-02").all() and (rows["date_out"] == "").all()
    gsm = rows[rows["measure"] == "GSM"]["stage"].tolist()
    assert "Stage 0" in gsm and any(s.startswith("Stage ") and s != "Stage 0" for s in gsm)
    clean, errors, _ = D.load_file(rows.to_csv(index=False), "surveillance")
    assert not errors and len(clean) == len(rows)


@pytest.mark.parametrize("desc, stage", [("ASM IBC Stage I and GSM Stage 0", "Stage 0"),
                                         ("Graded Surveillance Measure - Stage VI", "Stage VI"),
                                         ("Shortlisted under Graded Surveillance Measure", "Shortlisted under Graded Surveillance Measure")])
def test_gsm_stage_is_read_from_the_description(desc, stage):
    assert N._gsm_stage(desc) == stage


def test_credit_rating_disclosures_become_rating_rows():
    records = json.loads(fixture("credit_rating_excerpt.json"))
    rows, skipped = N.rating_rows(records, {"INE002A": "RELIANCE"})
    assert len(rows) >= 1 and (rows["symbol"] == "RELIANCE").all()
    assert set(rows["action"]) <= set(D.RATING_ACTIONS) and skipped == 0
    clean, errors, _ = D.load_file(rows.to_csv(index=False), "ratings")
    assert not errors and len(clean) == len(rows)


@pytest.mark.parametrize("record, action", [
    ({"RatingAction": "New"}, "assigned"),
    ({"RatingAction": "Reaffirm"}, "reaffirmed"),
    ({"RatingAction": "Other", "SpecifyOthRatingActn": "Rating Downgraded"}, "downgraded"),
    ({"RatingAction": "Other", "CreditRating": "CARE BB+; Rating Watch with Negative Implications"}, "placed on watch"),
    ({"RatingAction": "Other", "CreditRating": "IND A", "CreditRatingEarlier": "IND A+"}, "downgraded"),
    ({"RatingAction": "Other", "CreditRating": "IND AA", "CreditRatingEarlier": "IND A+"}, "upgraded"),
    ({"RatingAction": "Other", "CreditRating": "IND AA"}, None),  # not readable: skipped, never guessed
])
def test_rating_action_mapping(record, action):
    assert N.rating_action(record) == action


@pytest.mark.parametrize("rating, rank", [("CRISIL AAA/Stable", 0), ("[ICRA]A+(Negative)", 4), ("IND AA+", 1),
                                          ("CARE BB+", 10), ("CRISIL A1+", -1), ("[ICRA]A4+", -1), ("CARE D", 19)])
def test_short_term_ratings_are_not_read_as_long_term_grades(rating, rank):
    assert E.rating_rank(rating) == rank


def test_rating_implied_pd_from_the_published_studies():
    table = C.load_default_rates()
    crisil = C.rating_implied_pd("CRISIL BBB-/Stable", "CRISIL", table)
    assert crisil["category"] == "BBB" and crisil["pd"] == pytest.approx(0.0046) and not crisil["fallback"]
    assert crisil["page"] == 10
    icra = C.rating_implied_pd("[ICRA]A+(Negative)", "ICRA", table)
    assert icra["pd"] == pytest.approx(0.002) and icra["agency_used"] == "ICRA"
    # No CARE study on file: CRISIL's is used and flagged
    care = C.rating_implied_pd("CARE AA", "CARE", table)
    assert care["fallback"] and care["agency_used"] == "CRISIL" and care["pd"] == pytest.approx(0.0005)
    assert C.rating_implied_pd("CARE D", "CARE", table)["pd"] == 1.0
    assert C.rating_implied_pd("CRISIL A1+", "CRISIL", table) is None
    # The studies are ordinal: a lower grade never has a lower default rate
    for agency in ("CRISIL", "ICRA"):
        rates = table[table["agency"] == agency].set_index("rating_category")["one_year_default_rate_pct"]
        assert rates.loc[["AAA", "AA", "A", "BBB", "BB", "B", "C"]].is_monotonic_increasing


def test_dd_percentile():
    universe = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    assert C.dd_percentile(0.5, universe) == 0.0
    assert C.dd_percentile(5.0, universe) == 1.0
    assert C.dd_percentile(2.0, universe) == pytest.approx(0.375)  # 1 below, 1 tied
    assert np.isnan(C.dd_percentile(np.nan, universe)) and np.isnan(C.dd_percentile(2.0, pd.Series(dtype=float)))


def test_rating_status_prefers_the_long_term_scale():
    actions = pd.DataFrame({"symbol": ["X", "X"], "agency": ["CRISIL", "CRISIL"], "rating": ["CRISIL AA+/Stable", "CRISIL A1+"],
                            "action": ["reaffirmed", "reaffirmed"], "action_date": ["2026-01-10", "2026-03-10"]})
    assert C.rating_status(actions, "X")["rating"] == "CRISIL AA+/Stable"


def test_as_of_stamp():
    files = pd.DataFrame({"Dataset": ["Price bands", "Promoter holding and pledges", "Price bands"],
                          "As of": ["2026-10-01", "2026-06-30", "2026-10-02"]})
    assert D.as_of_stamp(files) == "Data as of: Price bands 02 Oct 2026 · Promoter holding and pledges 30 Jun 2026"
    assert D.as_of_stamp(pd.DataFrame()) == ""


def test_the_40_percent_band_in_nses_file_is_accepted():
    clean, errors, _ = D.load_file("Symbol,Series,Security Name,Band,Remarks\nXYZ,EQ,XYZ LTD,40,-\n", "price_bands", as_of="2026-10-02")
    assert not errors and clean.iloc[0]["band_pct"] == 40.0
