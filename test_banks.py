"""
Banks and NBFCs: RBI PCA bands at every edge (RBI/2021-22/118 and RBI/2021-22/139), distance to trigger, point in
time, early warnings, the bundled HDFC Bank and Yes Bank files, and the results-XBRL parser with its scale and ROA
checks on trimmed real filings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import banks as B
import nse_sources as N

PCA = B.load_pca()
BANK, NBFC = PCA["bank"]["indicators"], PCA["nbfc"]["indicators"]
FIXTURES = Path(__file__).parent / "test_data" / "nse"


@pytest.mark.parametrize("value, expected", [(11.5, "none"), (11.49, "RT1"), (9.0, "RT1"), (8.99, "RT2"), (7.5, "RT2"),
                                             (7.49, "RT3")])
def test_bank_crar_bands(value, expected):
    # Indicator 9% + 2.5% CCB = 11.5%; RT1 up to 250 bps below, RT2 250-400 bps, RT3 beyond 400 bps
    assert B.band(value, BANK["crar"])[0] == expected


@pytest.mark.parametrize("value, expected", [(8.625, "none"), (8.62, "RT1"), (7.0, "RT1"), (6.99, "RT2"), (5.5, "RT2"),
                                             (5.49, "RT3")])
def test_bank_cet1_bands(value, expected):
    # Indicator = PST 6.125% + CCB 2.5% = 8.625%; bands 162.5 and 312.5 bps below
    assert B.band(value, BANK["cet1"])[0] == expected


@pytest.mark.parametrize("value, expected", [(5.99, "none"), (6.0, "RT1"), (8.99, "RT1"), (9.0, "RT2"), (12.0, "RT3")])
def test_bank_nnpa_bands(value, expected):
    assert B.band(value, BANK["nnpa"])[0] == expected


@pytest.mark.parametrize("value, d_sib, expected", [(4.0, True, "none"), (3.99, True, "RT1"), (3.49, True, "RT2"),
                                                    (2.99, True, "RT3"), (3.5, False, "none"), (3.49, False, "RT1"),
                                                    (2.99, False, "RT2"), (2.49, False, "RT3")])
def test_bank_leverage_bands_depend_on_dsib(value, d_sib, expected):
    # Minimum 4% for D-SIBs, 3.5% for other banks (RBI/2018-19/225)
    assert B.band(value, BANK["leverage"], d_sib)[0] == expected


@pytest.mark.parametrize("key, value, expected", [("crar", 15.0, "none"), ("crar", 14.9, "RT1"), ("crar", 11.9, "RT2"),
                                                  ("crar", 8.9, "RT3"), ("tier1", 10.0, "none"), ("tier1", 7.9, "RT2"),
                                                  ("nnpa", 6.0, "none"), ("nnpa", 6.01, "RT1"), ("nnpa", 12.01, "RT3")])
def test_nbfc_bands(key, value, expected):
    # NBFC NNPA bands are strict: > 6% (not >= 6%)
    assert B.band(value, NBFC[key])[0] == expected


def test_distance_to_trigger_by_hand():
    status = B.pca_status({"crar": (12.0,), "cet1": (8.0,), "nnpa": (2.5,), "leverage": (4.6,)}, "bank", d_sib=True)
    t = status["table"].set_index("Key")
    assert t.loc["crar", "Distance to trigger (pp)"] == pytest.approx(0.5)
    assert t.loc["cet1", "Distance to trigger (pp)"] == pytest.approx(-0.625) and t.loc["cet1", "Band"] == "RT1"
    assert t.loc["nnpa", "Distance to trigger (pp)"] == pytest.approx(3.5)
    assert t.loc["leverage", "Distance to trigger (pp)"] == pytest.approx(0.6)
    assert status["worst"] == "RT1" and status["min_headroom"] == pytest.approx(-0.625)


def _frame(rows, tmp_path, symbol="TESTBANK"):
    frame = pd.DataFrame(rows, columns=B.COLUMNS)
    frame.to_csv(tmp_path / f"{symbol}.csv", index=False)
    return frame


def test_point_in_time_and_early_warnings(tmp_path):
    rows = [["TESTBANK", "bank", False, "2025-03-31", "gnpa", 2.0, "2025-04-20", "x"],
            ["TESTBANK", "bank", False, "2025-03-31", "crar", 14.0, "2025-04-20", "x"],
            ["TESTBANK", "bank", False, "2026-03-31", "gnpa", 3.5, "2026-04-20", "x"],
            ["TESTBANK", "bank", False, "2026-03-31", "crar", 12.0, "2026-04-20", "x"],
            ["TESTBANK", "bank", False, "2026-03-31", "lcr", 95.0, "2026-04-20", "x"]]
    _frame(rows, tmp_path)
    before = B.assess("TESTBANK", "2026-04-19", tmp_path)  # the 2026 rows are not public yet
    assert before["values"]["crar"][0] == 14.0 and before["warnings"] == []
    after = B.assess("TESTBANK", "2026-04-20", tmp_path)
    assert after["values"]["crar"][0] == 12.0
    joined = " | ".join(after["warnings"])
    assert "within 1 pp of its PCA trigger" in joined and "Gross NPA up 1.50 pp" in joined and "LCR 95%" in joined
    assert not B.assess("NOFILE", None, tmp_path)["available"]


def test_bundled_bank_files():
    hdfc = B.assess("HDFCBANK")
    assert hdfc["entity_type"] == "bank" and hdfc["d_sib"] and hdfc["pca"]["worst"] == "none"
    assert hdfc["values"]["crar"][0] == pytest.approx(19.71) and hdfc["values"]["leverage"][0] == pytest.approx(10.99)
    assert list(hdfc["trend"].columns) == ["FY2022", "FY2023", "FY2024", "FY2025", "FY2026"]
    # Yes Bank as of 31 Dec 2019: the Sep 2019 quarter is public, the Dec quarter (filed March 2020) is not
    yes = B.assess("YESBANK", "2019-12-31")
    assert yes["values"]["gnpa"][0] == pytest.approx(7.39) and yes["values"]["cet1"][0] == pytest.approx(8.4)
    assert yes["pca"]["table"].set_index("Key").loc["cet1", "Band"] == "RT1"
    assert any("Gross NPA up" in w for w in yes["warnings"])
    # Annual-report net NPA matches the XBRL for every period where both exist
    f = B.load_bank("HDFCBANK")
    n = f[f["metric"] == "nnpa"]
    ar = n[n["source"].str.startswith("Annual report")].set_index("period_end")["value_pct"]
    xb = n[n["source"].str.startswith("https://nsearchives")].set_index("period_end")["value_pct"]
    both = ar.index.intersection(xb.index)
    assert len(both) >= 3 and np.allclose(ar[both], xb[both], atol=0.05)


@pytest.mark.parametrize("band, tier", [("none", "Low"), ("RT1", "Elevated"), ("RT2", "High"), ("RT3", "High")])
def test_pca_band_sets_the_event_tier_of_a_bank(band, tier):
    import events as E
    signal = {"pca_band": band, "pca_detail": "CET1 ratio 8.40%", "warnings": []}
    assert E.classify({"merton": signal}, E.load_rules())["tier"] == tier
    warned = E.classify({"merton": {**signal, "pca_band": "none", "warnings": ["Gross NPA up 6.08 pp in a year"]}}, E.load_rules())
    assert warned["tier"] == "Elevated"


def test_results_xbrl_ratios_on_a_clean_filing():
    ratios, note = N.bank_results_ratios((FIXTURES / "bank_HDFCBANK_FY2024_excerpt.xml").read_text(encoding="utf-8"), True)
    assert note == "" and ratios == pytest.approx({"gnpa": 1.24, "nnpa": 0.33, "roa": 1.98})


def test_results_filed_100x_too_small_are_dropped():
    # Yes Bank, Sep 2020: GNPA filed as 0.0017 (0.17%) against a ₹32,344 crore NPA book: an implied 0.09% yield
    ratios, note = N.bank_results_ratios((FIXTURES / "bank_YESBANK_2020-09_excerpt.xml").read_text(encoding="utf-8"), False)
    assert ratios == {} and "yield on advances" in note


def test_an_inconsistent_roa_is_dropped():
    # Yes Bank FY2020: filed ROA +0.05% in a year with a ₹16,418 crore loss (profit ÷ assets = -6.4%)
    ratios, note = N.bank_results_ratios((FIXTURES / "bank_YESBANK_FY2020_excerpt.xml").read_text(encoding="utf-8"), True)
    assert "roa" not in ratios and "ROA dropped" in note and "-6.37%" in note
    assert ratios["gnpa"] == pytest.approx(16.80)
