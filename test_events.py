"""Event pillar: signals as of a date, tier rules at their thresholds, margin calls by hand, and the mixture ES
against a large simulation."""

import numpy as np
import pandas as pd
import pytest

import disclosures as D
import events as E

RULES = E.load_rules()


def frame(dataset, text):
    clean, errors, _ = D.load_file(text, dataset)
    assert errors == []
    return clean


# ---------------------------------------------------------------
# Signals, point in time
# ---------------------------------------------------------------

PLEDGES = frame("pledges", "symbol,quarter_end,promoter_holding_pct,pledged_pct_of_promoter,disclosure_date\n"
                           "ABC,2025-03-31,50,10,2025-04-15\nABC,2025-06-30,50,15,\nABC,2026-03-31,50,35,2026-04-18\n"
                           "ABC,2026-06-30,50,55,2026-07-20\n")


def test_pledge_signal_uses_only_public_rows():
    assert E.pledge_signal(PLEDGES, "ABC", "2026-07-19")["pledged_pct_of_promoter"] == 35  # Q1 FY27 not filed yet
    latest = E.pledge_signal(PLEDGES, "ABC", "2026-07-20")
    assert latest["pledged_pct_of_promoter"] == 55 and latest["change_4q_pp"] == pytest.approx(40)  # vs Jun 2025
    # The June 2025 row has no disclosure date: public at quarter end + 21 days
    assert E.pledge_signal(PLEDGES, "ABC", "2025-07-20")["quarter_end"] == pd.Timestamp("2025-03-31")
    assert E.pledge_signal(PLEDGES, "ABC", "2025-07-21")["quarter_end"] == pd.Timestamp("2025-06-30")
    assert E.pledge_signal(PLEDGES, "XYZ", "2026-12-31") is None
    assert E.pledge_signal(pd.DataFrame(), "ABC", "2026-12-31") is None


def test_surveillance_in_force_on_the_date():
    surv = frame("surveillance", "symbol,measure,stage,date_in,date_out\nABC,ASM-ST,Stage I,2026-01-05,2026-02-05\n"
                                 "ABC,ASM-LT,Stage III,2026-03-01,\nDEF,GSM,Stage 0,2026-01-01,\n")
    assert E.surveillance_signal(surv, "ABC", "2026-01-10")["measure"] == "ASM-ST"
    assert E.surveillance_signal(surv, "ABC", "2026-02-10")["measure"] is None  # exited, not yet on ASM-LT
    later = E.surveillance_signal(surv, "ABC", "2026-04-01")
    assert later["measure"] == "ASM-LT" and later["stage"] == 3
    assert E.stage_number("Stage IV") == 4 and E.stage_number("2") == 2 and E.stage_number("?") == 0


def test_fo_ban_window():
    bans = frame("fo_ban", "Securities in Ban For Trade Date 01-SEP-2026:\n1,ABC\n")
    assert E.fo_ban_signal(bans, "ABC", "2026-09-20")["last_ban"] == pd.Timestamp("2026-09-01")
    assert E.fo_ban_signal(bans, "ABC", "2026-10-15")["last_ban"] is None  # more than 30 days ago
    assert E.fo_ban_signal(bans, "ABC", "2026-08-31")["last_ban"] is None  # not yet happened
    assert E.fo_ban_signal(pd.DataFrame(), "ABC", "2026-09-20") is None


def test_rating_scale_and_signal():
    assert E.rating_rank("CRISIL AA+") == 1 and E.rating_rank("CARE BBB- (CE)") == 9 and E.rating_rank("IND D") == 19
    assert E.rating_rank("[ICRA]A(Stable)") == 5 and E.rating_rank("no rating") == -1
    ratings = frame("ratings", "symbol,agency,instrument,rating,outlook,action,action_date\n"
                               "ABC,CRISIL,NCD,CRISIL A,Stable,downgraded,2026-01-10\n"
                               "ABC,CRISIL,NCD,CRISIL BB+,Negative,downgraded,2026-06-10\n"
                               "ABC,ICRA,Loans,ICRA A-,Negative,placed on watch,2026-05-01\n")
    s = E.rating_signal(ratings, "ABC", "2026-07-01")
    assert s["downgrades"] == 2 and s["below_investment_grade"] and s["negative_watch"] and s["latest"] == "CRISIL BB+"
    assert E.rating_signal(ratings, "ABC", "2026-02-01")["downgrades"] == 1
    assert E.rating_signal(ratings, "ABC", "2027-07-01")["downgrades"] == 0  # outside 12 months


def test_auditor_and_dd_signals():
    aud = frame("auditor_events", "symbol,event_type,event_date\nABC,Resignation,2025-03-01\nABC,Qualified opinion,2026-05-30\n")
    assert E.auditor_signal(aud, "ABC", "2026-06-01")["events"] == ["qualified opinion", "resignation"]
    assert E.auditor_signal(aud, "ABC", "2027-04-01")["events"] == ["qualified opinion"]  # resignation > 24 months ago
    rolling = pd.DataFrame({"Date": pd.date_range("2025-01-31", periods=18, freq="ME"), "DD": np.linspace(6, 2.4, 18)})
    d = E.dd_signal(rolling, "2026-06-30")
    assert d["dd"] == pytest.approx(2.4) and d["dd_6m_ago"] == pytest.approx(rolling["DD"].iloc[-7])
    assert d["fall_6m"] == pytest.approx((d["dd_6m_ago"] - 2.4) / d["dd_6m_ago"])
    assert E.dd_signal(rolling, "2024-12-31") is None


# ---------------------------------------------------------------
# Tier rules
# ---------------------------------------------------------------

def pledge(pct, change=np.nan):
    return {"pledged_pct_of_promoter": pct, "change_4q_pp": change}


@pytest.mark.parametrize("signals, tier", [
    ({}, E.LOW),
    ({"pledge": pledge(19.9)}, E.LOW),
    ({"pledge": pledge(20)}, E.ELEVATED),
    ({"pledge": pledge(49.9)}, E.ELEVATED),
    ({"pledge": pledge(50)}, E.HIGH),
    ({"pledge": pledge(5, change=10)}, E.ELEVATED),
    ({"surveillance": {"measure": "ASM-ST", "stage": 1, "stage_text": "Stage I"}}, E.ELEVATED),
    ({"surveillance": {"measure": "ASM-LT", "stage": 2, "stage_text": "Stage II"}}, E.ELEVATED),
    ({"surveillance": {"measure": "ASM-LT", "stage": 3, "stage_text": "Stage III"}}, E.HIGH),
    ({"surveillance": {"measure": "GSM", "stage": 0, "stage_text": "Stage 0"}}, E.HIGH),
    ({"surveillance": {"measure": None, "stage": 0}}, E.LOW),
    ({"fo_ban": {"last_ban": pd.Timestamp("2026-09-01"), "bans_in_window": 1}}, E.ELEVATED),
    ({"rating": {"downgrades": 1, "negative_watch": False, "below_investment_grade": False}}, E.ELEVATED),
    ({"rating": {"downgrades": 1, "negative_watch": False, "below_investment_grade": True}}, E.HIGH),
    ({"rating": {"downgrades": 2, "negative_watch": False, "below_investment_grade": False}}, E.HIGH),
    ({"rating": {"downgrades": 0, "negative_watch": True, "below_investment_grade": False}}, E.ELEVATED),
    ({"auditor": {"events": ["qualified opinion"]}}, E.ELEVATED),
    ({"auditor": {"events": ["resignation"]}}, E.HIGH),
    ({"auditor": {"events": ["emphasis of matter"]}}, E.LOW),
    ({"merton": {"dd": 1.49, "fall_6m": 0.0}}, E.HIGH),
    ({"merton": {"dd": 2.99, "fall_6m": 0.0}}, E.ELEVATED),
    ({"merton": {"dd": 5.0, "fall_6m": 0.30}}, E.ELEVATED),
    ({"merton": {"dd": 5.0, "fall_6m": 0.29}}, E.LOW),
    ({"circuit": {"lower_circuit_days": 3, "longest_run": 2}}, E.ELEVATED),
    ({"circuit": {"lower_circuit_days": 2, "longest_run": 2}}, E.LOW),
    ({"group": {"group": "Tata", "same_group_holdings": 2}}, E.ELEVATED),
    ({"group": {"group": "Tata", "same_group_holdings": 1}}, E.LOW),
])
def test_tier_rules_at_their_thresholds(signals, tier):
    assert E.classify(signals, RULES)["tier"] == tier


def test_missing_signals_are_counted_not_treated_as_safe():
    result = E.classify({"merton": {"dd": 8.0, "fall_6m": 0.0}, "circuit": {"lower_circuit_days": 0, "longest_run": 0}}, RULES)
    assert result["tier"] == E.LOW and result["basis"] == "based on 2 of 8 signals"
    assert result["available"] == ["merton", "circuit"]
    high = E.classify({"pledge": pledge(60), "auditor": {"events": ["qualified opinion"]}}, RULES)
    assert high["tier"] == E.HIGH and len(high["fired"]) == 2  # every rule that fired is listed


# ---------------------------------------------------------------
# Margin calls
# ---------------------------------------------------------------

def test_margin_call_by_hand():
    mc = E.margin_call(1_000_000, 100.0, 200_000, initial_cover=2.0, trigger_cover=1.5)
    assert mc["trigger_fall"] == pytest.approx(0.25) and mc["trigger_price"] == pytest.approx(75.0)
    assert mc["loan"] == pytest.approx(50_000_000) and mc["days_all"] == pytest.approx(5.0)
    # Sell x at ₹75 so that (1,000,000 − x)·75 / (50,000,000 − 75x) = 2  →  x = 333,333
    assert mc["shares_to_restore"] == pytest.approx(1_000_000 / 3)
    x = mc["shares_to_restore"]
    assert (1_000_000 - x) * 75 / (50_000_000 - 75 * x) == pytest.approx(2.0)
    none = E.margin_call(np.nan, 100.0, 200_000)
    assert none["trigger_fall"] == pytest.approx(0.25) and np.isnan(none["days_all"])


# ---------------------------------------------------------------
# Jump overlay
# ---------------------------------------------------------------

BASE = {"loc": 0.0005, "scale": 0.012, "nu": 5.0}


def test_without_jumps_the_mixture_is_the_plain_student_t():
    from garch import standardized_t_es, standardized_t_quantile
    res = E.event_adjusted_es(BASE, {"A": 1.0}, {"A": (0.0, -0.2)}, 0.975)
    assert res["gap"] == 0 and res["event_es"] == pytest.approx(res["es"])
    # Same figure as the GARCH-t closed form with σ = scale·√(ν/(ν−2))
    sigma = BASE["scale"] * np.sqrt(5 / 3)
    assert res["var"] == pytest.approx(-(BASE["loc"] + sigma * standardized_t_quantile(0.025, 5)))
    assert res["es"] == pytest.approx(sigma * standardized_t_es(0.025, 5) - BASE["loc"])


def test_mixture_es_matches_a_large_simulation():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    jumps = {"A": (0.005, -0.2), "B": (0.01, -0.1), "C": (0.0, 0.0)}
    res = E.event_adjusted_es(BASE, weights, jumps, 0.95)
    rng = np.random.default_rng(0)
    n = 2_000_000
    x = BASE["loc"] + BASE["scale"] * rng.standard_t(5, n)
    for t, (p, j) in jumps.items():
        x += weights[t] * j * (rng.random(n) < p)
    q = np.quantile(x, 0.05)
    assert res["event_var"] == pytest.approx(-q, rel=0.005)
    assert res["event_es"] == pytest.approx(-x[x <= q].mean(), rel=0.005)
    assert sum(res["by_holding"].values()) == pytest.approx(res["gap"])
    assert res["by_holding"]["A"] > res["by_holding"]["B"] > 0 and "C" not in res["by_holding"]


def test_jump_states_are_exact_and_sum_to_one():
    states = E.jump_states({"A": 1.0, "B": 1.0}, {"A": (0.1, -0.1), "B": (0.2, -0.2)})
    probs = dict((round(s, 6), p) for p, s in states)
    assert probs[0.0] == pytest.approx(0.9 * 0.8) and probs[-0.3] == pytest.approx(0.1 * 0.2)
    assert sum(p for p, _ in states) == pytest.approx(1.0)
    many = {f"T{i}": (0.01, -0.1) for i in range(20)}
    big = E.jump_states({t: 0.05 for t in many}, many)
    assert len(big) == 1 + 20 + 190 and sum(p for p, _ in big) == pytest.approx(1.0)  # up to two jumps beyond 12


def test_base_distribution_prefers_garch_and_falls_back():
    from var_calculator import calculate_all_var
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.standard_t(5, 1500) * 0.01)
    points = calculate_all_var(returns, 1.0, 0.95, 1, num_simulations=1000)
    base = E.base_distribution(points)
    if not points["GARCH(1,1)-t"].get("fallback"):
        assert base["model"] == "GARCH(1,1)-t"
        es = E.event_adjusted_es(base, {"X": 1.0}, {"X": (0.0, 0.0)}, 0.95)["es"]
        assert es == pytest.approx(points["GARCH(1,1)-t"]["cvar_daily_pct"], rel=1e-6)
    fallback = E.base_distribution({**points, "GARCH(1,1)-t": {"fallback": True}})
    assert fallback["model"].startswith("Student-t") and fallback["nu"] == points["Student-t"]["degrees_of_freedom"]


def test_excel_events_sheet():
    import io
    import openpyxl
    from excel_exporter import generate_excel_var_report
    from trust import TrustedMetric
    from var_calculator import backtest_all_methods, calculate_all_var, historical_worst_losses, rolling_var_forecasts

    rng = np.random.default_rng(1)
    r = pd.Series(rng.standard_t(5, 600) * 0.01)
    df = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=600), "Close": 100 * np.cumprod(1 + r), "Returns": r,
                       "Log_Returns": np.log1p(r), "Rolling_30d_Vol": r.rolling(30).std()})
    points = calculate_all_var(r, 1e6, 0.95, 1, num_simulations=1000)
    overlay = E.event_adjusted_es(BASE, {"ABC.NS": 1.0}, {"ABC.NS": (0.005, -0.2)}, 0.95)
    panel = pd.DataFrame({"Ticker": ["ABC.NS"], "Tier": ["High"], "Basis": ["based on 3 of 8 signals"],
                          "Reasons": ["60.0% of promoter shares pledged"]})
    xlsx = generate_excel_var_report(
        "ABC.NS", "Test", "INR", 1e6, 0.95, 1, df, var_by_level={m: {0.95: v} for m, v in points.items()},
        backtest_table=backtest_all_methods(r, rolling_var_forecasts(r, 0.95, 250, models=["Historical"]), 0.95),
        backtest_window=250, stress_table=None, worst_df=historical_worst_losses(r, 1e6), benchmark_name="Nifty 50", beta=1.0,
        events={"metrics": {"event_es": TrustedMetric("Event-adjusted ES", overlay["event_es"] * 1e6, grade="C")},
                "panel": panel, "margin": {"ABC.NS": E.margin_call(1e6, 100.0, 2e5)}, "overlay": overlay,
                "missing": ["F&O ban list"]})
    sheet = openpyxl.load_workbook(io.BytesIO(xlsx))["Events"]
    values = [c.value for row in sheet.iter_rows(min_col=2, max_col=5) for c in row]
    assert "High" in values and "Gap from ABC.NS" in values and "60.0% of promoter shares pledged" in values
    assert "F&O ban list" in sheet["B3"].value


def test_annual_probability_by_hand():
    # 0.5% a day over 252 days: 1 − 0.995^252 = 0.7171 (the review's "about 72% a year"): 0.7172
    assert E.annual_probability(0.005) == pytest.approx(1 - 0.995 ** 252)
    assert E.annual_probability(0.005) == pytest.approx(0.7172, abs=1e-4)
    assert E.annual_probability(0.001) == pytest.approx(0.2229, abs=1e-4)
    assert E.annual_probability(0.0) == 0.0


def test_es_sensitivity_is_monotone_and_matches_the_overlay():
    weights = {"A": 0.6, "B": 0.4}
    table = E.es_sensitivity(BASE, weights, ["A"], 0.95)
    assert list(table.index) == list(E.SENSITIVITY_P) and list(table.columns) == list(E.SENSITIVITY_J)
    # More likely or bigger jumps never lower ES
    assert (table.diff(axis=0).dropna() >= -1e-12).all().all()
    assert (table.diff(axis=1).dropna(axis=1) >= -1e-12).all().all()
    # Each cell is exactly the overlay with that (p, J) on the target holding only
    direct = E.event_adjusted_es(BASE, weights, {"A": (0.005, -0.20)}, 0.95)["event_es"]
    assert table.loc[0.005, -0.20] == pytest.approx(direct, rel=1e-12)
    # and never below the standard ES
    assert (table >= E.event_adjusted_es(BASE, weights, {}, 0.95)["es"] - 1e-12).all().all()
