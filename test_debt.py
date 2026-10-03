"""Debt positions: bond pricing against hand values, the spread-duration approximation against full repricing,
expected loss by hand, the Merton-implied spread link, and debt inside the linked stress engine."""

import numpy as np
import pandas as pd
import pytest

import credit as C
import debt as D
import integration as I

AS_OF = pd.Timestamp("2026-10-01")
CONFIG = D.load_config()


def bond(**overrides):
    row = {"Name": "Test NCD", "Issuer Ticker": "", "Face Value": 1_000_000.0, "Coupon %": 8.0,
           "Maturity": AS_OF + pd.DateOffset(years=3), "Rating": "CRISIL AA", "Seniority": "senior unsecured",
           "Spread bps": 150.0}
    row.update(overrides)
    return row


def test_par_bond_prices_at_par():
    # Coupon 8%, yield 8%, three whole years to maturity: the price is the face value
    t = D.coupon_times(AS_OF + pd.DateOffset(years=3), AS_OF)
    assert t == pytest.approx([1.0, 2.0, 3.0], abs=0.01)
    assert D.price(100.0, 0.08, AS_OF + pd.DateOffset(years=3), AS_OF, 0.08) == pytest.approx(100.0, rel=1e-3)


def test_price_by_hand():
    p = D.price(100.0, 0.06, AS_OF + pd.DateOffset(years=2), AS_OF, 0.10)
    t = D.coupon_times(AS_OF + pd.DateOffset(years=2), AS_OF)
    assert p == pytest.approx(6 / 1.1 ** t[0] + 106 / 1.1 ** t[1])


@pytest.mark.parametrize("shock_bps", [10, 100, 300])
def test_spread_duration_against_full_repricing(shock_bps):
    face, coupon, mat, y = 1_000_000.0, 0.08, AS_OF + pd.DateOffset(years=5), 0.085
    p0 = D.price(face, coupon, mat, AS_OF, y)
    d_mod, conv = D.duration_convexity(face, coupon, mat, AS_OF, y)
    ds = shock_bps / 1e4
    full = p0 - D.price(face, coupon, mat, AS_OF, y + ds)
    first = p0 * d_mod * ds
    second = p0 * (d_mod * ds - 0.5 * conv * ds ** 2)
    # Duration alone overstates the loss (convexity); adding convexity leaves only a third-order error in Δs
    assert first >= full
    assert abs(second - full) < abs(first - full)
    assert abs(second - full) / full < {10: 1e-4, 100: 1e-3, 300: 1e-2}[shock_bps]
    # Numerical derivative check of the analytic duration
    h = 1e-6
    numeric = -(D.price(face, coupon, mat, AS_OF, y + h) - D.price(face, coupon, mat, AS_OF, y - h)) / (2 * h) / p0
    assert d_mod == pytest.approx(numeric, rel=1e-5)


def test_expected_loss_by_hand():
    rates = C.load_default_rates()
    s = D.holding_summary(bond(), 0.065, AS_OF, rates, CONFIG)
    # CRISIL AA: 0.05% a year (FY2025 study); senior unsecured LGD 45% (Basel F-IRB)
    assert s["PD (1y, agency)"] == pytest.approx(0.0005) and s["LGD"] == 0.45
    assert s["Expected Loss (1y)"] == pytest.approx(s["Market Value"] * 0.0005 * 0.45)
    assert s["Jump-to-Default Loss"] == pytest.approx(s["Market Value"] * 0.45)
    sub = D.holding_summary(bond(**{"Seniority": "subordinated"}), 0.065, AS_OF, rates, CONFIG)
    assert sub["LGD"] == 0.75


def test_spread_from_pd():
    assert D.spread_from_pd(0.02, 0.45, 1.0) == pytest.approx(-np.log(1 - 0.009))
    assert D.spread_from_pd(0.0, 0.45) == pytest.approx(0.0, abs=1e-9)


def test_bucket_widening_scales_with_the_market_fall():
    assert D.bucket_widening("CRISIL AA", -0.50, CONFIG) == pytest.approx(0.025)
    assert D.bucket_widening("CRISIL AA", -0.25, CONFIG) == pytest.approx(0.0125)
    assert D.bucket_widening("CRISIL AA", +0.10, CONFIG) == 0.0
    assert D.bucket_widening("CRISIL AA", -0.90, CONFIG) == pytest.approx(0.025 * 1.5)  # capped
    assert D.bucket_widening("CARE BBB-", -0.50, CONFIG) == pytest.approx(0.06)


def test_validation():
    frame = pd.DataFrame([bond(), {**bond(), "Seniority": "junior"}, {**bond(), "Name": ""}])
    clean, errors = D.validate(frame)
    assert len(clean) == 1 and len(errors) == 1 and "seniority" in errors[0]


# ---------------------------------------------------------------
# Inside the linked engine
# ---------------------------------------------------------------

from test_integration import PARAMS, SCENARIO, holdings_frame  # noqa: E402

DEBT_PARAMS = {**PARAMS, "as_of": AS_OF, "debt_config": CONFIG}


def debt_frame(**overrides):
    return D.prepare(pd.DataFrame([bond(**overrides)]), PARAMS["r"], AS_OF)


def test_debt_off_leaves_the_equity_result_unchanged():
    without = I.linked_stress(holdings_frame(), SCENARIO, DEBT_PARAMS)
    with_debt = I.linked_stress(holdings_frame(), SCENARIO, DEBT_PARAMS, debt=debt_frame())
    equity = with_debt["table"][~with_debt["table"]["Ticker"].str.startswith("Debt:")]
    assert equity["Total"].tolist() == pytest.approx(without["table"]["Total"].tolist())
    t = with_debt["table"]
    assert t[["Market", "Liquidity", "Credit", "Events"]].sum(axis=1).tolist() == pytest.approx(t["Total"].tolist())


def test_unlinked_debt_loses_by_rating_bucket_and_only_through_credit():
    res = I.linked_stress(holdings_frame(), SCENARIO, DEBT_PARAMS, debt=debt_frame())
    row = res["table"].set_index("Ticker").loc["Debt: Test NCD"]
    d = debt_frame().iloc[0]
    change = D.bucket_widening("CRISIL AA", SCENARIO["market"], CONFIG)
    loss = D.stressed_loss(d, d, PARAMS["r"], AS_OF, change)["loss"] + d["Expected Loss (1y)"]
    assert row["Total"] == pytest.approx(loss) and row["Spread Change bps"] == pytest.approx(change * 1e4)
    # Credit off: no debt loss (the market link alone does not touch the bond in this engine)
    off = I.linked_stress(holdings_frame(), SCENARIO, DEBT_PARAMS, {"credit": False}, debt=debt_frame())
    assert off["table"].set_index("Ticker").loc["Debt: Test NCD", "Total"] == 0.0


def test_debt_of_an_equity_issuer_follows_its_stressed_merton_pd():
    # B is held as equity (equity 2e4, default point 1.5e4) and its fall is deepened by pledge selling and a lock
    res = I.linked_stress(holdings_frame(), SCENARIO, DEBT_PARAMS, debt=debt_frame(**{"Issuer Ticker": "B"}))
    t = res["table"].set_index("Ticker")
    row = t.loc["Debt: Test NCD"]
    assert row["Method"] == "issuer's stressed Merton PD"
    b = holdings_frame().set_index("Ticker").loc["B"]
    sigma = SCENARIO["holdings"]["B"]["sigma"] * np.sqrt(C.TRADING_DAYS)
    x = t.loc["B", "Final Price"] / b["Price"]
    pd_now = C.solve_merton(b["Equity"], sigma, b["Default Point"], PARAMS["r"], PARAMS["T"])["PD"]
    pd_str = C.solve_merton(b["Equity"] * x, sigma, b["Default Point"], PARAMS["r"], PARAMS["T"])["PD"]
    change = D.spread_from_pd(pd_str, 0.45) - D.spread_from_pd(pd_now, 0.45)
    assert row["Spread Change bps"] == pytest.approx(change * 1e4)
    # The equity links (pledge selling, the lock) deepen the issuer's fall, so the bond loses more with them on:
    # a cross-pillar interaction that the old additive engine could not show
    assert row["Interaction"] > 0
