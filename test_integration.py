"""Integration and decisions: linked stress against plain market stress, reverse stress against a numerical
optimiser, Shapley parts summing to the total, trades, hedge, limits at their boundaries, and the CRO memo."""

import io

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import decisions as Dz
import integration as I
import memo


# ---------------------------------------------------------------
# Linked engine
# ---------------------------------------------------------------

def holdings_frame(**overrides):
    base = {"Ticker": ["A", "B"], "Value": [600.0, 400.0], "Quantity": [6.0, 8.0], "Price": [100.0, 50.0],
            "ADV": [1000.0, 50.0], "Spread Mean": [0.002, 0.01], "Spread Std": [0.001, 0.004], "Band": [np.nan, 0.05],
            "Freeze Days": [0, 3], "Equity": [1e5, 2e4], "Default Point": [3e4, 1.5e4], "PD": [1e-6, 0.01],
            "Financial": [False, False], "Pledged Shares": [np.nan, 4000.0], "Volume Ratio": [0.6, 0.5]}
    base.update(overrides)
    return pd.DataFrame(base)


SCENARIO = {"name": "Crash", "kind": "historical", "market": -0.35,
            "holdings": {"A": {"return": -0.30, "method": "replay", "sigma": 0.04},
                         "B": {"return": -0.45, "method": "replay", "sigma": 0.06}}}
PARAMS = {"bangia_k": 3.0, "impact_y": 1.0, "r": 0.065, "T": 1.0, "initial_cover": 2.0, "trigger_cover": 1.5}


LINKS_OFF = {"liquidity": False, "credit": False, "events": False}
THETA = I.PERMANENT_SHARE


def test_linked_engine_with_links_off_is_plain_market_stress():
    res = I.linked_stress(holdings_frame(), SCENARIO, PARAMS, LINKS_OFF)
    assert res["totals"]["Total"] == pytest.approx(600 * 0.30 + 400 * 0.45)
    assert res["totals"]["Market"] == pytest.approx(res["totals"]["Total"])
    assert res["totals"][["Liquidity", "Credit", "Events", "Interaction"]].abs().sum() == pytest.approx(0)


@pytest.mark.parametrize("link", ["liquidity", "credit", "events"])
def test_a_single_link_alone_has_zero_interaction(link):
    res = I.linked_stress(holdings_frame(), SCENARIO, PARAMS, {**LINKS_OFF, link: True})
    assert res["totals"]["Interaction"] == pytest.approx(0, abs=1e-9)


def test_shapley_parts_sum_to_the_total_and_credit_adds_no_loss_for_equity():
    res = I.linked_stress(holdings_frame(), SCENARIO, PARAMS)
    t = res["table"].set_index("Ticker")
    assert t[["Market", "Liquidity", "Credit", "Events"]].sum(axis=1).tolist() == pytest.approx(t["Total"].tolist())
    # C2: the equity price already carries the default risk, so credit is a null player for equity holders...
    assert (t["Credit"] == 0).all()
    # ...but the stressed distance to default is still reported as a signal, and it falls with the price
    assert np.isfinite(t.loc["B", "Stressed DD"]) and t.loc["B", "Stressed PD"] > 0.01
    # B fell 45%, through its 5% band and its 25% pledge trigger: lenders sell and the price falls further
    assert t.loc["B", "Forced Shares"] > 0 and t.loc["B", "Locked"] and t.loc["B", "Feedback Fall"] < 0
    assert t.loc["A", "Forced Shares"] == 0 and t.loc["A", "Events"] == 0


def test_no_pledge_selling_above_the_trigger():
    mild = {**SCENARIO, "holdings": {"A": {"return": -0.1, "method": "replay", "sigma": 0.02},
                                     "B": {"return": -0.2, "method": "replay", "sigma": 0.03}}}
    no_band = holdings_frame(**{"Band": [np.nan, np.nan]})
    t = I.linked_stress(no_band, mild, PARAMS)["table"].set_index("Ticker")
    # a 20% fall (plus about 1% from our own exit's impact) is short of the 25% margin-call trigger
    assert t.loc["B", "Forced Shares"] == 0 and t.loc["B", "Events"] == pytest.approx(0)
    # With B's 5% band, the same fall locks it for 3 lower circuits (0.8 × 0.95³ ≈ 0.69), which takes the price
    # through the trigger: the circuit sets off the margin call, a cross effect no single link shows
    spiral = I.linked_stress(holdings_frame(), mild, PARAMS)["table"].set_index("Ticker")
    assert spiral.loc["B", "Locked"] and spiral.loc["B", "Forced Shares"] > 0 and spiral.loc["B", "Interaction"] > 0


# A pledged, thinly traded holding whose fall sells every pledged share at once, so the fixed point is reached
# in the second round and every coalition can be worked out by hand.
PLEDGED = pd.DataFrame({"Ticker": ["P"], "Value": [1000.0], "Quantity": [100.0], "Price": [10.0], "ADV": [1000.0],
                        "Spread Mean": [np.nan], "Spread Std": [np.nan], "Band": [np.nan], "Freeze Days": [0],
                        "Equity": [np.nan], "Default Point": [np.nan], "PD": [np.nan], "Financial": [False],
                        "Pledged Shares": [4000.0], "Volume Ratio": [1.0]})
HALF = {"name": "Halved", "kind": "custom", "market": -0.5, "holdings": {"P": {"return": -0.5, "method": "β-proxy", "sigma": 0.05}}}


def test_pledge_and_thin_volume_interaction_matches_a_hand_calculation():
    # Loan = 4,000 × 10 / 2 = 20,000. At a price of 5 the cover is 4,000 × 5 / 20,000 = 1.0 < 1.5, and restoring
    # a cover of 2 needs (2 × 20,000 − 4,000 × 5) / (5 × 1) = 4,000 shares: all of them, at any lower price too.
    impact = lambda q: 0.05 * np.sqrt(q / 1000)  # noqa: E731  (Y = 1, σ = 5%, ADV 1,000)
    v = {}
    v["M"] = 500.0
    x = 0.5 * (1 - THETA * impact(100))                       # market + our exit
    v["ML"] = 1000 - 1000 * x * (1 - (1 - THETA) * impact(100))
    x = 0.5 * (1 - THETA * impact(4000))                      # market + lenders' sale, we do not sell
    v["ME"] = 1000 - 1000 * x
    x = 0.5 * (1 - THETA * impact(4100))                      # both, on the combined flow
    v["MLE"] = 1000 - 1000 * x * (1 - (1 - THETA) * impact(4100))
    x = 1 - THETA * impact(100)                               # no market fall: our exit only, no margin call
    v["L"] = v["LE"] = 1000 - 1000 * x * (1 - (1 - THETA) * impact(100))
    v[""] = v["E"] = 0.0
    res = I.linked_stress(PLEDGED, HALF, PARAMS)
    row = res["table"].iloc[0]
    assert row["Total"] == pytest.approx(v["MLE"]) and row["Rounds"] == 2 and row["Converged"]
    assert row["Forced Shares"] == pytest.approx(4000)
    interaction = v["MLE"] - (v["M"] + (v["ML"] - v["M"]) + (v["ME"] - v["M"]))
    assert row["Interaction"] == pytest.approx(interaction) and interaction > 0
    # Exact Shapley over market, liquidity and events (credit is a null player here)
    third, sixth = 1 / 3, 1 / 6
    phi_m = third * v["M"] + sixth * (v["ML"] - v["L"]) + sixth * (v["ME"] - v["E"]) + third * (v["MLE"] - v["LE"])
    phi_l = third * v["L"] + sixth * (v["ML"] - v["M"]) + sixth * (v["LE"] - v["E"]) + third * (v["MLE"] - v["ME"])
    phi_e = third * v["E"] + sixth * (v["ME"] - v["M"]) + sixth * (v["LE"] - v["L"]) + third * (v["MLE"] - v["ML"])
    assert [row["Market"], row["Liquidity"], row["Events"], row["Credit"]] == pytest.approx([phi_m, phi_l, phi_e, 0.0])


def strong_feedback_frame():
    # Cover just above the trigger before our own exit; thin volume and a 5% band
    return PLEDGED.assign(**{"Quantity": [300.0], "Value": [3000.0], "ADV": [400.0], "Band": [0.05], "Freeze Days": [2],
                             "Pledged Shares": [2000.0]})


NEAR_TRIGGER = {"name": "Near trigger", "kind": "custom", "market": -0.24,
                "holdings": {"P": {"return": -0.24, "method": "β-proxy", "sigma": 0.03}}}


def test_strong_feedback_converges_through_the_trigger_and_the_band():
    row = I.linked_stress(strong_feedback_frame(), NEAR_TRIGGER, PARAMS)["table"].iloc[0]
    # 24% is above the 25% trigger, but our exit's permanent impact pushes the price through it, the lenders'
    # sale pushes it further, and the stock locks at its band
    assert row["Converged"] and 3 <= row["Rounds"] <= I.MAX_ROUNDS
    assert row["Forced Shares"] > 0 and row["Locked"] and row["Feedback Fall"] < -0.05
    assert row["Interaction"] > 0
    alone = I.linked_stress(strong_feedback_frame(), NEAR_TRIGGER, PARAMS, {**LINKS_OFF, "events": True})["table"].iloc[0]
    assert alone["Forced Shares"] == 0  # without our exit, a 24% fall never reaches the trigger


def test_feedback_that_has_not_settled_is_reported(monkeypatch):
    monkeypatch.setattr(I, "MAX_ROUNDS", 2)
    row = I.linked_stress(strong_feedback_frame(), NEAR_TRIGGER, PARAMS)["table"].iloc[0]
    assert row["Rounds"] == 2 and not row["Converged"]


def test_jump_to_default_only_below_the_threshold():
    res = I.linked_stress(holdings_frame(), SCENARIO, {**PARAMS, "jtd_dd": 100.0})
    t = res["table"].set_index("Ticker")
    # Every holding is below a DD of 100: the loss if the equity goes to 10% or 0% recovery, never added to Total
    assert t.loc["B", "JTD Loss (10% recovery)"] == pytest.approx(400 * 0.9)
    assert t.loc["B", "JTD Loss (0% recovery)"] == pytest.approx(400.0)
    none = I.linked_stress(holdings_frame(), SCENARIO, {**PARAMS, "jtd_dd": 0.0})["table"]
    assert none["JTD Loss (0% recovery)"].isna().all()
    assert none["Total"].tolist() == pytest.approx(t["Total"].tolist())


def test_crisis_volume_never_raises_liquidity():
    calm = I.linked_stress(holdings_frame(**{"Volume Ratio": [2.0, 3.0]}), SCENARIO, PARAMS)["table"]
    normal = I.linked_stress(holdings_frame(**{"Volume Ratio": [1.0, 1.0]}), SCENARIO, PARAMS)["table"]
    assert calm["Liquidity"].tolist() == pytest.approx(normal["Liquidity"].tolist())


def _with_path(path):
    """SCENARIO with B's replayed daily path given (its return is the compounded path)."""
    path = pd.Series(path, index=pd.bdate_range("2020-03-02", periods=len(path)))
    b = {"return": float(np.prod(1 + path) - 1), "method": "replay", "sigma": 0.06, "path": path}
    return {**SCENARIO, "holdings": {"A": SCENARIO["holdings"]["A"], "B": b}}


LIQUIDITY_ONLY = {**LINKS_OFF, "liquidity": True}


@pytest.mark.parametrize("freeze, locked_in_path, extra", [(3, 3, 0), (5, 3, 2), (3, 0, 3)])
def test_circuit_freeze_counts_only_days_beyond_the_replayed_locks(freeze, locked_in_path, extra):
    # B's band is 5%; the path has `locked_in_path` lower-circuit days among ordinary falls
    path = [-0.05] * locked_in_path + [-0.03] * 6 + [0.01] * 2
    scenario = _with_path(path)
    frame = holdings_frame(**{"Freeze Days": [0, freeze]})
    t = I.linked_stress(frame, scenario, PARAMS, LIQUIDITY_ONLY)["table"].set_index("Ticker")
    no_band = I.linked_stress(holdings_frame(**{"Band": [np.nan, np.nan]}), scenario, PARAMS, LIQUIDITY_ONLY)["table"].set_index("Ticker")
    # The freeze multiplies the final price by (1 − band) for each extra lower circuit only
    assert t.loc["B", "Final Price"] / no_band.loc["B", "Final Price"] == pytest.approx(0.95 ** extra)
    assert t.loc["B", "Locked Days in Replay"] == locked_in_path


def test_circuit_freeze_without_a_path_is_unchanged():
    # β-proxy and custom scenarios have no daily path: the whole freeze applies
    t = I.linked_stress(holdings_frame(), SCENARIO, PARAMS, LIQUIDITY_ONLY)["table"].set_index("Ticker")
    no_band = I.linked_stress(holdings_frame(**{"Band": [np.nan, np.nan]}), SCENARIO, PARAMS, LIQUIDITY_ONLY)["table"].set_index("Ticker")
    assert t.loc["B", "Final Price"] / no_band.loc["B", "Final Price"] == pytest.approx(0.95 ** 3)


def test_liquidity_alone_by_hand_and_close_to_the_old_additive_costs():
    # Without a feedback partner: x* = x_m·(1 − θI); proceeds = V₀·x*·(1 − (1−θ)I − spread rate). The old engine
    # charged market loss + spread + impact additively; the two differ only at second order.
    frame = holdings_frame(**{"Band": [np.nan, np.nan]})
    t = I.linked_stress(frame, SCENARIO, PARAMS, LIQUIDITY_ONLY)["table"].set_index("Ticker")
    for tk, v, q, adv, ret, s, m, sd in (("A", 600, 6, 1000 * 0.6, -0.30, 0.04, 0.002, 0.001),
                                         ("B", 400, 8, 50 * 0.5, -0.45, 0.06, 0.01, 0.004)):
        i = 1.0 * s * np.sqrt(q / adv)
        spread_rate = 0.5 * (m + 3.0 * sd)
        x = (1 + ret) * (1 - THETA * i)
        assert t.loc[tk, "Total"] == pytest.approx(v - v * x * (1 - (1 - THETA) * i - spread_rate))
        old = -ret * v + v * (1 + ret) * (spread_rate + i)
        assert t.loc[tk, "Total"] == pytest.approx(old, rel=0.01)


def test_run_linked_reports_the_parts_and_the_interaction():
    row = I.run_linked(holdings_frame(), [SCENARIO], PARAMS).iloc[0]
    assert row["Market Loss"] == pytest.approx(600 * 0.30 + 400 * 0.45)
    assert row[["Market", "Liquidity", "Credit", "Events"]].sum() == pytest.approx(row["Linked Total"])
    assert row["Linked Total"] > row["Market Loss"] and row["Converged"]
    assert "Siloed Sum" not in row.index


def test_forced_sale_restores_the_cover():
    import events as E
    s, p0, c0, ct = 1000.0, 10.0, 2.0, 1.5
    # At the trigger price it equals margin_call's shares to restore
    trigger = E.margin_call(s, p0, 100.0, c0, ct)
    assert E.forced_sale(s, p0, trigger["trigger_price"] * (1 - 1e-12), c0, ct) == pytest.approx(trigger["shares_to_restore"])
    # Above the trigger nothing is sold
    assert E.forced_sale(s, p0, 8.0, c0, ct) == 0.0
    # At 6: loan 5,000; selling x restores (1,000 − x)·6 / (5,000 − 6x) = 2 → x = 666.67 (more than at the trigger)
    x = E.forced_sale(s, p0, 6.0, c0, ct)
    assert x == pytest.approx(4000 / 6) and (s - x) * 6 / (5000 - 6 * x) == pytest.approx(2.0)
    assert x > trigger["shares_to_restore"]
    # At 3 restoring would need more than every pledged share: capped
    assert E.forced_sale(s, p0, 3.0, c0, ct) == s


def test_scenario_inputs_replay_and_proxy():
    dates = pd.bdate_range("2019-06-01", "2021-06-30")
    rng = np.random.default_rng(0)
    market_ret = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    market_ret[(dates >= "2020-02-20") & (dates <= "2020-03-23")] = -0.015
    prices = 100 * (1 + market_ret).cumprod()
    full = (market_ret * 1.2 + rng.normal(0, 0.005, len(dates))).rename("A")
    young = full[full.index >= "2020-06-01"].rename("B")
    scenarios = pd.DataFrame({"scenario": ["COVID"], "start": [pd.Timestamp("2020-01-01")], "end": [pd.Timestamp("2020-04-30")]})
    out = I.scenario_inputs({"A": full, "B": young}, prices, scenarios, {"A": 0.01, "B": 0.01}, {"A": full, "B": young})
    covid = out[0]
    assert covid["holdings"]["A"]["method"] == "replay" and covid["holdings"]["B"]["method"] == "β-proxy"
    assert covid["holdings"]["A"]["return"] < -0.2
    # The replayed path is kept, and compounds to the replayed return
    path = covid["holdings"]["A"]["path"]
    assert np.prod(1 + path) - 1 == pytest.approx(covid["holdings"]["A"]["return"])
    assert "path" not in covid["holdings"]["B"]
    assert [o["name"] for o in out[1:]] == ["Market -10%", "Market -20%", "Market -30%"]


# ---------------------------------------------------------------
# Reverse stress
# ---------------------------------------------------------------

COV = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]) * 1e-4 * 21
W = np.array([0.5, 0.3, 0.2])


@pytest.mark.parametrize("loss", [0.05, 0.15, 0.30])
def test_closed_form_reverse_stress_matches_an_optimiser(loss):
    res = I.reverse_stress(W, COV, loss)
    inv = np.linalg.inv(COV)
    num = minimize(lambda x: x @ inv @ x, np.zeros(3), constraints=[{"type": "eq", "fun": lambda x: W @ x + loss}],
                   options={"ftol": 1e-14, "maxiter": 500})
    assert res["shock"] == pytest.approx(num.x, abs=1e-6)
    assert res["d2"] == pytest.approx(num.fun, rel=1e-6)
    assert W @ res["shock"] == pytest.approx(-loss)


def _draws(nu, n, seed):
    """Normal and multivariate Student-t draws whose COVARIANCE is COV (the t's dispersion is COV·(ν−2)/ν)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3)) @ np.linalg.cholesky(COV).T
    t = z * np.sqrt((nu - 2) / nu) / np.sqrt(rng.chisquare(nu, n) / nu)[:, None]
    return z, t


def test_extreme_share_matches_a_simulation_with_covariance_distance():
    nu = 5
    p = I.plausibility(9.0, 3, nu=nu)
    from scipy.stats import chi2
    assert p["normal"] == pytest.approx(chi2.sf(9.0, 3))
    z, t = _draws(nu, 400_000, 1)
    inv = np.linalg.inv(COV)
    d2_z, d2_t = np.einsum("ij,jk,ik->i", z, inv, z), np.einsum("ij,jk,ik->i", t, inv, t)
    # d² is measured with the covariance, as reverse_stress does; the old F(k, ν) formula assumed the dispersion
    assert (d2_z >= 9.0).mean() == pytest.approx(p["normal"], rel=0.03)
    assert (d2_t >= 9.0).mean() == pytest.approx(p["student_t"], rel=0.03)


@pytest.mark.parametrize("loss", [0.05, 0.10, 0.15])
def test_loss_probability_matches_a_simulation(loss):
    nu = 5
    sigma = I.reverse_stress(W, COV, loss)["portfolio_sigma"]
    p = I.loss_probability(loss, sigma, nu)
    z, t = _draws(nu, 2_000_000, 3)
    sim_n, sim_t = (z @ W <= -loss).mean(), (t @ W <= -loss).mean()
    assert sim_n == pytest.approx(p["normal"], rel=0.05, abs=2e-5)
    assert sim_t == pytest.approx(p["student_t"], rel=0.05, abs=2e-5)


def test_loss_probability_is_one_sided_and_far_below_the_any_direction_share():
    from scipy.stats import norm
    res = I.reverse_stress(W, COV, 0.10)
    p = I.loss_probability(0.10, res["portfolio_sigma"], None)
    assert p["normal"] == pytest.approx(norm.cdf(-0.10 / res["portfolio_sigma"]))
    assert np.isnan(p["student_t"])
    assert p["normal"] < I.plausibility(res["d2"], 3)["normal"]
    # Once every N years: the loss is over one month, so N = 1 / (12 p)
    assert p["normal_years"] == pytest.approx(1 / (12 * p["normal"]))


def test_macro_reverse_finds_the_planted_analogue():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2015-01-01", periods=2500)
    macro = pd.DataFrame(rng.normal(0, 0.01, (2500, 2)), columns=["Index", "Oil"], index=dates)
    # Plant a month that looks like the most plausible 15% loss (index about −14%, oil about −3% over 21 days)
    macro.iloc[1000:1021, 0] = -0.0071
    macro.iloc[1000:1021, 1] = -0.0014
    port = 1.0 * macro["Index"] + 0.2 * macro["Oil"] + rng.normal(0, 0.003, 2500)
    res = I.macro_reverse(port, macro, 0.15, fit_days=2500)
    assert res["betas"]["Index"] == pytest.approx(1.0, abs=0.05) and res["shock"]["Index"] < -0.1
    assert res["analogue"]["end"] in dates[1015:1026]
    assert res["analogue"]["portfolio_return"] < -0.1 and res["r2"] > 0.8


# ---------------------------------------------------------------
# Risk change, trades, hedge, limits
# ---------------------------------------------------------------

def returns_frame(seed=0, scale=1.0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]) * 1e-4 * scale
    return pd.DataFrame(rng.multivariate_normal([0.0005, 0.0003, 0.0001], cov, 500), columns=list("ABC"),
                        index=pd.bdate_range(start, periods=500))


def test_recolor_with_own_moments_returns_the_window():
    R = returns_frame()
    assert Dz.recolor(R, R.std(ddof=1), R.corr()) == pytest.approx(R.to_numpy(), abs=1e-12)


def test_shapley_parts_sum_to_the_change():
    old = Dz.snapshot(returns_frame(0), pd.Series([0.5, 0.3, 0.2], index=list("ABC")), "Historical", "2025-12-31", 1e6, 0.95)
    new = Dz.snapshot(returns_frame(1, 2.5, "2026-01-01"), pd.Series([0.3, 0.3, 0.4], index=list("ABC")),
                      "Parametric (Normal)", "2027-12-31", 1e6, 0.95)
    rc = Dz.risk_change(old, new, 0.05, 1e6)
    assert sum(rc["contributions"].values()) == pytest.approx(rc["end"] - rc["start"])
    assert rc["contributions"]["volatility"] > 0  # volatility rose 2.5× in variance


def test_shapley_on_an_additive_game_gives_each_player_its_own_effect():
    values = {"a": (0, 1), "b": (0, 2), "c": (0, 3)}
    res = Dz.shapley(values, lambda s: 10 * s["a"] + 5 * s["b"] + s["c"])
    assert res["contributions"] == pytest.approx({"a": 10, "b": 10, "c": 3})


def test_only_changed_players_get_credit():
    R = returns_frame()
    w = pd.Series([0.5, 0.3, 0.2], index=list("ABC"))
    snap = Dz.snapshot(R, w, "Historical", "2025-12-31", 1e6, 0.95)
    moved = Dz.snapshot(R, pd.Series([0.2, 0.3, 0.5], index=list("ABC")), "Historical", "2025-12-31", 1e6, 0.95)
    rc = Dz.risk_change(snap, moved, 0.05, 1e6)
    assert abs(rc["contributions"]["positions"] - (rc["end"] - rc["start"])) < 1e-6
    assert all(abs(v) < 1e-6 for k, v in rc["contributions"].items() if k != "positions")


def test_snapshot_json_round_trip():
    snap = Dz.snapshot(returns_frame(), pd.Series([0.5, 0.3, 0.2], index=list("ABC")), "Student-t", "2025-12-31", 1e6, 0.95)
    back = Dz.snapshot_from_json(Dz.snapshot_to_json(snap))
    assert back["model"] == "Student-t" and back["weights"].to_dict() == pytest.approx(snap["weights"].to_dict())
    assert back["returns"].to_numpy() == pytest.approx(snap["returns"].to_numpy(), abs=1e-9)


def test_component_es_sums_and_best_trade_cuts_es():
    R = returns_frame()
    w = pd.Series([0.6, 0.3, 0.1], index=list("ABC"))
    comp = Dz.component_es(R, w, 0.05)
    assert comp.sum() == pytest.approx(Dz.portfolio_es(R.to_numpy() @ w.to_numpy(), 0.05))
    trades = Dz.candidate_trades(R, w, 0.05)
    assert trades["ES Change"].iloc[0] == trades["ES Change"].min() < 0
    best = trades.iloc[0]["weights"]
    assert best.sum() <= 1.0 + 1e-12 and (best >= -1e-12).all()


def test_hedge_removes_pure_market_risk():
    rng = np.random.default_rng(3)
    market = pd.Series(rng.normal(0, 0.01, 1000), index=pd.bdate_range("2022-01-03", periods=1000))
    portfolio = 0.8 * market + rng.normal(0, 0.001, 1000)
    h = Dz.hedge_ratio(portfolio, market, 0.05)
    assert h["ratio"] == pytest.approx(0.8, abs=0.05) and h["es_after"] < 0.2 * h["es_before"]


@pytest.mark.parametrize("utilisation, status", [(0.79, "green"), (0.8, "amber"), (1.0, "amber"), (1.01, "red")])
def test_traffic_lights_at_their_boundaries(utilisation, status):
    assert Dz.light(utilisation) == status


def test_evaluate_limits():
    limits = Dz.load_limits()
    measures = {"es_pct": 0.032, "max_weight": 0.31, "max_sector_weight": 0.2, "days_to_liquidate_50": np.nan,
                "max_weighted_pd": 0.0, "max_high_tier": 1.0, "min_trust_grade": "D"}
    t = Dz.evaluate_limits(limits, measures).set_index("Key")
    assert t.loc["es_pct", "Status"] == "amber" and t.loc["max_weight", "Status"] == "red"
    assert t.loc["max_sector_weight", "Status"] == "green" and t.loc["days_to_liquidate_50", "Status"] == "not available"
    assert t.loc["max_high_tier", "Status"] == "amber"  # one High-tier holding against a limit of one
    assert t.loc["min_trust_grade", "Status"] == "red"  # D is worse than the C limit
    assert Dz.evaluate_limits(limits, {**measures, "min_trust_grade": "B"}).set_index("Key").loc["min_trust_grade", "Status"] == "green"


def test_top_risks_are_ranked_by_rule():
    # max_weight 0.2 of a 0.3 limit is green; an amber limit (≥ 90 points) would outrank a High-tier holding (55)
    limits = Dz.evaluate_limits(Dz.load_limits(), {"es_pct": 0.05, "max_weight": 0.2, "max_sector_weight": 0.2,
                                                   "days_to_liquidate_50": 0.1, "max_weighted_pd": 0.0,
                                                   "max_high_tier": 0.0, "min_trust_grade": "B"})
    risks = Dz.top_risks(limits, {"name": "GFC", "loss_pct": 0.4, "interaction_pct": 0.02},
                         pd.Series({"A": 55.0, "B": 45.0}), pd.Series({"A": 0.5, "B": 0.5}), {"A": "High", "B": "Low"},
                         {"Market ES": "B"}, ["F&O ban list"])
    assert risks[0].startswith("Limit breached: ES") and "GFC" in risks[1] and "High event-risk tier" in risks[2]


# ---------------------------------------------------------------
# Memo
# ---------------------------------------------------------------

def memo_content():
    limits = Dz.evaluate_limits(Dz.load_limits(), {"es_pct": 0.05, "max_weight": 0.3, "max_sector_weight": 0.3,
                                                   "days_to_liquidate_50": 0.1, "max_weighted_pd": 0.0,
                                                   "max_high_tier": 0.0, "min_trust_grade": "B"})
    rows = [{**r, "Shown": str(r["Value"]), "Limit Shown": str(r["Limit Value"]), "Utilisation Shown": f"{r['Utilisation']:.0%}"}
            for r in limits.to_dict("records")]
    return {"name": "5-stock portfolio", "as_of": pd.Timestamp("2026-10-01"), "confidence": "95%, 1-day", "value": "₹1,000,000",
            "currency": "INR", "pillars": [{"Pillar": "Market", "Headline": "ES ₹16,125", "Range": "90% range ₹15,165–₹17,114",
                                            "Grade": "B", "Status": "red"}],
            "stress": {"name": "GFC", "market": "₹250,000", "liquidity": "₹5,000", "credit": "₹100", "events": "₹0",
                       "linked": "₹255,100", "plain": "₹250,000", "interaction": "₹1,100", "linked_pct": 0.2551,
                       "interaction_pct": 0.0011, "market_value": 250000, "liquidity_value": 5000, "credit_value": 100,
                       "events_value": 0, "plain_value": 250000},
            "reverse": "Most plausible way to lose 15% in a month: Nifty 50 −12%.", "limits": rows,
            "risks": ["Limit breached: ES"], "actions": ["Sell 5% of A into cash"], "sources": ["Yahoo Finance"]}


def test_memo_markdown_has_every_section():
    md = memo.markdown(memo_content())
    for section in ("## Bottom line", "## Pillar summary", "## Integrated stress", "## What the numbers miss", "## Limits",
                    "## Actions", "## Data sources and caveats"):
        assert section in md, section
    assert "1 limit(s) breached" in memo.bottom_line(memo_content()) and "no language model" in md


def test_memo_pdf_is_one_page():
    import re
    data = memo.pdf(memo_content())
    assert data[:4] == b"%PDF"
    assert len(re.findall(rb"/Type\s*/Page[^s]", data)) == 1  # exactly one page object


def test_excel_integrated_sheet():
    import openpyxl
    from excel_exporter import generate_excel_var_report
    from var_calculator import backtest_all_methods, calculate_all_var, historical_worst_losses, rolling_var_forecasts

    R = returns_frame()
    r = R["A"]
    df = pd.DataFrame({"Date": R.index, "Close": 100 * np.cumprod(1 + r), "Returns": r.to_numpy(), "Log_Returns": np.log1p(r).to_numpy(),
                       "Rolling_30d_Vol": r.rolling(30).std().to_numpy()})
    points = calculate_all_var(r, 1e6, 0.95, 1, num_simulations=1000)
    linked = I.run_linked(holdings_frame(), [SCENARIO], PARAMS).drop(columns=["detail"])
    limits = Dz.evaluate_limits(Dz.load_limits(), {"es_pct": 0.05})
    xlsx = generate_excel_var_report(
        "A", "Test", "INR", 1e6, 0.95, 1, df, var_by_level={m: {0.95: v} for m, v in points.items()},
        backtest_table=backtest_all_methods(r, rolling_var_forecasts(r, 0.95, 250, models=["Historical"]), 0.95),
        backtest_window=250, stress_table=None, worst_df=historical_worst_losses(r, 1e6), benchmark_name="Nifty 50", beta=1.0,
        integrated={"linked": linked, "limits": limits, "risks": ["Limit breached: ES"], "actions": ["Sell 5% of A"],
                    "change": {"start": 100.0, "end": 120.0, "contributions": {"volatility": 20.0}}})
    sheet = openpyxl.load_workbook(io.BytesIO(xlsx))["Integrated & Decisions"]
    values = [c.value for row in sheet.iter_rows(min_col=2, max_col=4) for c in row]
    assert "Crash" in values and "Limit breached: ES" in values and "ES now" in values and "volatility" in values
