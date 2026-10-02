"""
Event and governance risk pillar (Pillar 5, India): early-warning signals per holding, a Low / Elevated / High
event-risk tier from written rules, promoter-pledge margin calls, and a jump overlay on Expected Shortfall.

Every signal uses only data public by the analysis date (disclosures.as_of). Signals whose data are not loaded
are reported as not available and never treated as "no risk". Thresholds live in config/event_rules.json;
jump probabilities and sizes are editable assumptions. Methods: docs/methodology.md section 12.
"""

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t as student_t

from disclosures import as_of as public_rows

RULES_PATH = Path(__file__).parent / "config" / "event_rules.json"
LOW, ELEVATED, HIGH = "Low", "Elevated", "High"
TIERS = (LOW, ELEVATED, HIGH)
DEFAULT_JUMPS = {LOW: (0.0, 0.0), ELEVATED: (0.001, -0.10), HIGH: (0.005, -0.20)}  # (daily probability, size)
INITIAL_COVER, TRIGGER_COVER = 2.0, 1.5
EXACT_ENUMERATION_MAX = 12
SIGNALS = ("pledge", "surveillance", "fo_ban", "rating", "auditor", "merton", "circuit", "group")
MEASURE_SEVERITY = {"GSM": 4, "ASM-LT": 3, "ASM-ST": 2, "ESM": 1}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
RATING_SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-",
                "C+", "C", "C-", "D"]


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(frame: pd.DataFrame, dataset: str, symbol: str, as_of) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    return public_rows(frame[frame["symbol"] == symbol], dataset, as_of)


# ---------------------------------------------------------------
# Signals
# ---------------------------------------------------------------

def pledge_signal(pledges: pd.DataFrame, symbol: str, as_of) -> dict:
    """Latest pledged % of promoter holding public by `as_of`, and its change from about four quarters earlier."""
    rows = _rows(pledges, "pledges", symbol, as_of)
    if rows.empty:
        return None
    rows = rows.sort_values("quarter_end")
    last = rows.iloc[-1]
    earlier = rows[pd.to_datetime(rows["quarter_end"]) <= pd.Timestamp(last["quarter_end"]) - pd.Timedelta(days=360)]
    change = float(last["pledged_pct_of_promoter"] - earlier.iloc[-1]["pledged_pct_of_promoter"]) if len(earlier) else np.nan
    return {"pledged_pct_of_promoter": float(last["pledged_pct_of_promoter"]),
            "pledged_pct_of_total": float(last["pledged_pct_of_total"]) if pd.notna(last.get("pledged_pct_of_total")) else np.nan,
            "pledged_shares": float(last["pledged_shares"]) if pd.notna(last.get("pledged_shares")) else np.nan,
            "promoter_holding_pct": float(last["promoter_holding_pct"]), "change_4q_pp": change,
            "quarter_end": pd.Timestamp(last["quarter_end"])}


def stage_number(stage) -> int:
    """'Stage III' / 'III' / '3' → 3; 0 if no stage can be read."""
    text = str(stage).upper().replace("STAGE", "").strip()
    if text.isdigit():
        return int(text)
    return ROMAN.get(text, 0)


def surveillance_signal(surveillance: pd.DataFrame, symbol: str, as_of) -> dict:
    """The most severe surveillance measure in force on `as_of` (entered on or before it, not yet exited)."""
    rows = _rows(surveillance, "surveillance", symbol, as_of)
    if rows.empty:
        return None
    exit_dates = pd.to_datetime(rows["date_out"])
    active = rows[exit_dates.isna() | (exit_dates > pd.Timestamp(as_of))]
    if active.empty:
        return {"measure": None, "stage": 0, "since": None}
    active = active.assign(sev=active["measure"].map(MEASURE_SEVERITY), num=active["stage"].map(stage_number))
    top = active.sort_values(["sev", "num"]).iloc[-1]
    return {"measure": top["measure"], "stage": int(top["num"]), "stage_text": top["stage"], "since": pd.Timestamp(top["date_in"])}


def fo_ban_signal(bans: pd.DataFrame, symbol: str, as_of, days: int = 30) -> dict:
    """The latest F&O ban date within `days` calendar days before `as_of`."""
    rows = _rows(bans, "fo_ban", symbol, as_of)
    if bans is None or bans.empty:
        return None
    recent = rows[pd.to_datetime(rows["trade_date"]) >= pd.Timestamp(as_of) - pd.Timedelta(days=days)] if len(rows) else rows
    return {"last_ban": pd.Timestamp(recent["trade_date"].max()) if len(recent) else None, "bans_in_window": int(len(recent))}


def rating_rank(rating) -> int:
    """Position on the AAA…D scale (0 = AAA); -1 if no grade can be read. Agency prefixes and suffixes are ignored."""
    text = str(rating).upper()
    for grade in sorted(RATING_SCALE, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(grade)}(?![A-Z+\-])", text):
            return RATING_SCALE.index(grade)
    return -1


def rating_signal(ratings: pd.DataFrame, symbol: str, as_of, floor: str = "BBB-", months: int = 12) -> dict:
    """Downgrades and negative watches in the last `months`, and whether any downgrade landed below `floor`."""
    if ratings is None or ratings.empty:
        return None
    rows = _rows(ratings, "ratings", symbol, as_of)
    window = rows[pd.to_datetime(rows["action_date"]) >= pd.Timestamp(as_of) - pd.DateOffset(months=months)] if len(rows) else rows
    downgrades = window[window["action"] == "downgraded"] if len(window) else window
    watch = window[(window["action"] == "placed on watch") & window["outlook"].fillna("").str.lower().str.contains("negative")] \
        if len(window) else window
    below = [r for r in downgrades["rating"]] if len(downgrades) else []
    return {"downgrades": int(len(downgrades)), "negative_watch": bool(len(watch)),
            "below_investment_grade": any(rating_rank(r) > RATING_SCALE.index(floor) for r in below),
            "latest": rows.sort_values("action_date").iloc[-1]["rating"] if len(rows) else None}


def auditor_signal(events: pd.DataFrame, symbol: str, as_of, months: int = 24) -> dict:
    """Auditor events (resignations, modified opinions) in the last `months`."""
    if events is None or events.empty:
        return None
    rows = _rows(events, "auditor_events", symbol, as_of)
    window = rows[pd.to_datetime(rows["event_date"]) >= pd.Timestamp(as_of) - pd.DateOffset(months=months)] if len(rows) else rows
    return {"events": sorted(set(window["event_type"])) if len(window) else [],
            "latest": pd.Timestamp(window["event_date"].max()) if len(window) else None}


def dd_signal(rolling: pd.DataFrame, as_of) -> dict:
    """Merton DD at the latest month end and six months earlier (from the credit pillar's month-end series)."""
    if rolling is None or rolling.empty:
        return None
    series = rolling.set_index("Date")["DD"].dropna()
    series = series[series.index <= pd.Timestamp(as_of)]
    if series.empty:
        return None
    now = float(series.iloc[-1])
    # Compare calendar months: 30 June minus six months must find the 31 December month end, not 30 November
    months = series.index.to_period("M")
    earlier = series[months <= months[-1] - 6]
    before = float(earlier.iloc[-1]) if len(earlier) else np.nan
    return {"dd": now, "dd_6m_ago": before, "fall_6m": (before - now) / before if np.isfinite(before) and before > 0 else np.nan}


# ---------------------------------------------------------------
# Tier
# ---------------------------------------------------------------

def classify(signals: dict, rules: dict) -> dict:
    """
    Low / Elevated / High from the rules, with every rule that fired. Signals that are None (data not loaded)
    are counted as unavailable; the tier is "based on n of N signals".
    """
    hi, el = rules["high"], rules["elevated"]
    fired = []

    def fire(level, text):
        fired.append((level, text))

    p = signals.get("pledge")
    if p is not None:
        if p["pledged_pct_of_promoter"] >= hi["pledged_pct_of_promoter_min"]:
            fire(HIGH, f"{p['pledged_pct_of_promoter']:.1f}% of promoter shares pledged")
        elif p["pledged_pct_of_promoter"] >= el["pledged_pct_of_promoter_min"]:
            fire(ELEVATED, f"{p['pledged_pct_of_promoter']:.1f}% of promoter shares pledged")
        if np.isfinite(p["change_4q_pp"]) and p["change_4q_pp"] >= el["pledge_increase_4q_pp_min"]:
            fire(ELEVATED, f"pledge up {p['change_4q_pp']:.1f} pp in four quarters")
    s = signals.get("surveillance")
    if s is not None and s["measure"]:
        if s["measure"] == "GSM" and hi["gsm_any_stage"]:
            fire(HIGH, f"under GSM ({s['stage_text']})")
        elif s["measure"] == "ASM-LT" and s["stage"] >= hi["asm_lt_stage_min"]:
            fire(HIGH, f"long-term ASM stage {s['stage']}")
        elif el["any_asm_stage"]:
            fire(ELEVATED, f"under {s['measure']} ({s['stage_text']})")
    b = signals.get("fo_ban")
    if b is not None and b["last_ban"] is not None:
        fire(ELEVATED, f"F&O ban on {b['last_ban']:%d %b %Y} (within {el['fo_ban_days']} days)")
    r = signals.get("rating")
    if r is not None:
        if r["downgrades"] >= hi["downgrades_12m_min"]:
            fire(HIGH, f"{r['downgrades']} rating downgrades in 12 months")
        elif r["below_investment_grade"] and hi["downgrade_below_investment_grade_12m"]:
            fire(HIGH, "downgraded below investment grade in 12 months")
        elif r["downgrades"] >= el["downgrades_12m_min"]:
            fire(ELEVATED, "rating downgrade in 12 months")
        if r["negative_watch"] and el["negative_watch_12m"]:
            fire(ELEVATED, "placed on watch with negative implications")
    a = signals.get("auditor")
    if a is not None:
        severe = [e for e in a["events"] if e in hi["auditor_events_24m"]]
        mild = [e for e in a["events"] if e in el["auditor_events_24m"]]
        if severe:
            fire(HIGH, "auditor: " + ", ".join(severe) + " in 24 months")
        elif mild:
            fire(ELEVATED, "auditor: " + ", ".join(mild) + " in 24 months")
    d = signals.get("merton")
    if d is not None:
        if d["dd"] < hi["merton_dd_below"]:
            fire(HIGH, f"Merton DD {d['dd']:.2f}")
        elif d["dd"] < el["merton_dd_below"]:
            fire(ELEVATED, f"Merton DD {d['dd']:.2f}")
        if np.isfinite(d["fall_6m"]) and d["fall_6m"] >= el["merton_dd_fall_6m_min"]:
            fire(ELEVATED, f"Merton DD down {d['fall_6m']:.0%} in six months")
    c = signals.get("circuit")
    if c is not None and c["lower_circuit_days"] >= el["lower_circuit_days_min"]:
        fire(ELEVATED, f"{c['lower_circuit_days']} lower-circuit days (longest run {c['longest_run']})")
    g = signals.get("group")
    if g is not None and g["same_group_holdings"] >= el["same_group_holdings_min"]:
        fire(ELEVATED, f"{g['same_group_holdings']} holdings in the {g['group']} group")

    tier = HIGH if any(l == HIGH for l, _ in fired) else ELEVATED if fired else LOW
    available = [k for k in SIGNALS if signals.get(k) is not None]
    return {"tier": tier, "fired": fired, "available": available,
            "basis": f"based on {len(available)} of {len(SIGNALS)} signals"}


# ---------------------------------------------------------------
# Pledge margin calls
# ---------------------------------------------------------------

def margin_call(pledged_shares: float, price: float, adv: float, initial_cover: float = INITIAL_COVER,
                trigger_cover: float = TRIGGER_COVER) -> dict:
    """
    Loan sized at today's price at `initial_cover` (pledged value ÷ loan). Lenders can invoke the pledge when cover
    falls to `trigger_cover`, i.e. after a price fall of 1 − trigger/initial. Then either all pledged shares are
    sold, or just enough to restore the initial cover: x = L·(C₀ − C_t) / (P_t·(C₀ − 1)). Selling in days of ADV.
    """
    fall = 1 - trigger_cover / initial_cover
    out = {"trigger_fall": fall, "trigger_price": price * (1 - fall)}
    if not (np.isfinite(pledged_shares) and pledged_shares > 0):
        return {**out, "loan": np.nan, "days_all": np.nan, "shares_to_restore": np.nan, "days_restore": np.nan}
    loan = pledged_shares * price / initial_cover
    p_t = out["trigger_price"]
    restore = loan * (initial_cover - trigger_cover) / (p_t * (initial_cover - 1))
    adv_ok = adv if adv and adv > 0 else np.nan
    return {**out, "loan": loan, "days_all": pledged_shares / adv_ok, "shares_to_restore": restore,
            "days_restore": restore / adv_ok}


# ---------------------------------------------------------------
# Jump overlay on ES
# ---------------------------------------------------------------

def jump_states(weights: dict, jumps: dict) -> list:
    """
    (probability, portfolio shift) for every combination of holdings jumping on the same day: exact for up to
    EXACT_ENUMERATION_MAX holdings with a jump; beyond that, up to two jumps (the rest of the probability mass is
    put on the no-jump state). `jumps` maps ticker → (p, J); the shift is Σ wᵢ·Jᵢ over the holdings that jump.
    """
    active = [(t, p, j) for t, (p, j) in jumps.items() if p > 0 and weights.get(t, 0) != 0]
    if not active:
        return [(1.0, 0.0)]
    max_k = len(active) if len(active) <= EXACT_ENUMERATION_MAX else 2
    states = []
    for k in range(max_k + 1):
        for combo in itertools.combinations(range(len(active)), k):
            prob = 1.0
            shift = 0.0
            for i, (t, p, j) in enumerate(active):
                if i in combo:
                    prob *= p
                    shift += weights[t] * j
                else:
                    prob *= 1 - p
            states.append((prob, shift))
    missing = 1.0 - sum(p for p, _ in states)
    states[0] = (states[0][0] + missing, states[0][1])
    return states


def mixture_var_es(loc: float, scale: float, nu: float, states: list, alpha: float) -> tuple:
    """
    VaR and ES (positive losses) of the mixture Σ πₖ · (loc + shiftₖ + scale·T_ν). VaR solves F(x) = α; ES uses the
    Student-t partial expectation E[T·1{T ≤ z}] = −(ν + z²)/(ν − 1)·f_ν(z), so there is no simulation noise.
    """
    probs = np.array([p for p, _ in states])
    shifts = np.array([s for _, s in states])

    def cdf(x):
        return float((probs * student_t.cdf((x - loc - shifts) / scale, nu)).sum())

    lo, hi = loc + shifts.min() - 50 * scale, loc + shifts.max() + 50 * scale
    x = brentq(lambda v: cdf(v) - alpha, lo, hi, xtol=1e-12)
    z = (x - loc - shifts) / scale
    partial_t = -(nu + z ** 2) / (nu - 1) * student_t.pdf(z, nu)
    tail_mean = (probs * ((loc + shifts) * student_t.cdf(z, nu) + scale * partial_t)).sum()
    return -x, -tail_mean / alpha


def event_adjusted_es(base: dict, weights: dict, jumps: dict, confidence_level: float) -> dict:
    """
    Standard vs event-adjusted 1-day VaR and ES, and each holding's share of the ES gap (the gap with only that
    holding's jump switched on, scaled so the shares add up to the total gap).
    `base` = {"loc", "scale", "nu"} of the one-day return distribution (a scaled Student-t).
    """
    alpha = 1 - confidence_level
    std_var, std_es = mixture_var_es(base["loc"], base["scale"], base["nu"], [(1.0, 0.0)], alpha)
    ev_var, ev_es = mixture_var_es(base["loc"], base["scale"], base["nu"], jump_states(weights, jumps), alpha)
    alone = {}
    for t, (p, j) in jumps.items():
        if p > 0:
            alone[t] = mixture_var_es(base["loc"], base["scale"], base["nu"], jump_states(weights, {t: (p, j)}), alpha)[1] - std_es
    total_alone = sum(alone.values())
    gap = ev_es - std_es
    shares = {t: (v / total_alone * gap if total_alone else 0.0) for t, v in alone.items()}
    return {"var": std_var, "es": std_es, "event_var": ev_var, "event_es": ev_es, "gap": gap, "by_holding": shares}


def base_distribution(var_selected: dict) -> dict:
    """
    One-day return distribution as loc + scale·T_ν: the fitted GARCH(1,1)-t (unit-variance t rescaled), or the
    Student-t maximum-likelihood fit if GARCH did not converge.
    """
    g = var_selected.get("GARCH(1,1)-t", {})
    if not g.get("fallback") and g.get("garch_params") is not None:
        p = g["garch_params"]
        sigma = g["sigma_forecast"]
        return {"loc": p.mu / 100, "scale": sigma * np.sqrt((p.nu - 2) / p.nu), "nu": p.nu, "model": "GARCH(1,1)-t"}
    t = var_selected["Student-t"]
    return {"loc": t["loc"], "scale": t["scale"], "nu": t["degrees_of_freedom"], "model": "Student-t (GARCH fit failed)"}
