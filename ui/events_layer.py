"""Event and governance calculations shared by every page (Pillar 5), with ranges and grades."""

import numpy as np
import pandas as pd

import events as E
from disclosures import DATASETS, normalise_symbol
from trust import TrustedMetric, grade
from ui.context import export

DISCLOSURE_SETS = ("pledges", "surveillance", "fo_ban", "ratings", "auditor_events")
MISSING_DATASET_CAP = 2


def holding_signals(ticker: str, data: dict, as_of, credit_result: dict, liquidity_row: dict, group: str,
                    group_counts: dict, rules: dict) -> dict:
    """Every early-warning signal of one holding, public by `as_of`; None where the data are not loaded."""
    symbol = normalise_symbol(ticker)
    indian = ticker.upper().endswith((".NS", ".BO"))
    signals = {
        "pledge": E.pledge_signal(data["pledges"], symbol, as_of) if indian else None,
        "surveillance": E.surveillance_signal(data["surveillance"], symbol, as_of) if indian else None,
        "fo_ban": E.fo_ban_signal(data["fo_ban"], symbol, as_of, rules["elevated"]["fo_ban_days"]) if indian else None,
        "rating": E.rating_signal(data["ratings"], symbol, as_of, rules["investment_grade_floor"]),
        "auditor": E.auditor_signal(data["auditor_events"], symbol, as_of),
        "merton": E.dd_signal(credit_result.get("rolling"), as_of) if credit_result and not credit_result["financial"] else None,
        "circuit": {"lower_circuit_days": int(liquidity_row["Lower-Circuit Days"]), "longest_run": int(liquidity_row["Longest Run"])}
        if liquidity_row is not None and indian else None,
        "group": {"group": group, "same_group_holdings": group_counts.get(group, 0)} if group else None,
    }
    return signals


def compute_events(ctx):
    """Early-warning panel, event tiers, pledge margin calls and the jump overlay on ES."""
    rules = E.load_rules()
    data = ctx.disclosures["data"]
    as_of = ctx.prices_as_of
    tags = ctx.group_tags.assign(Group=ctx.group_tags["Group"].fillna("").astype(str).str.strip())
    groups = dict(zip(tags["Ticker"], tags["Group"]))
    counts = tags.loc[tags["Group"] != "", "Group"].value_counts().to_dict()
    liquidity = ctx.liquidity_holdings.set_index("Ticker")
    capacity = ctx.liquidity_capacity.set_index("Ticker")

    rows, signals_by_ticker, tiers, margin = [], {}, {}, {}
    for p in ctx.positions.to_dict("records"):
        t = p["Ticker"]
        liq = liquidity.loc[t].to_dict() if t in liquidity.index else None
        signals = holding_signals(t, data, as_of, ctx.credit_results.get(t), liq, groups.get(t, ""), counts, rules)
        result = E.classify(signals, rules)
        signals_by_ticker[t], tiers[t] = signals, result
        pledge = signals["pledge"]
        shares_pledged = np.nan
        if pledge is not None:
            if np.isfinite(pledge["pledged_shares"]):
                shares_pledged = pledge["pledged_shares"]
            else:
                shares_out = (ctx.fundamentals_by_ticker[t]["profile"].get("shares_outstanding") or {}).get("value")
                if shares_out and np.isfinite(pledge["pledged_pct_of_total"]):
                    shares_pledged = pledge["pledged_pct_of_total"] / 100 * shares_out
        margin[t] = E.margin_call(shares_pledged, p["Price"], capacity.loc[t, "ADV 60d"], ctx.initial_cover, ctx.trigger_cover)
        rows.append({"Ticker": t, "Value": p["Value"], "Tier": result["tier"], "Basis": result["basis"],
                     "Reasons": "; ".join(text for _, text in result["fired"]) or "no rule fired"})
    panel = pd.DataFrame(rows)

    weights = dict(zip(ctx.positions["Ticker"], ctx.positions["Weight"]))
    jumps = {t: ctx.jump_settings[r["tier"]] for t, r in tiers.items()}
    base = E.base_distribution(ctx.var_selected)
    overlay = E.event_adjusted_es(base, weights, jumps, ctx.confidence_level)
    lo = E.event_adjusted_es(base, weights, {t: (p * 0.5, j) for t, (p, j) in jumps.items()}, ctx.confidence_level)["event_es"]
    hi = E.event_adjusted_es(base, weights, {t: (min(p * 2, 1.0), j) for t, (p, j) in jumps.items()}, ctx.confidence_level)["event_es"]
    inv = ctx.investment_amount

    # --- trust -------------------------------------------------------------------------
    loaded = [d for d in DISCLOSURE_SETS if not data[d].empty]
    missing = [DATASETS[d].title for d in DISCLOSURE_SETS if data[d].empty]
    extra = (("Disclosure data", min(len(missing), MISSING_DATASET_CAP), "not loaded: " + ", ".join(missing)),) if missing else ()
    data_q = float(min(q["score"] for q in ctx.quality.values())) if ctx.quality else np.nan
    n_days = len(ctx.returns)
    sources = [f"{DATASETS[d].title}: " + ", ".join(sorted(set(data[d]["file"]))) for d in loaded] + [ctx.data_note]
    # Only the tiers some holding actually falls in bring their jump assumptions into the figures
    used_tiers = {r["tier"] for r in tiers.values()}
    jump_assumptions = [f"{tier}: p = {p:.2%}/day, J = {j:.0%}" for tier, (p, j) in ctx.jump_settings.items()
                        if p > 0 and tier in used_tiers]

    def metric(name, value, low, high, assumptions, n_inputs, has_range=True, label="range across jump assumptions (½× to 2× p)"):
        if not has_range:
            width = None
        elif value == 0 and low == 0 and high == 0:
            width = 0.0  # no jump applies: the figure is exactly zero, not unknown
        else:
            width = (high - low) / abs(value) if value and np.isfinite(low) else np.nan
        letter, reasons = grade(width, None, None, data_q, n_days, len(assumptions) / n_inputs, extra=extra)
        return TrustedMetric(name, value, low, high, letter, reasons, sources, assumptions, range_label=label)

    flagged_value = float(panel.loc[panel["Tier"] != E.LOW, "Value"].sum()) / float(panel["Value"].sum())
    metrics = {
        "event_es": metric(f"Event-adjusted ES ({ctx.cl_label}, 1-day)", overlay["event_es"] * inv, lo * inv, hi * inv,
                           jump_assumptions, 3),
        "gap": metric("ES added by event risk", overlay["gap"] * inv, (lo - overlay["es"]) * inv, (hi - overlay["es"]) * inv,
                      jump_assumptions, 3),
        "flagged": metric("Value in Elevated or High tier", flagged_value, np.nan, np.nan,
                          ["tier thresholds (config/event_rules.json)"], 2, has_range=False),
        "high": metric("High-tier holdings", float((panel["Tier"] == E.HIGH).sum()), np.nan, np.nan,
                       ["tier thresholds (config/event_rules.json)"], 2, has_range=False),
    }
    export(ctx, {"event_panel": panel, "event_signals": signals_by_ticker, "event_tiers": tiers, "event_margin": margin,
                 "event_overlay": overlay, "event_base": base, "event_metrics": metrics, "event_loaded": loaded,
                 "event_missing": missing, "event_jumps": jumps})
