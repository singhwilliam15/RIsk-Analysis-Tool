"""
Debt positions (corporate bonds / NCDs and loans): pricing, durations, expected credit loss and stressed spread
losses. Equity holders bear default risk through the share price (Phase B); bond holders bear it as an explicit
credit loss, which is why the credit link adds loss only for debt.

- Price of a fixed-coupon bond at yield y = r + spread, annual coupons, clean price ignoring accrued interest:
  P = Σ c·F/(1+y)^tᵢ + F/(1+y)^T, with tᵢ the years to each remaining coupon date.
- Modified (= spread, for a fixed coupon) duration D = −(1/P)·dP/dy and convexity C = (1/P)·d²P/dy².
- Expected loss over one year: EL = EAD × PD × LGD, EAD = market value, PD = the rating's published 1-year default
  rate (credit.rating_implied_pd), LGD by seniority (assumptions; Basel foundation-IRB values where they exist).
- Stressed spread change Δs:
    * issuer also held as equity with Merton inputs: from the stressed risk-neutral PD, s = −ln(1 − PD·LGD)/T, so
      Δs = s(stressed) − s(today) — the link from equity distress to the bond;
    * otherwise: a widening per rating category for a severe market fall (assumption table), scaled by the
      scenario's market move.
  Loss = P(y) − P(y + Δs), by full repricing; the duration–convexity approximation is reported next to it.

Methods: docs/methodology.md section 10.7.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import credit as C
from events import RATING_SCALE, rating_rank

CONFIG_PATH = Path(__file__).parent / "config" / "debt_assumptions.json"
COLUMNS = ["Name", "Issuer Ticker", "Face Value", "Coupon %", "Maturity", "Rating", "Seniority", "Spread bps"]
SENIORITIES = ("secured", "senior unsecured", "subordinated")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def coupon_times(maturity, as_of) -> np.ndarray:
    """Years from `as_of` to each remaining annual coupon date (counting back from maturity)."""
    maturity, as_of = pd.Timestamp(maturity), pd.Timestamp(as_of)
    dates, k = [], 0
    while maturity - pd.DateOffset(years=k) > as_of:  # coupon dates by calendar, strictly after the valuation date
        dates.append(maturity - pd.DateOffset(years=k))
        k += 1
    return np.array(sorted((d - as_of).days / 365.25 for d in dates))


def price(face: float, coupon: float, maturity, as_of, y: float) -> float:
    """Clean price (currency) of `face` with annual coupon rate `coupon` at annually compounded yield `y`."""
    t = coupon_times(maturity, as_of)
    if not len(t):
        return np.nan
    cf = np.full(len(t), coupon * face)
    cf[-1] += face
    return float((cf / (1 + y) ** t).sum())


def duration_convexity(face: float, coupon: float, maturity, as_of, y: float) -> tuple:
    """(modified duration, convexity) at yield `y`, analytically."""
    t = coupon_times(maturity, as_of)
    cf = np.full(len(t), coupon * face)
    cf[-1] += face
    p = (cf / (1 + y) ** t).sum()
    d_mod = (t * cf / (1 + y) ** (t + 1)).sum() / p
    conv = (t * (t + 1) * cf / (1 + y) ** (t + 2)).sum() / p
    return float(d_mod), float(conv)


def spread_from_pd(pd_q: float, lgd: float, horizon: float = 1.0) -> float:
    """Credit spread implied by a risk-neutral default probability over `horizon`: s = −ln(1 − PD·LGD)/T."""
    if not np.isfinite(pd_q):
        return np.nan
    return float(-np.log(max(1e-12, 1 - min(pd_q, 1.0) * lgd)) / horizon)


def bucket_widening(rating, market_move: float, config: dict) -> float:
    """Assumed spread widening (decimal) for the rating's category, scaled by |market move| ÷ the reference fall."""
    rank = rating_rank(rating)
    if rank < 0:
        category = "unrated"
    else:
        category = RATING_SCALE[rank].rstrip("+-")
    bps = config["widening_bps_severe"].get(category, config["widening_bps_severe"]["unrated"])
    scale = min(abs(min(market_move, 0.0)) / config["severe_market_fall"], config["max_scale"])
    return bps / 1e4 * scale


def holding_summary(row: dict, r: float, as_of, rates: pd.DataFrame, config: dict) -> dict:
    """Today's figures for one debt holding: price, market value, durations, PD, LGD and one-year EL."""
    face, coupon, spread = float(row["Face Value"]), float(row["Coupon %"]) / 100, float(row["Spread bps"]) / 1e4
    y = r + spread
    mv = price(face, coupon, row["Maturity"], as_of, y)
    d_mod, conv = duration_convexity(face, coupon, row["Maturity"], as_of, y) if np.isfinite(mv) else (np.nan, np.nan)
    agency = str(row["Rating"]).split()[0].strip("[]") if str(row["Rating"]).strip() else None
    implied = C.rating_implied_pd(row["Rating"], agency, rates)
    lgd = config["lgd"][str(row["Seniority"]).lower()]
    pd_1y = implied["pd"] if implied else np.nan
    return {"Name": row["Name"], "Issuer": row.get("Issuer Ticker") or "", "Market Value": mv, "Yield": y,
            "Modified Duration": d_mod, "Convexity": conv, "PD (1y, agency)": pd_1y, "LGD": lgd,
            "Expected Loss (1y)": mv * pd_1y * lgd if np.isfinite(pd_1y) else np.nan,
            "PD source": (implied["study"] + (" (CRISIL used: no study for this agency)" if implied["fallback"] else ""))
            if implied else "rating not readable", "Jump-to-Default Loss": mv * lgd}


def stressed_loss(row: dict, summary: dict, r: float, as_of, delta_s: float) -> dict:
    """Repricing loss for a spread change `delta_s`, the duration–convexity approximation, and the new price."""
    face, coupon = float(row["Face Value"]), float(row["Coupon %"]) / 100
    p0 = summary["Market Value"]
    p1 = price(face, coupon, row["Maturity"], as_of, summary["Yield"] + delta_s)
    approx = p0 * (summary["Modified Duration"] * delta_s - 0.5 * summary["Convexity"] * delta_s ** 2)
    return {"loss": p0 - p1, "approx": approx, "price_after": p1}


def prepare(frame: pd.DataFrame, r: float, as_of, rates: pd.DataFrame = None, config: dict = None) -> pd.DataFrame:
    """Validated debt rows merged with today's figures (price, durations, PD, LGD, EL), ready for the stress engine."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    config = config or load_config()
    rates = rates if rates is not None else C.load_default_rates()
    rows = []
    for row in frame.to_dict("records"):
        summary = holding_summary(row, r, as_of, rates, config)
        rows.append({**row, **summary})
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> tuple:
    """Clean debt rows and errors (blank rows skipped)."""
    rows, errors = [], []
    for i, r in frame.reset_index(drop=True).iterrows():
        if pd.isna(r.get("Name")) or not str(r.get("Name")).strip():
            continue
        line = i + 1
        try:
            face, coupon, spread = float(r["Face Value"]), float(r["Coupon %"]), float(r["Spread bps"])
            maturity = pd.Timestamp(r["Maturity"])
        except (TypeError, ValueError):
            errors.append(f"Debt row {line}: face value, coupon, spread must be numbers and maturity a date.")
            continue
        seniority = str(r["Seniority"]).strip().lower()
        if seniority not in SENIORITIES:
            errors.append(f"Debt row {line}: seniority must be one of {', '.join(SENIORITIES)}.")
            continue
        if face <= 0 or coupon < 0 or spread < 0:
            errors.append(f"Debt row {line}: face value must be positive; coupon and spread non-negative.")
            continue
        rows.append({"Name": str(r["Name"]).strip(), "Issuer Ticker": str(r.get("Issuer Ticker") or "").strip().upper(),
                     "Face Value": face, "Coupon %": coupon, "Maturity": maturity, "Rating": str(r.get("Rating") or "").strip(),
                     "Seniority": seniority, "Spread bps": spread})
    return pd.DataFrame(rows, columns=COLUMNS), errors
