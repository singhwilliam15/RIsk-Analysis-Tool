"""
Liquidity risk pillar (Pillar 2): how fast the positions can be sold, at what cost, and what happens when
volume dries up or an Indian stock locks at its lower circuit.

- Trading capacity: average daily volume and traded value; days to liquidate at a participation rate;
  share of the portfolio sellable in 1/5/10 days.
- SEBI/AMFI-style stress test: days to sell 25% / 50% of the portfolio pro rata at 10% of 3-month average
  volume, with the least liquid 20% of the portfolio (by value) excluded.
- Stressed volume: volume in each crisis window relative to the 120 trading days before it.
- Spread and cost: Corwin-Schultz (2012) high-low spread; Bangia et al. (1999) exogenous spread cost;
  square-root-law market impact; the liquidity-adjusted VaR waterfall.
- Amihud (2002) illiquidity.
- Circuit-lock risk (India): price band, past lower-circuit days and the exit-freeze loss.

Every parameter that is not estimated from data is an assumption, labelled in the UI and justified in
docs/methodology.md section 9.
"""

import numpy as np
import pandas as pd

# Assumptions and conventions (editable in the sidebar)
DEFAULT_PARTICIPATION = 0.20   # share of a day's volume one can trade without dominating it
AMFI_PARTICIPATION = 0.10      # AMFI/SEBI liquidity stress-test convention
AMFI_ADV_DAYS = 63             # "3-month average volume" in trading days
AMFI_EXCLUDE = 0.20            # least liquid 20% of the portfolio excluded
AMFI_FRACTIONS = (0.25, 0.50)
BANGIA_K = 3.0                 # spread volatility multiplier (Bangia et al., 1999)
IMPACT_Y = 1.0                 # square-root-law constant
PRE_WINDOW = 120               # trading days before a crisis window used as "normal" volume
MAX_STRESS_FACTOR = 1.0        # assumption: crisis volume is never taken as higher than normal
MIN_WINDOW_DAYS = 10
FREEZE_MIN = 3                 # exit-freeze scenario: at least this many consecutive lower circuits
BANDS = (0.02, 0.05, 0.10, 0.20)
BAND_TOLERANCE = 0.001         # a move within ±0.1 percentage points of a band counts as hitting it
MIN_BAND_HITS = 3              # band hits needed before a band is inferred from history
SELL_DAYS = (1, 5, 10)
CRORE = 1e7
BASIS_POINTS = 1e4


def _volume(df: pd.DataFrame) -> pd.Series:
    return df["Volume"] if "Volume" in df else pd.Series(np.nan, index=df.index)


def average_volume(df: pd.DataFrame, days: int) -> tuple:
    """Average daily volume (shares) and traded value (volume × close) over the last `days` rows; NaN days skipped."""
    tail = df.tail(days)
    volume = _volume(tail)
    return float(volume.mean()), float((volume * tail["Close"]).mean())


# ---------------------------------------------------------------
# Trading capacity
# ---------------------------------------------------------------

def days_to_liquidate(shares, adv, participation: float):
    """Days to sell `shares` at `participation` of average daily volume; infinite without volume."""
    shares, adv = np.asarray(shares, dtype=float), np.asarray(adv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        days = shares / (participation * adv)
    return np.where(np.isfinite(adv) & (adv > 0), days, np.inf)


def sellable_share(values, days, horizons=SELL_DAYS) -> dict:
    """Share of portfolio value that can be sold within each horizon, selling every holding at its own pace."""
    values, days = np.asarray(values, dtype=float), np.asarray(days, dtype=float)
    total = values.sum()
    return {h: float((values * np.minimum(1.0, np.where(days > 0, h / days, 1.0))).sum() / total) for h in horizons}


def trading_capacity(positions: pd.DataFrame, frames: dict, participation: float = DEFAULT_PARTICIPATION) -> pd.DataFrame:
    """One row per holding: ADV and traded value (20/60 days), days to liquidate the whole position."""
    rows = []
    for p in positions.to_dict("records"):
        df = frames[p["Ticker"]]
        adv20, advt20 = average_volume(df, 20)
        adv60, advt60 = average_volume(df, 60)
        rows.append({"Ticker": p["Ticker"], "Quantity": p["Quantity"], "Value": p["Value"], "Weight": p["Weight"],
                     "ADV 20d": adv20, "ADV 60d": adv60, "Traded Value 20d": advt20, "Traded Value 60d": advt60,
                     "Position / ADV 60d": p["Quantity"] / adv60 if adv60 > 0 else np.inf,
                     "Days to Liquidate": float(days_to_liquidate(p["Quantity"], adv60, participation))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# SEBI/AMFI-style stress test
# ---------------------------------------------------------------

def amfi_stress(positions: pd.DataFrame, adv: dict, fraction: float, participation: float = AMFI_PARTICIPATION,
                exclude: float = AMFI_EXCLUDE) -> dict:
    """
    Days to sell `fraction` of every holding (pro rata) at `participation` of its average volume.

    Holdings are ranked from least to most liquid (days to liquidate the whole position) and removed until
    `exclude` of the portfolio value is gone; the holding on the boundary is removed only in part, and its
    remaining part is sold pro rata like the others. Days = the slowest remaining holding.
    Returns {"days", "excluded" (table of what was removed), "binding" (slowest remaining holding)}.
    """
    table = positions[["Ticker", "Quantity", "Value"]].copy()
    table["ADV"] = [adv.get(t, np.nan) for t in table["Ticker"]]
    table["Full Days"] = days_to_liquidate(table["Quantity"], table["ADV"], participation)
    table = table.sort_values("Full Days", ascending=False, kind="stable").reset_index(drop=True)
    total = table["Value"].sum()
    to_remove = exclude * total
    kept = []
    for value in table["Value"]:
        removed = min(value, max(to_remove, 0.0))
        to_remove -= removed
        kept.append(1.0 - removed / value if value > 0 else 0.0)
    table["Kept Share"] = kept
    table["Days"] = fraction * table["Full Days"] * table["Kept Share"]
    remaining = table[table["Kept Share"] > 1e-12]
    if remaining.empty:
        return {"days": 0.0, "table": table, "binding": None}
    binding = remaining.loc[remaining["Days"].idxmax()]
    return {"days": float(binding["Days"]), "table": table, "binding": binding["Ticker"]}


# ---------------------------------------------------------------
# Stressed volume
# ---------------------------------------------------------------

def stressed_volume(volume: pd.Series, scenarios: pd.DataFrame, pre_window: int = PRE_WINDOW) -> dict:
    """
    For each crisis window: average volume in the window ÷ average volume over the `pre_window` trading days
    before it. Comparing with the days just before cancels long-run growth in volume and old share splits.
    `measured` is the median ratio over the windows with enough data (at least MIN_WINDOW_DAYS in the window
    and half of `pre_window` before it); `factor`, the one applied, is capped at MAX_STRESS_FACTOR.
    """
    volume = volume.dropna()
    volume = volume[volume > 0]
    rows = []
    for s in scenarios.to_dict("records"):
        start, end = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
        inside = volume[(volume.index >= start) & (volume.index <= end)]
        before = volume[volume.index < start].tail(pre_window)
        usable = len(inside) >= MIN_WINDOW_DAYS and len(before) >= pre_window // 2
        rows.append({"Scenario": s["scenario"], "Start": start, "End": end, "Window Days": len(inside),
                     "Window Avg": inside.mean() if len(inside) else np.nan,
                     "Before Avg": before.mean() if len(before) else np.nan,
                     "Ratio": inside.mean() / before.mean() if usable else np.nan, "Used": usable})
    table = pd.DataFrame(rows, columns=["Scenario", "Start", "End", "Window Days", "Window Avg", "Before Avg", "Ratio", "Used"])
    used = table.loc[table["Used"], "Ratio"]
    measured = float(used.median()) if len(used) else np.nan
    # Assumption: a crisis never makes a position easier to sell. Large caps often trade more in a sell-off, but
    # that volume is other sellers' too, so the factor applied is capped at 1 (the measured median is still shown).
    return {"table": table, "measured": measured, "factor": min(measured, MAX_STRESS_FACTOR) if len(used) else np.nan,
            "windows_used": int(len(used))}


# ---------------------------------------------------------------
# Spread and cost
# ---------------------------------------------------------------

_CS_DENOM = 3 - 2 * np.sqrt(2)


def corwin_schultz(high, low, close=None, clip: bool = True) -> pd.Series:
    """
    Corwin and Schultz (2012) two-day high-low spread estimate, one value per pair of days (dated on the
    second day). With closes, the second day's range is shifted for overnight moves as in the paper.
    With `clip`, negative two-day estimates are set to zero; without it they are kept, so they can be averaged
    first (the paper's alternative, used by spread_profile).
    """
    h, lo = pd.Series(high, dtype=float).copy(), pd.Series(low, dtype=float).copy()
    if close is not None:
        c = pd.Series(close, dtype=float).shift(1)
        up = (lo - c).where(lo > c, 0.0).fillna(0.0)      # gap up: the low is above yesterday's close
        down = (c - h).where(h < c, 0.0).fillna(0.0)      # gap down: the high is below yesterday's close
        h, lo = h - up + down, lo - up + down
    log_hl = np.log(h / lo) ** 2
    beta = log_hl + log_hl.shift(1)
    gamma = np.log(np.maximum(h, h.shift(1)) / np.minimum(lo, lo.shift(1))) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _CS_DENOM - np.sqrt(gamma / _CS_DENOM)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0) if clip else spread


def spread_profile(df: pd.DataFrame, months: int = 12) -> dict:
    """
    Mean and standard deviation of the monthly Corwin-Schultz spread over the last `months` months. The two-day
    estimates are averaged within each month with their negative values kept, and only the monthly average is
    floored at zero: setting each negative two-day estimate to zero first biases the spread upwards, which on
    simulated prices with no spread at all gave 0.39% instead of 0.12%. NaN when high/low data are missing.
    """
    if not {"High", "Low"} <= set(df.columns) or df[["High", "Low"]].isna().all().any():
        return {"mean": np.nan, "std": np.nan, "monthly": pd.Series(dtype=float), "source": "not available (no high/low)"}
    daily = corwin_schultz(df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), clip=False)
    daily.index = pd.to_datetime(df["Date"]).to_numpy()
    monthly = daily.resample("ME").mean().dropna().clip(lower=0).tail(months)
    return {"mean": float(monthly.mean()), "std": float(monthly.std(ddof=1)) if len(monthly) > 1 else 0.0,
            "monthly": monthly, "source": "Corwin-Schultz estimate"}


def bangia_cost(value: float, mean_spread: float, std_spread: float, k: float = BANGIA_K) -> float:
    """Exogenous liquidity cost (Bangia et al., 1999): ½ × value × (mean relative spread + k × its std)."""
    return 0.5 * value * (mean_spread + k * std_spread)


def sqrt_impact(value: float, sigma_daily: float, shares: float, daily_volume: float, y: float = IMPACT_Y) -> float:
    """Square-root-law impact cost: Y × σ_daily × √(shares / daily volume) × value; capped at the whole value."""
    if not (daily_volume > 0) or not np.isfinite(sigma_daily):
        return np.nan
    return float(min(1.0, y * sigma_daily * np.sqrt(shares / daily_volume)) * value)


# ---------------------------------------------------------------
# Amihud illiquidity
# ---------------------------------------------------------------

def amihud(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling Amihud (2002) illiquidity: mean of |daily return| / daily traded value, in basis points of price
    move per ₹1 crore (10 million currency units) traded. Each value uses only days up to its date.
    """
    traded = _volume(df) * df["Close"]
    ratio = (df["Returns"].abs() / traded.where(traded > 0)) * CRORE * BASIS_POINTS
    series = ratio.rolling(window, min_periods=window // 2).mean()
    series.index = pd.to_datetime(df["Date"]).to_numpy()
    return series


# ---------------------------------------------------------------
# Circuit-lock risk (India)
# ---------------------------------------------------------------

def _at_extreme(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df or df[column].isna().all():
        return pd.Series(True, index=df.index)  # without high/low data, rely on the return alone
    return (df["Close"] - df[column]).abs() <= df["Close"] * 0.001


def infer_band(df: pd.DataFrame) -> tuple:
    """
    Infer the price band from history: the band (2/5/10/20%) hit most often, counting days that close at the
    low with a return of −band, or at the high with +band, within BAND_TOLERANCE. Needs MIN_BAND_HITS hits.
    Returns (band or None, hits).
    """
    r = df["Returns"]
    at_low, at_high = _at_extreme(df, "Low"), _at_extreme(df, "High")
    counts = {b: int((((r + b).abs() <= BAND_TOLERANCE) & at_low).sum() + (((r - b).abs() <= BAND_TOLERANCE) & at_high).sum())
              for b in BANDS}
    band = max(counts, key=counts.get)
    return (band, counts[band]) if counts[band] >= MIN_BAND_HITS else (None, counts[band])


def lower_circuit_days(df: pd.DataFrame, band: float) -> pd.Series:
    """Days that closed at the low after falling by the band (within BAND_TOLERANCE)."""
    return (df["Returns"] <= -band + BAND_TOLERANCE) & _at_extreme(df, "Low")


def longest_run(flags) -> int:
    best = run = 0
    for flag in np.asarray(flags, dtype=bool):
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def circuit_lock(df: pd.DataFrame, official_band=None, freeze_days: int = None) -> dict:
    """
    Circuit-lock risk of one holding. The band comes from the official file when given, otherwise it is inferred
    (labelled). Exit freeze: N consecutive lower circuits (default: the longest past run, at least FREEZE_MIN)
    during which the position cannot be sold; loss = 1 − (1 − band)^N.
    """
    if official_band is not None and np.isfinite(official_band):
        band, source = float(official_band), "official price-band file"
    else:
        band, hits = infer_band(df)
        source = f"inferred from {hits} band hits" if band is not None else "no fixed band found"
    if band is None:
        return {"band": np.nan, "source": source, "circuit_days": 0, "longest_run": 0, "freeze_days": 0, "loss_pct": 0.0}
    flags = lower_circuit_days(df, band)
    run = longest_run(flags)
    n = int(freeze_days) if freeze_days else max(run, FREEZE_MIN)
    return {"band": band, "source": source, "circuit_days": int(flags.sum()), "longest_run": run,
            "freeze_days": n, "loss_pct": 1 - (1 - band) ** n}


# ---------------------------------------------------------------
# Liquidity-adjusted VaR
# ---------------------------------------------------------------

def lvar_waterfall(var_amount: float, spread_cost: float, impact_cost: float, circuit_loss: float) -> pd.DataFrame:
    """
    VaR → + spread cost → + market impact → + circuit-lock add-on. The add-on is the loss beyond VaR if every
    banded holding is frozen at its lower circuit for its exit-freeze scenario: max(0, circuit loss − VaR).
    It is a stress add-on, not a probability-based figure.
    """
    add_on = max(0.0, circuit_loss - var_amount) if np.isfinite(circuit_loss) else 0.0
    steps = [("VaR", var_amount), ("+ Spread cost", spread_cost), ("+ Market impact", impact_cost),
             ("+ Circuit-lock add-on", add_on)]
    table = pd.DataFrame(steps, columns=["Step", "Amount"])
    table["Amount"] = table["Amount"].fillna(0.0)
    table["Cumulative"] = table["Amount"].cumsum()
    return table


def rolling_adv(df: pd.DataFrame, days: int, lookback: int = 252) -> pd.Series:
    """Rolling `days`-day average volume at each of the last `lookback` dates (for ranges on days to liquidate)."""
    vol = _volume(df).rolling(days, min_periods=days // 2).mean()
    return vol.tail(lookback).dropna()
