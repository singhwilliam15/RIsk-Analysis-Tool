"""
Trust layer: how far each headline number can be relied on.

- TrustedMetric wraps any pillar's headline number as value + 90% range + A-D grade + reasons, sources
  and assumptions. Every pillar reports its headline numbers through it.
- Market-risk uncertainty: 90% ranges for every model's VaR and ES.
    * Historical, Normal, Student-t, Cornish-Fisher: stationary block bootstrap of the returns
      (Politis and Romano, 1994), mean block length by Politis and White (2004) with a fallback.
    * EWMA and FHS: today's volatility is held fixed and the volatility-standardised returns are
      bootstrapped, because resampling raw returns would scramble the volatility the model tracks.
    * GARCH(1,1)-t: parameter uncertainty, from draws of the fitted parameters' asymptotic covariance.
    * Monte Carlo: the same parameter draws plus the simulation's own noise.
- Model risk: ES spread across the models that pass the backtests, the model-risk add-on, lookback
  sensitivity and the "ghost effect" in Historical VaR.
- Grades from explicit rules (GRADE_RULES), documented in docs/methodology.md section 8.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sample_kurtosis, norm, skew as sample_skew

from data_fetcher import suspicious_returns
from garch import GarchParams, PCT, garch_filter, standardized_t_es, standardized_t_quantile
from var_calculator import (
    calculate_all_var,
    cornish_fisher_quantile,
    ewma_volatility,
    fit_student_t,
    LOW_POWER,
    monte_carlo_name,
    student_t_var_es,
)

CI_LEVEL = 0.90
N_BOOT = 500          # bootstrap resamples for the fast models
N_BOOT_REFIT = 200    # for Student-t, which is refitted by maximum likelihood on every resample
N_PARAM_DRAWS = 1000  # GARCH parameter draws
N_MC_DRAWS, MC_SIMS_PER_DRAW = 200, 2000
SEED = 20260930

METHOD_BLOCK = "stationary block bootstrap of returns"
METHOD_RESIDUAL = "stationary block bootstrap of volatility-standardised returns (today's volatility fixed)"
METHOD_PARAMS = "draws from the GARCH parameters' asymptotic covariance"
METHOD_PARAMS_MC = "GARCH parameter draws plus simulation noise"


# ---------------------------------------------------------------
# The reusable interface
# ---------------------------------------------------------------

@dataclass
class TrustedMetric:
    """A headline number with its 90% range, A-D grade and the reasons, sources and assumptions behind it."""
    name: str
    value: float
    low: float = float("nan")
    high: float = float("nan")
    grade: str = "not graded"
    reasons: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)

    @property
    def has_range(self) -> bool:
        return bool(np.isfinite(self.low) and np.isfinite(self.high))

    def range_text(self, fmt=lambda v: f"{v:,.0f}") -> str:
        return f"{CI_LEVEL:.0%} range {fmt(self.low)}–{fmt(self.high)}" if self.has_range else "range not available"

    def format(self, fmt=lambda v: f"{v:,.0f}") -> str:
        """'value (90% range low–high, grade X)'."""
        return f"{fmt(self.value)} ({self.range_text(fmt)}, grade {self.grade})"


# ---------------------------------------------------------------
# Stationary block bootstrap
# ---------------------------------------------------------------

def block_length(returns) -> tuple:
    """
    Mean block length for the stationary bootstrap: the Politis-White (2004, with the Patton, Politis
    and White 2009 correction) estimate, taken on the returns and on the squared returns, whichever is
    longer, so volatility clustering is kept. Falls back to n^(1/3) if the estimate fails.
    Returns (block length, how it was chosen).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    fallback = max(1.0, n ** (1 / 3))
    try:
        from arch.bootstrap import optimal_block_length
        estimate = float(max(optimal_block_length(r)["stationary"].iloc[0],
                             optimal_block_length(r ** 2)["stationary"].iloc[0]))
        if not np.isfinite(estimate) or estimate <= 0:
            raise ValueError("non-finite estimate")
        return float(np.clip(estimate, 1.0, n / 10)), "Politis-White"
    except Exception:
        return float(fallback), "fallback n^(1/3)"


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    """
    Index matrix (n_boot × n) for the stationary bootstrap: each resample is built from blocks that start
    at a random day and have geometric lengths with mean `mean_block`, wrapping around the sample's end.
    """
    p_new = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    starts = rng.integers(0, n, (n_boot, n))
    new_block = rng.random((n_boot, n)) < p_new
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def _empirical_rows(samples: np.ndarray, alpha: float):
    """VaR and ES (positive losses) of each row, the same estimator as var_calculator._empirical_var_es."""
    cutoff = np.percentile(samples, alpha * 100, axis=1)
    in_tail = samples <= cutoff[:, None]
    es = -(np.where(in_tail, samples, 0).sum(axis=1) / in_tail.sum(axis=1))
    return -cutoff, es


def _interval(values) -> tuple:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return float("nan"), float("nan")
    tail = (1 - CI_LEVEL) / 2 * 100
    return float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))


# ---------------------------------------------------------------
# GARCH parameter uncertainty
# ---------------------------------------------------------------

def garch_parameter_draws(params: GarchParams, cov: np.ndarray, n_draws: int, rng, max_rounds: int = 20) -> tuple:
    """
    Draws of (mu, omega, alpha, beta, nu) from N(estimate, asymptotic covariance), keeping only draws that
    are valid GARCH-t parameters (omega > 0, alpha, beta ≥ 0, alpha + beta < 1, nu > 2.05): a truncated
    normal. Returns (draws array K × 5, share of draws rejected).
    """
    mean = np.array([params.mu, params.omega, params.alpha, params.beta, params.nu])
    kept, tried, rejected = [], 0, 0
    for _ in range(max_rounds):
        draws = rng.multivariate_normal(mean, cov, n_draws, method="eigh")
        ok = ((draws[:, 1] > 0) & (draws[:, 2] >= 0) & (draws[:, 3] >= 0) & (draws[:, 2] + draws[:, 3] < 1)
              & (draws[:, 4] > 2.05))
        tried += n_draws
        rejected += int((~ok).sum())
        kept.append(draws[ok])
        if sum(len(k) for k in kept) >= n_draws:
            break
    return np.vstack(kept)[:n_draws], rejected / tried


def garch_next_sigma(returns, draws: np.ndarray) -> np.ndarray:
    """Tomorrow's GARCH volatility (percent) under each parameter draw, filtering all draws at once."""
    r = np.asarray(returns, dtype=float) * PCT
    mu, omega, a, b = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
    eps0 = r[None, :] - mu[:, None]
    variance = np.var(eps0, axis=1, ddof=1)  # the same start as garch.garch_filter
    for t in range(len(r)):
        e = eps0[:, t]
        variance = omega + a * e * e + b * variance
    return np.sqrt(variance)


# ---------------------------------------------------------------
# 90% ranges for every model (1-day, as positive decimal losses)
# ---------------------------------------------------------------

def bootstrap_ranges(returns: pd.Series, confidence_level: float, fits: dict, num_simulations: int,
                     n_boot: int = None, n_boot_refit: int = None, n_draws: int = None, seed: int = SEED) -> dict:
    """
    90% ranges of each model's 1-day VaR and ES (decimal losses). Returns
    {"models": {model: {"var": (lo, hi), "es": (lo, hi), "method", "resamples", "note"}}, "block_length",
    "block_method"}. Model names match var_calculator.calculate_all_var. Resample counts default to the
    module constants, read at call time.
    """
    n_boot = n_boot or N_BOOT
    n_boot_refit = n_boot_refit or N_BOOT_REFIT
    n_draws = n_draws or N_PARAM_DRAWS
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, dtype=float)
    n = len(r)
    alpha = 1.0 - confidence_level
    z = norm.ppf(confidence_level)
    block, block_method = block_length(r)
    idx = stationary_bootstrap_indices(n, n_boot, block, rng)
    samples = r[idx]
    out = {}

    def store(model, var_values, es_values, method, resamples, note=""):
        out[model] = {"var": _interval(var_values), "es": _interval(es_values), "method": method,
                      "resamples": int(resamples), "note": note}

    hv, he = _empirical_rows(samples, alpha)
    store("Historical", hv, he, METHOD_BLOCK, n_boot)

    mu, sd = samples.mean(axis=1), samples.std(axis=1, ddof=1)
    store("Parametric (Normal)", z * sd - mu, norm.pdf(z) / alpha * sd - mu, METHOD_BLOCK, n_boot)

    t_var, t_es = [], []
    for row in samples[:n_boot_refit]:
        fit = fit_student_t(row)
        v, e = student_t_var_es(alpha, fit["nu"], fit["loc"], fit["scale"])
        t_var.append(v)
        t_es.append(e)
    store("Student-t", t_var, t_es, METHOD_BLOCK + " (refitted each time)", n_boot_refit)

    sk = sample_skew(samples, axis=1, bias=False)
    ku = sample_kurtosis(samples, axis=1, bias=False)
    z_cf = cornish_fisher_quantile(norm.ppf(alpha), sk, ku)
    tail_u = alpha * (np.arange(2000) + 0.5) / 2000
    tail_z = cornish_fisher_quantile(norm.ppf(tail_u)[None, :], sk[:, None], ku[:, None]).mean(axis=1)
    store("Cornish-Fisher", -(mu + z_cf * sd), -(mu + tail_z * sd), METHOD_BLOCK, n_boot)

    # Filtered models: bootstrap the standardised residuals; tomorrow's volatility stays as estimated
    series = pd.Series(r)
    sigma, sigma_next = ewma_volatility(series)
    resid = (series / sigma).to_numpy()
    resid = resid[np.isfinite(resid)]
    r_block, _ = block_length(resid)
    r_samples = resid[stationary_bootstrap_indices(len(resid), n_boot, r_block, rng)]
    rms = np.sqrt(np.mean(resid ** 2))
    scale = np.sqrt(np.mean(r_samples ** 2, axis=1)) / rms  # how far the residual scale could be off
    store("EWMA (RiskMetrics)", z * sigma_next * scale, norm.pdf(z) / alpha * sigma_next * scale, METHOD_RESIDUAL, n_boot)
    fv, fe = _empirical_rows(r_samples, alpha)
    store("FHS (EWMA-filtered)", fv * sigma_next, fe * sigma_next, METHOD_RESIDUAL, n_boot)

    garch_fit = fits.get("garch", {})
    mc_name = monte_carlo_name(num_simulations)
    cov = garch_fit.get("param_cov")
    if garch_fit.get("converged") and cov is not None and np.all(np.isfinite(cov)):
        p = garch_fit["params"]
        draws, rejected = garch_parameter_draws(p, cov, n_draws, rng)
        note = f"{rejected:.0%} of draws were outside the valid parameter region and redrawn" if rejected > 0.01 else ""
        if rejected > 0.5:
            note = (f"{rejected:.0%} of draws were invalid: the parameters are poorly identified on this sample, "
                    "so the range is unreliable")
        if len(draws) >= 50:
            sig = garch_next_sigma(r, draws)
            nu = draws[:, 4]
            q = standardized_t_quantile(alpha, nu)
            es_unit = standardized_t_es(alpha, nu)
            store("GARCH(1,1)-t", -(draws[:, 0] + sig * q) / PCT, (sig * es_unit - draws[:, 0]) / PCT,
                  METHOD_PARAMS, len(draws), note)
            mc_v, mc_e = [], []
            for k in range(min(N_MC_DRAWS, len(draws), n_draws)):
                shocks = rng.standard_t(nu[k], MC_SIMS_PER_DRAW) * np.sqrt((nu[k] - 2) / nu[k])
                v, e = _empirical_rows(((draws[k, 0] + sig[k] * shocks) / PCT)[None, :], alpha)
                mc_v.append(v[0])
                mc_e.append(e[0])
            store(mc_name, mc_v, mc_e, METHOD_PARAMS_MC, len(mc_v), note)
        else:
            for model in ("GARCH(1,1)-t", mc_name):
                store(model, [], [], METHOD_PARAMS, 0, "too few valid parameter draws")
    else:
        reason = "GARCH did not converge, so its row shows EWMA" if not garch_fit.get("converged") else \
            "the parameter covariance is not available"
        out["GARCH(1,1)-t"] = {**out["EWMA (RiskMetrics)"], "note": reason}
        store(mc_name, [], [], METHOD_PARAMS_MC, 0, reason)
    return {"models": out, "block_length": block, "block_method": block_method}


def scale_ranges(ranges: dict, var_selected: dict, investment: float) -> pd.DataFrame:
    """
    Turn 1-day decimal ranges into money at the selected horizon. Each bound is multiplied by the
    model's own ratio of t-day to 1-day figure, so multi-day ranges follow the model's scaling rule.
    """
    rows = []
    for model, res in var_selected.items():
        rng_ = ranges["models"].get(model, {"var": (np.nan, np.nan), "es": (np.nan, np.nan), "method": "", "note": ""})
        var_ratio = res["var_scaled_pct"] / res["var_daily_pct"] if res["var_daily_pct"] else np.nan
        es_ratio = res["cvar_scaled_pct"] / res["cvar_daily_pct"] if res["cvar_daily_pct"] else np.nan
        rows.append({
            "Model": model, "VaR": res["var_scaled_amount"],
            "VaR Low": rng_["var"][0] * var_ratio * investment, "VaR High": rng_["var"][1] * var_ratio * investment,
            "ES": res["cvar_scaled_amount"],
            "ES Low": rng_["es"][0] * es_ratio * investment, "ES High": rng_["es"][1] * es_ratio * investment,
            "Range Method": rng_["method"], "Range Note": rng_["note"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Model risk
# ---------------------------------------------------------------

def backtest_status(backtest_table: pd.DataFrame, backtest_name: str) -> str:
    """'pass', 'fail' (VaR tests or ES test failed), 'low power', or 'not tested'."""
    if backtest_table is None or backtest_name not in set(backtest_table["Method"]):
        return "not tested"
    row = backtest_table.set_index("Method").loc[backtest_name]
    if row["Verdict"] == LOW_POWER:
        return "low power"
    if row["Verdict"] == "FAIL" or row.get("ES Test") == "FAIL":
        return "fail"
    return "pass"


def model_risk(var_selected: dict, backtest_table: pd.DataFrame, recommended_model, point_name) -> dict:
    """
    ES range across the models that pass every backtest (VaR tests, and the ES test where it ran), and the
    model-risk add-on = highest passing ES − recommended model's ES. Without a recommendation the add-on is
    not defined and the range is taken across all models, flagged.
    """
    names = list(backtest_table["Method"]) if backtest_table is not None else []
    passing = [point_name(m) for m in names if backtest_status(backtest_table, m) == "pass"]
    es = {m: var_selected[m]["cvar_scaled_amount"] for m in var_selected}
    recommended = point_name(recommended_model) if recommended_model else None
    if passing and recommended in passing:
        values = [es[m] for m in passing]
        return {"basis": "passing models", "models": passing, "low": min(values), "high": max(values),
                "recommended_es": es[recommended], "add_on": max(values) - es[recommended],
                "dispersion": (max(values) - min(values)) / es[recommended] if es[recommended] else np.nan}
    values = list(es.values())
    reference = es[recommended] if recommended else float(np.median(values))
    if not names:
        basis = "all models (not backtested)"
    elif passing:
        basis = "all models (the recommended model fails the ES test)"
    else:
        basis = "all models (none passes every backtest)"
    return {"basis": basis,
            "models": list(es), "low": min(values), "high": max(values),
            "recommended_es": es[recommended] if recommended else np.nan, "add_on": np.nan,
            "dispersion": (max(values) - min(values)) / reference if reference else np.nan}


LOOKBACKS = (("1y", 252), ("2y", 504), ("5y", 1260), ("max", None))


def lookback_sensitivity(history: pd.Series, investment: float, confidence_level: float, holding_period: int) -> pd.DataFrame:
    """
    ES of every model except Monte Carlo on the last 252, 504 and 1,260 trading days and on the full history.
    A lookback longer than the history is reported as not available rather than shortened.
    Cornish-Fisher outside its valid region is noted (and blanked if its ES is not positive), and a window containing reversing spikes
    (likely data errors, see data_fetcher.suspicious_returns) is flagged in the Note column.
    """
    history = history.dropna()
    rows = []
    for label, days in LOOKBACKS:
        if days is not None and len(history) < days:
            rows.append({"Lookback": label, "Days": days, "Available": False, "Note": "history too short"})
            continue
        window = history if days is None else history.iloc[-days:]
        results = calculate_all_var(window, investment, confidence_level, holding_period, num_simulations=1000)
        es = {m: r["cvar_scaled_amount"] for m, r in results.items() if not m.startswith("Monte Carlo")}
        notes = []
        if not results["Cornish-Fisher"]["cf_valid"]:
            # As on the Market page the value is kept with a warning, unless the expansion gives a meaningless ES
            if es["Cornish-Fisher"] <= 0:
                es["Cornish-Fisher"] = np.nan
                notes.append("Cornish-Fisher outside its valid region (ES not meaningful)")
            else:
                notes.append("Cornish-Fisher outside its valid region")
        spikes = suspicious_returns(pd.DataFrame({"Date": window.index, "Close": np.nan, "Returns": window.to_numpy()}))
        if spikes["Reversed"].any():
            notes.append(f"{int(spikes['Reversed'].sum())} reversing spike(s), likely data errors")
        rows.append({"Lookback": label, "Days": len(window), "Available": True, "From": window.index[0], **es,
                     "Note": "; ".join(notes)})
    return pd.DataFrame(rows)


GHOST_HORIZON = 21   # trading days ahead
GHOST_JUMP = 0.05    # assumption: a day-to-day VaR change above 5% counts as a jump


def ghost_effect(window: pd.Series, history: pd.Series, confidence_level: float, horizon: int = GHOST_HORIZON,
                 jump: float = GHOST_JUMP, investment: float = 1.0) -> dict:
    """
    The "ghost effect" in Historical VaR: big losses moving VaR only because they enter or leave the window.

    - Ahead: the tail losses among the oldest `horizon` days of the current window, which leave it within
      `horizon` trading days, and Historical VaR on the window without those days (as if the coming days
      add nothing to the tail).
    - Past: the rolling Historical VaR over the full history with the current window length; a day-to-day
      change above `jump` (relative) is attributed to the return leaving the window ("exit") when that
      return was in the previous window's tail and the new one was not, to the new return ("entry") in the
      opposite case, or to "both".
    """
    alpha = 1.0 - confidence_level
    window = window.dropna()
    n = len(window)
    var_now = -np.percentile(window, alpha * 100)
    oldest = window.iloc[:horizon]
    leaving = oldest[oldest <= -var_now]
    var_after = -np.percentile(window.iloc[horizon:], alpha * 100) if n > horizon else np.nan

    history = history.dropna()
    q = history.rolling(n).quantile(alpha, interpolation="linear")
    var_path = -q
    events = []
    values = history.to_numpy()
    for t in range(n, len(history)):
        prev, cur = var_path.iloc[t - 1], var_path.iloc[t]
        if not (np.isfinite(prev) and np.isfinite(cur)) or prev <= 0 or abs(cur - prev) / prev <= jump:
            continue
        cutoff = q.iloc[t - 1]
        left_tail, new_tail = values[t - n] <= cutoff, values[t] <= cutoff
        cause = "both" if left_tail and new_tail else "exit" if left_tail else "entry" if new_tail else "reordering"
        events.append({"Date": history.index[t], "VaR Before": prev * investment, "VaR After": cur * investment,
                       "Change": (cur - prev) / prev, "Cause": cause,
                       "Return Leaving": values[t - n], "Date Leaving": history.index[t - n], "Return Entering": values[t]})
    events = pd.DataFrame(events, columns=["Date", "VaR Before", "VaR After", "Change", "Cause", "Return Leaving",
                                           "Date Leaving", "Return Entering"])
    return {
        "window_days": n, "horizon": horizon, "jump": jump, "var_now": var_now * investment,
        "var_after": var_after * investment if np.isfinite(var_after) else np.nan,
        "leaving": pd.DataFrame({"Date": leaving.index, "Return": leaving.to_numpy()}),
        "events": events, "exits": int((events["Cause"] == "exit").sum()),
        "rolling_var": var_path.dropna() * investment,
    }


# ---------------------------------------------------------------
# Grades
# ---------------------------------------------------------------

# Thresholds per rule. "Lower is better" rules: at or below the first threshold costs 0 points, at or below
# the second 1 point, above it 2 points. "Higher is better" rules (data quality, sample): the reverse.
GRADE_RULES = {
    "range_width": (0.20, 0.40),     # (90% range high − low) / value
    "dispersion": (0.15, 0.30),      # ES spread across passing models / recommended ES
    "data_quality": (90, 75),        # lowest data-quality score: ≥ 90 → 0, ≥ 75 → 1, below → 2
    "sample": (500, 250),            # returns: ≥ 500 → 0, ≥ 250 → 1, below → 2
    "assumptions": (0.25, 0.50),     # share of inputs that are assumptions
}
GRADES = ("A", "B", "C", "D")
GRADE_BANDS = ((1, "A"), (3, "B"), (5, "C"))  # total deductions ≤ 1 → A, ≤ 3 → B, ≤ 5 → C, more → D
BACKTEST_POINTS = {"pass": 0, "low power": 1, "not tested": 1, "fail": 2}
WORST_IF_FAILED = "C"
DATA_QUALITY_FLOOR = 50


def _lower_is_better(value, limits) -> int:
    if value is None or not np.isfinite(value):
        return 1
    return 0 if value <= limits[0] else 1 if value <= limits[1] else 2


def _higher_is_better(value, limits) -> int:
    if value is None or not np.isfinite(value):
        return 1
    return 0 if value >= limits[0] else 1 if value >= limits[1] else 2


def grade(range_width, dispersion, backtest, data_quality: float, n_obs: int, assumption_share: float,
          sample_label: str = "daily returns") -> tuple:
    """
    A-D grade and the reasons for every point deducted. A missing (NaN) input costs one point. Pass None for a
    rule that does not apply to the metric (e.g. no backtest exists for days to liquidate, no range for a
    deterministic scenario); that rule is skipped. Market-risk metrics use every rule.
    """
    parts = []
    if range_width is not None:
        parts.append(("90% range width", _lower_is_better(range_width, GRADE_RULES["range_width"]),
                      f"{range_width:.0%} of the value" if np.isfinite(range_width) else "not available"))
    if dispersion is not None:
        parts.append(("Model dispersion", _lower_is_better(dispersion, GRADE_RULES["dispersion"]),
                      f"ES spread across models {dispersion:.0%} of the recommended ES" if np.isfinite(dispersion)
                      else "not available"))
    if backtest is not None:
        parts.append(("Backtest", BACKTEST_POINTS[backtest], backtest))
    parts += [
        ("Data quality", _higher_is_better(data_quality, GRADE_RULES["data_quality"]),
         f"lowest holding score {data_quality:.0f}/100" if np.isfinite(data_quality) else "not available"),
        ("Sample length", _higher_is_better(n_obs, GRADE_RULES["sample"]), f"{n_obs} {sample_label}"),
        ("Assumptions", _lower_is_better(assumption_share, GRADE_RULES["assumptions"]),
         f"{assumption_share:.0%} of inputs are assumptions"),
    ]
    total = sum(points for _, points, _ in parts)
    letter = next((g for limit, g in GRADE_BANDS if total <= limit), "D")
    reasons = [f"{name}: {detail} (−{points})" for name, points, detail in parts if points]
    if backtest == "fail" and GRADES.index(letter) < GRADES.index(WORST_IF_FAILED):
        letter = WORST_IF_FAILED
        reasons.append(f"Capped at {WORST_IF_FAILED}: the model fails its backtest")
    if np.isfinite(data_quality) and data_quality < DATA_QUALITY_FLOOR:
        letter = "D"
        reasons.append(f"Set to D: data-quality score below {DATA_QUALITY_FLOOR}")
    return letter, reasons


SQRT_T_MODELS = ("Historical", "FHS (EWMA-filtered)")
FIXED_LAMBDA_MODELS = ("EWMA (RiskMetrics)", "FHS (EWMA-filtered)")


def model_assumptions(model: str, holding_period: int) -> tuple:
    """(inputs, assumptions) of a market-risk model, for the 'share of inputs that are assumptions' rule."""
    inputs, assumptions = ["daily returns (observed)"], []
    if model in FIXED_LAMBDA_MODELS:
        inputs.append("λ = 0.94")
        assumptions.append("λ = 0.94 is the RiskMetrics convention, not estimated")
    if holding_period > 1 and model in SQRT_T_MODELS:
        inputs.append("√t scaling")
        assumptions.append("√t scaling assumes independent daily returns")
    return inputs, assumptions


def market_trust(var_selected: dict, ranges_table: pd.DataFrame, backtest_table: pd.DataFrame, recommended_model,
                 point_name, n_obs: int, data_quality: float, holding_period: int, sources: list) -> dict:
    """
    TrustedMetric for every model's VaR and ES, the model-risk summary and the headline metrics.
    Each model is graded on its own range width and backtest, with the portfolio-wide model dispersion,
    data quality and sample length.
    """
    risk = model_risk(var_selected, backtest_table, recommended_model, point_name)
    backtest_names = {point_name(m): m for m in (backtest_table["Method"] if backtest_table is not None else [])}
    metrics, rows = {}, []
    for row in ranges_table.to_dict("records"):
        model = row["Model"]
        status = backtest_status(backtest_table, backtest_names.get(model, model))
        inputs, assumptions = model_assumptions(model, holding_period)
        width = (row["ES High"] - row["ES Low"]) / row["ES"] if row["ES"] and np.isfinite(row["ES Low"]) else np.nan
        letter, reasons = grade(width, risk["dispersion"], status, data_quality, n_obs, len(assumptions) / len(inputs))
        if row["Range Note"]:
            reasons = reasons + [f"Range: {row['Range Note']}"]
        common = dict(grade=letter, reasons=reasons, sources=sources, assumptions=assumptions)
        metrics[model] = {"VaR": TrustedMetric(f"{model} VaR", row["VaR"], row["VaR Low"], row["VaR High"], **common),
                          "ES": TrustedMetric(f"{model} ES", row["ES"], row["ES Low"], row["ES High"], **common)}
        rows.append({"Model": model, "Backtest": status, "Grade": letter,
                     "Reasons": "; ".join(reasons) or "no deductions"})
    recommended = point_name(recommended_model) if recommended_model else None
    return {"metrics": metrics, "grades": pd.DataFrame(rows), "model_risk": risk,
            "headline_model": recommended or "Historical"}
