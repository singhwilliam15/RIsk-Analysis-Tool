"""
GARCH(1,1) with Student-t innovations, and Filtered Historical Simulation (FHS).

GARCH is fitted with the `arch` package on percentage returns (×100) for numerical
stability; every value this module returns is converted back to decimal returns.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PCT = 100.0


def standardized_t_quantile(p, nu):
    """p-quantile of a Student-t rescaled to unit variance (the distribution `arch` fits)."""
    return student_t.ppf(p, nu) * np.sqrt((nu - 2) / nu)


def standardized_t_es(alpha, nu):
    """
    Expected Shortfall of a unit-variance Student-t at tail probability alpha, as a positive
    multiple of sigma: φ_ν(q)/α · (ν + q²)/(ν − 1) · √((ν − 2)/ν), with q the (1 − α) t-quantile.
    """
    q = student_t.ppf(1 - alpha, nu)
    return student_t.pdf(q, nu) / alpha * (nu + q ** 2) / (nu - 1) * np.sqrt((nu - 2) / nu)


@dataclass
class GarchParams:
    """GARCH(1,1)-t parameters in percentage-return units."""
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float

    def is_valid(self) -> bool:
        values = (self.mu, self.omega, self.alpha, self.beta, self.nu)
        return (all(np.isfinite(values)) and self.omega > 0 and self.alpha >= 0 and self.beta >= 0
                and self.alpha + self.beta < 1 and self.nu > 2.05)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta


def fit_garch_t(returns: pd.Series) -> dict:
    """
    Fit GARCH(1,1) with Student-t errors and a constant mean.
    Returns {"params": GarchParams | None, "converged": bool, "error": str | None, "param_cov": ndarray | None}.
    A fit counts as converged only if the optimiser succeeded and the parameters are stationary.
    `param_cov` is the robust (sandwich) asymptotic covariance of (mu, omega, alpha, beta, nu), in the
    same percentage units as the parameters; the Trust layer uses it for parameter uncertainty.
    """
    try:
        from arch import arch_model
        model = arch_model(np.asarray(returns, dtype=float) * PCT, mean="Constant", vol="GARCH",
                           p=1, q=1, dist="t", rescale=False)
        res = model.fit(disp="off", show_warning=False)
        p = res.params
        params = GarchParams(float(p["mu"]), float(p["omega"]), float(p["alpha[1]"]), float(p["beta[1]"]), float(p["nu"]))
        converged = res.convergence_flag == 0 and params.is_valid()
        try:
            param_cov = res.param_cov.loc[GARCH_PARAM_NAMES, GARCH_PARAM_NAMES].to_numpy(dtype=float)
        except Exception:
            param_cov = None
        return {"params": params if converged else None, "converged": converged,
                "error": None if converged else "optimiser did not converge or parameters are not stationary",
                "param_cov": param_cov if converged else None}
    except Exception as exc:
        return {"params": None, "converged": False, "error": str(exc), "param_cov": None}


GARCH_PARAM_NAMES = ["mu", "omega", "alpha[1]", "beta[1]", "nu"]


def garch_filter(returns, params: GarchParams, initial_variance: float = None) -> np.ndarray:
    """
    Conditional volatility in percent: element t is the forecast for day t using returns before t,
    and the last element (index n) is the forecast for the next, unseen day.
    σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1),  ε = r − μ.
    """
    eps = np.asarray(returns, dtype=float) * PCT - params.mu
    variance = np.empty(len(eps) + 1)
    variance[0] = initial_variance if initial_variance is not None else (np.var(eps, ddof=1) if len(eps) > 1 else params.omega / (1 - params.persistence))
    for i, e in enumerate(eps):
        variance[i + 1] = params.omega + params.alpha * e * e + params.beta * variance[i]
    return np.sqrt(variance)


def garch_variance_term_structure(params: GarchParams, next_variance: float, horizon: int) -> np.ndarray:
    """Expected daily variances for the next `horizon` days: v(1) = σ²(T+1), v(h) = ω + (α + β)·v(h−1)."""
    v = np.empty(horizon)
    v[0] = next_variance
    for h in range(1, horizon):
        v[h] = params.omega + params.persistence * v[h - 1]
    return v


def simulate_garch_paths(params: GarchParams, next_variance: float, horizon: int, num_simulations: int, seed=None) -> np.ndarray:
    """
    Simulate `num_simulations` GARCH-t return paths of length `horizon` (decimal returns),
    with volatility updating along each path. Shape: (num_simulations, horizon).
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(params.nu, size=(num_simulations, horizon)) * np.sqrt((params.nu - 2) / params.nu)
    variance = np.full(num_simulations, next_variance)
    paths = np.empty((num_simulations, horizon))
    for h in range(horizon):
        eps = np.sqrt(variance) * z[:, h]
        paths[:, h] = (params.mu + eps) / PCT
        variance = params.omega + params.alpha * eps * eps + params.beta * variance
    return paths


def fhs_var_es(standardized, sigma_next: float, alpha: float):
    """
    Filtered Historical Simulation: empirical alpha-quantile of the standardized residuals,
    rescaled by tomorrow's volatility. Returns (VaR, ES) as positive decimal losses.
    """
    z = np.asarray(standardized, dtype=float)
    q = np.percentile(z, alpha * 100)
    tail = z[z <= q]
    return -q * sigma_next, -(tail.mean() if len(tail) else q) * sigma_next
