"""
_support/pnlf_models.py
=======================
The luminosity-function models and the machinery that fits them.

This module holds the parts of the analysis that are not specific to any one
sample: the ten candidate forms of the radio planetary nebula luminosity
function, the selection-aware likelihood that fits them, and the diagnostics
used to compare them.  Step 5 imports it twice, once for each sample, so that
the two fits are done by identical code and any difference between them is a
difference in the data rather than in the treatment.

The likelihood
--------------
Every source has its own detection limit, set by the local noise where it
happens to sit in the mosaic.  A source is only in the catalogue because it
was above that limit, so the quantity we can write down is not p(x) but
p(x | x >= limit), the density renormalised over the range where that source
could have been seen.  Fitting the unconditional density instead would treat
the missing faint objects as though they did not exist and would bias the
characteristic luminosity.  Working conditionally is what makes it legitimate
to fit right down to the faint end without first throwing away everything
near the limit.

Models
------
    Canonical Ciardullo     the standard PNLF, alpha and beta fixed at the
                            values Ciardullo et al. (1989) derived optically
    Generalised Ciardullo   the same shape with the faint slope and the
                            cutoff sharpness free
    Radio expanding shell   a two-branch form: a nebula brightens as it
                            expands and ionises, then fades as it thins
    Schechter               the standard galaxy form, used here as a
                            flexible reference
    Log-normal              no bright cutoff at all
    Student-t               log-normal with heavier tails
    Gaussian mixture (2)    two populations

Comparison uses AICc and BIC, and the fit quality is checked with a
probability integral transform: if the model is right, the conditional CDF
evaluated at the observed luminosities is uniform, so a KS test against a
uniform distribution measures how well the shape matches.

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize, brentq
from scipy.special import expit, gammaincc, gammainccinv, gammaln, logsumexp
from scipy.stats import kstest, norm, t as student_t

SEED = 20260811
RNG = np.random.default_rng(SEED)

# bootstrap sizes; the reverse Kaplan-Meier band and the calibrated PIT
# p-value both need resampling, and these are the counts used throughout
N_RKM_BOOT = 500
N_PARAM_BOOT = 160
N_CUTOFF_BOOT = 300


def safe_aicc(loglike: float, k: int, n: int) -> float:
    aic = 2.0 * k - 2.0 * loglike
    if n <= k + 1:
        return np.inf
    return aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)


def pstring(values: np.ndarray, names: list[str]) -> str:
    return "; ".join(f"{name}={value:.6g}" for name, value in zip(names, values))


def ciardullo_integral(T: np.ndarray, a: float, g: float) -> np.ndarray:
    """Integral of exp(a t) [1-exp(-g t)] from t=0 to T."""
    T = np.asarray(T, dtype=float)
    first = np.expm1(a * T) / a
    ag = a - g
    if abs(ag) < 1.0e-8:
        second = T
    else:
        second = np.expm1(ag * T) / ag
    return np.maximum(first - second, 1.0e-300)


def shell_integral(T: np.ndarray, weight: float) -> np.ndarray:
    """Integral of the two-branch radio shell density over a log-L interval."""
    T = np.asarray(T, dtype=float)
    k_rise = np.log(10.0) / 2.0
    k_fade = np.log(10.0) / 3.0
    rising = weight * (-np.expm1(-k_rise * T)) / k_rise
    fading = (1.0 - weight) * np.expm1(k_fade * T) / k_fade
    return np.maximum(rising + fading, 1.0e-300)


@dataclass
class FitResult:
    model: str
    params: np.ndarray
    names: list[str]
    loglike: float
    k: int
    aicc: float
    bic: float
    ks_d: float
    ks_p_naive: float
    ks_p_boot: float = np.nan
    success: bool = True


MODEL_ORDER = [
    "Canonical Ciardullo",
    "Generalised Ciardullo",
    "Radio expanding shell",
    "Schechter",
    "Log-normal",
    "Student-t",
    "Gaussian mixture (2)",
]


def model_bounds(model: str, x: np.ndarray, limits: np.ndarray):
    xmin = float(min(np.min(x), np.min(limits)))
    xmax = float(np.max(x))
    if model == "Canonical Ciardullo":
        return [(xmax + 1.0e-4, xmax + 2.0)], ["logL_max"]
    if model == "Generalised Ciardullo":
        return [
            (xmax + 1.0e-4, xmax + 2.0),
            (0.02, 4.0),
            (0.20, 30.0),
        ], ["logL_max", "faint_slope_a", "cutoff_g"]
    if model == "Radio expanding shell":
        return [(xmax + 1.0e-4, xmax + 2.0), (-6.0, 6.0)], ["logL_max", "branch_logit"]
    if model == "Schechter":
        return [(xmin - 1.5, xmax + 1.5), (-0.95, 4.0)], ["logL_star", "alpha"]
    if model == "Log-normal":
        return [(xmin - 1.0, xmax + 1.0), (np.log(0.04), np.log(3.0))], ["mu", "log_sigma"]
    if model == "Student-t":
        return [
            (xmin - 1.0, xmax + 1.0),
            (np.log(0.04), np.log(3.0)),
            (np.log(0.7), np.log(150.0)),
        ], ["loc", "log_scale", "log_degrees_freedom"]
    if model == "Gaussian mixture (2)":
        return [
            (-7.0, 7.0),
            (xmin - 1.0, xmax + 0.5),
            (np.log(0.04), np.log(4.0)),
            (np.log(0.04), np.log(2.5)),
            (np.log(0.04), np.log(2.5)),
        ], ["weight_logit", "mu_1", "log_delta_mu", "log_sigma_1", "log_sigma_2"]
    raise KeyError(model)


def conditional_logpdf(model: str, params: np.ndarray, x: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Log p(x_i | x_i >= local_limit_i, model)."""
    x = np.asarray(x, dtype=float)
    limits = np.asarray(limits, dtype=float)

    if model in {"Canonical Ciardullo", "Generalised Ciardullo"}:
        if model == "Canonical Ciardullo":
            xstar = params[0]
            a, g = 2.5 * 0.307, 2.5 * 3.0
        else:
            xstar, a, g = params
        t = xstar - x
        T = xstar - limits
        good = (t > 0) & (T > 0) & (x >= limits)
        out = np.full(len(x), -np.inf)
        if np.any(good):
            logshape = a * t[good] + np.log(-np.expm1(-g * t[good]))
            out[good] = logshape - np.log(ciardullo_integral(T[good], a, g))
        return out

    if model == "Radio expanding shell":
        xstar, raw_weight = params
        weight = float(expit(raw_weight))
        t = xstar - x
        T = xstar - limits
        good = (t >= 0) & (T > 0) & (x >= limits)
        out = np.full(len(x), -np.inf)
        if np.any(good):
            k_rise = np.log(10.0) / 2.0
            k_fade = np.log(10.0) / 3.0
            logshape = logsumexp(
                np.vstack(
                    [
                        np.log(weight) - k_rise * t[good],
                        np.log1p(-weight) + k_fade * t[good],
                    ]
                ),
                axis=0,
            )
            out[good] = logshape - np.log(shell_integral(T[good], weight))
        return out

    if model == "Schechter":
        xstar, alpha = params
        shape = alpha + 1.0
        y = 10.0 ** (x - xstar)
        ymin = 10.0 ** (limits - xstar)
        qmin = np.maximum(gammaincc(shape, ymin), 1.0e-300)
        result = np.log(np.log(10.0)) + shape * np.log(y) - y - gammaln(shape) - np.log(qmin)
        return np.where(x >= limits, result, -np.inf)

    if model == "Log-normal":
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        result = norm.logpdf(x, loc=mu, scale=sigma) - norm.logsf((limits - mu) / sigma)
        return np.where(x >= limits, result, -np.inf)

    if model == "Student-t":
        loc, log_scale, log_df = params
        scale = np.exp(log_scale)
        df = np.exp(log_df)
        result = student_t.logpdf(x, df, loc=loc, scale=scale) - student_t.logsf(
            (limits - loc) / scale, df
        )
        return np.where(x >= limits, result, -np.inf)

    if model == "Gaussian mixture (2)":
        raw_w, mu1, log_delta, log_s1, log_s2 = params
        w = float(expit(raw_w))
        mu2 = mu1 + np.exp(log_delta)
        s1, s2 = np.exp(log_s1), np.exp(log_s2)
        lp = logsumexp(
            np.vstack(
                [
                    np.log(w) + norm.logpdf(x, mu1, s1),
                    np.log1p(-w) + norm.logpdf(x, mu2, s2),
                ]
            ),
            axis=0,
        )
        lsf = logsumexp(
            np.vstack(
                [
                    np.log(w) + norm.logsf((limits - mu1) / s1),
                    np.log1p(-w) + norm.logsf((limits - mu2) / s2),
                ]
            ),
            axis=0,
        )
        return np.where(x >= limits, lp - lsf, -np.inf)

    raise KeyError(model)


def conditional_cdf(model: str, params: np.ndarray, x: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Per-source conditional CDF, used for PIT and predictive overlays."""
    x = np.asarray(x, dtype=float)
    limits = np.asarray(limits, dtype=float)

    if model in {"Canonical Ciardullo", "Generalised Ciardullo"}:
        if model == "Canonical Ciardullo":
            xstar = params[0]
            a, g = 2.5 * 0.307, 2.5 * 3.0
        else:
            xstar, a, g = params
        T = np.maximum(xstar - limits, 0.0)
        t = np.clip(xstar - x, 0.0, T)
        full = ciardullo_integral(T, a, g)
        return np.clip(1.0 - ciardullo_integral(t, a, g) / full, 0.0, 1.0)

    if model == "Radio expanding shell":
        xstar, raw_weight = params
        weight = float(expit(raw_weight))
        T = np.maximum(xstar - limits, 0.0)
        t = np.clip(xstar - x, 0.0, T)
        return np.clip(1.0 - shell_integral(t, weight) / shell_integral(T, weight), 0.0, 1.0)

    if model == "Schechter":
        xstar, alpha = params
        shape = alpha + 1.0
        y = 10.0 ** (x - xstar)
        ymin = 10.0 ** (limits - xstar)
        qmin = np.maximum(gammaincc(shape, ymin), 1.0e-300)
        return np.clip(1.0 - gammaincc(shape, y) / qmin, 0.0, 1.0)

    if model == "Log-normal":
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        f0 = norm.cdf((limits - mu) / sigma)
        return np.clip((norm.cdf((x - mu) / sigma) - f0) / np.maximum(1.0 - f0, 1.0e-300), 0.0, 1.0)

    if model == "Student-t":
        loc, log_scale, log_df = params
        scale = np.exp(log_scale)
        df = np.exp(log_df)
        sf0 = np.maximum(student_t.sf((limits - loc) / scale, df), 1.0e-300)
        return np.clip(1.0 - student_t.sf((x - loc) / scale, df) / sf0, 0.0, 1.0)

    if model == "Gaussian mixture (2)":
        raw_w, mu1, log_delta, log_s1, log_s2 = params
        w = float(expit(raw_w))
        mu2 = mu1 + np.exp(log_delta)
        s1, s2 = np.exp(log_s1), np.exp(log_s2)
        cdf = w * norm.cdf((x - mu1) / s1) + (1.0 - w) * norm.cdf((x - mu2) / s2)
        f0 = w * norm.cdf((limits - mu1) / s1) + (1.0 - w) * norm.cdf((limits - mu2) / s2)
        return np.clip((cdf - f0) / np.maximum(1.0 - f0, 1.0e-300), 0.0, 1.0)

    raise KeyError(model)


def fit_model(model: str, x: np.ndarray, limits: np.ndarray, fast: bool = False, start=None) -> FitResult:
    bounds, names = model_bounds(model, x, limits)

    def objective(params):
        lp = conditional_logpdf(model, np.asarray(params), x, limits)
        if not np.all(np.isfinite(lp)):
            return 1.0e100
        return float(-np.sum(lp))

    result = None
    if start is not None:
        try:
            local = minimize(objective, np.asarray(start), method="L-BFGS-B", bounds=bounds)
            if local.success and np.isfinite(local.fun):
                result = local
        except Exception:
            result = None

    if result is None:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=SEED,
            maxiter=120 if fast else 650,
            popsize=8 if fast else 15,
            tol=2.0e-7 if fast else 2.0e-10,
            polish=True,
            updating="immediate",
        )

    params = np.asarray(result.x, dtype=float)
    loglike = -objective(params)
    pit = conditional_cdf(model, params, x, limits)
    pit = np.clip(pit, 1.0e-10, 1.0 - 1.0e-10)
    ks = kstest(pit, "uniform")
    k = len(params)
    return FitResult(
        model=model,
        params=params,
        names=names,
        loglike=loglike,
        k=k,
        aicc=safe_aicc(loglike, k, len(x)),
        bic=k * np.log(len(x)) - 2.0 * loglike,
        ks_d=float(ks.statistic),
        ks_p_naive=float(ks.pvalue),
        success=bool(result.success),
    )


def average_conditional_density(model: str, params: np.ndarray, grid: np.ndarray, limits: np.ndarray) -> np.ndarray:
    density = np.zeros_like(grid)
    for limit in limits:
        lp = conditional_logpdf(model, params, grid, np.full_like(grid, limit))
        vals = np.where(np.isfinite(lp), np.exp(np.clip(lp, -745, 700)), 0.0)
        density += vals
    return density / len(limits)


def average_conditional_cdf(model: str, params: np.ndarray, grid: np.ndarray, limits: np.ndarray) -> np.ndarray:
    result = np.zeros_like(grid)
    for limit in limits:
        result += conditional_cdf(model, params, grid, np.full_like(grid, limit))
    return result / len(limits)


def sample_conditional(model: str, params: np.ndarray, limits: np.ndarray, rng) -> np.ndarray:
    """Draw one luminosity above each supplied local limit."""
    draws = np.full(len(limits), np.nan)
    uniforms = rng.uniform(1.0e-8, 1.0 - 1.0e-8, len(limits))

    if model == "Log-normal":
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        f0 = norm.cdf((limits - mu) / sigma)
        return norm.ppf(f0 + uniforms * (1.0 - f0), mu, sigma)

    if model == "Student-t":
        loc, log_scale, log_df = params
        scale = np.exp(log_scale)
        df = np.exp(log_df)
        f0 = student_t.cdf((limits - loc) / scale, df)
        return loc + scale * student_t.ppf(f0 + uniforms * (1.0 - f0), df)

    if model == "Schechter":
        xstar, alpha = params
        shape = alpha + 1.0
        ymin = 10.0 ** (limits - xstar)
        qmin = np.maximum(gammaincc(shape, ymin), 1.0e-300)
        y = gammainccinv(shape, (1.0 - uniforms) * qmin)
        return xstar + np.log10(y)

    for i, (limit, uniform) in enumerate(zip(limits, uniforms)):
        if model in {"Canonical Ciardullo", "Generalised Ciardullo", "Radio expanding shell"}:
            xstar = float(params[0])
            high = xstar - 1.0e-9
        else:
            high = max(float(limit + 8.0), 25.0)

        def root(value):
            return float(conditional_cdf(model, params, np.array([value]), np.array([limit]))[0] - uniform)

        draws[i] = brentq(root, float(limit), high, xtol=1.0e-10, rtol=1.0e-10, maxiter=200)
    return draws


def calibrated_pit_pvalue(model: str, fit: FitResult, x: np.ndarray, limits: np.ndarray, n_boot: int) -> float:
    observed = fit.ks_d
    boot_stats = []
    bounds, _ = model_bounds(model, x, limits)
    for _ in range(n_boot):
        simulated = sample_conditional(model, fit.params, limits, RNG)
        try:
            refit = fit_model(model, simulated, limits, fast=True, start=fit.params)
            boot_stats.append(refit.ks_d)
        except Exception:
            continue
    if not boot_stats:
        return np.nan
    boot_stats = np.asarray(boot_stats)
    return float((1.0 + np.sum(boot_stats >= observed)) / (len(boot_stats) + 1.0))


def reverse_km_cdf(values: np.ndarray, limits: np.ndarray, detected: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Reverse Kaplan-Meier CDF for left-censored luminosities."""
    observed = np.where(detected, values, limits)
    y = -observed
    event = np.asarray(detected, dtype=bool)
    event_times = np.sort(np.unique(y[event]))
    survival = []
    current = 1.0
    for time in event_times:
        at_risk = int(np.sum(y >= time))
        deaths = int(np.sum(event & np.isclose(y, time, rtol=0.0, atol=1.0e-12)))
        if at_risk > 0:
            current *= 1.0 - deaths / at_risk
        survival.append(current)
    survival = np.asarray(survival)

    tgrid = -np.asarray(grid)
    idx = np.searchsorted(event_times, tgrid, side="right") - 1
    out = np.ones_like(grid, dtype=float)
    valid = idx >= 0
    out[valid] = survival[idx[valid]]
    return np.clip(out, 0.0, 1.0)


def reverse_km_band(values, limits, detected, grid, n_boot=N_RKM_BOOT):
    curves = np.full((n_boot, len(grid)), np.nan)
    n = len(values)
    for b in range(n_boot):
        take = RNG.integers(0, n, n)
        curves[b] = reverse_km_cdf(values[take], limits[take], detected[take], grid)
    return (
        np.nanpercentile(curves, 2.5, axis=0),
        np.nanpercentile(curves, 50.0, axis=0),
        np.nanpercentile(curves, 97.5, axis=0),
    )
