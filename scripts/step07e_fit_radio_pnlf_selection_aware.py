#!/usr/bin/env python3
"""
step07e_fit_radio_pnlf_selection_aware.py
================================================
Independent MeerKAT-only radio PNLF analysis.

This script deliberately does not alter or reproduce Steps 07a-07c.  It asks a
different statistical question: which luminosity-function shape is supported by
the unbinned 1.295-GHz MeerKAT measurements once the position-dependent survey
sensitivity is included?

Primary analysis
----------------
* Parent population is fixed without using radio brightness: Reid ``Known`` or
  ``True`` objects inside the MeerKAT RMS-map footprint.
* Primary detections are independent Aegean catalogue detections.  The 32
  position-guided visual recoveries are shown and tested separately, but do not
  define the primary luminosity-function shape.
* Integrated MeerKAT flux density is converted to spectral luminosity via
  L_nu = 4 pi d^2 S_nu at d=49.59 kpc.  (Updated 2026-08-30 from the
  previously adopted 49.97 kpc; Pietrzynski et al. 2019 gives 49.59 kpc
  for the quoted distance modulus mu = 18.477.)
* No bright interval, faint interval, individual luminosity, or non-empty bin is
  manually masked.  Each source instead carries its own local limiting
  luminosity from the MeerKAT RMS map.
* Fits use an unbinned, source-by-source truncated likelihood.
* Radio non-detections enter a reverse Kaplan-Meier estimate as upper limits.

Candidate shapes
----------------
1. Canonical Ciardullo: the optical functional form, held fixed as a null test.
2. Generalised Ciardullo: cutoff and both slopes free.
3. Radio expanding shell: optically thick rise (L~t^2) plus optically thin
   fading (L~t^-3), with a fitted branch weight and maximum luminosity.
4. Schechter: a soft bright-end exponential cutoff.
5. Log-normal: a one-component empirical density in log luminosity.
6. Student-t: a robust heavy-tailed empirical density.
7. Two-component Gaussian mixture: flexible unsupervised density estimation.

The flexible models are descriptions, not claims of distinct PN populations.
AICc and BIC penalise unnecessary flexibility.  A probability-integral-
transform diagnostic tests the complete fitted distribution rather than a
chosen set of histogram bins.

Outputs
-------
03_Outputs/step07e_radio_pnlf_selection_catalogue.vot
03_Outputs/step07e_radio_pnlf_model_comparison.vot
03_Outputs/step07e_radio_pnlf_summary.vot
04_Figures/step07e_radio_pnlf_selection_aware.pdf

References implemented in the method
------------------------------------
Ciardullo et al. 1989, ApJ, 339, 53 (canonical optical null form)
Feigelson & Nelson 1985, ApJ, 293, 192 (upper limits / survival analysis)
Bovy, Hogg & Roweis 2011, Ann. Appl. Stat., 5, 1657 (error-aware mixtures;
the present 1-D mixture is deliberately simpler)

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import differential_evolution, minimize, minimize_scalar, brentq
from scipy.special import expit, gammaincc, gammainccinv, gammaln, logsumexp
from scipy.stats import kstest, norm, t as student_t

from astropy.io import fits
from astropy.table import Column, Table
from astropy.wcs import WCS

from _support.plot_style import apply_paper_style


warnings.filterwarnings("ignore", category=RuntimeWarning)
apply_paper_style()

SEED = 20260811
RNG = np.random.default_rng(SEED)

BASE = Path(os.path.expanduser("~/Desktop/Research/PN LMC Paper"))
DATA = BASE / "01_Data"
OUTS = BASE / "03_Outputs"
FIGS = BASE / "04_Figures"

MASTER_FILE = OUTS / "step04b_multisurvey_master.vot"
RMS_FILE = DATA / "LMC_I_mosaic_ch0_rms.fits"

OUT_SELECTION = OUTS / "step07e_radio_pnlf_selection_catalogue.vot"
OUT_MODELS = OUTS / "step07e_radio_pnlf_model_comparison.vot"
OUT_SUMMARY = OUTS / "step07e_radio_pnlf_summary.vot"
OUT_PDF = FIGS / "step07e_radio_pnlf_selection_aware.pdf"

# DISTANCE_KPC = 49.97
# updated 2026-08-30: Pietrzynski et al. 2019 gives 49.59 kpc (mu = 18.477); 49.97 kpc was inconsistent with the quoted modulus
DISTANCE_KPC = 49.59
DISTANCE_M = DISTANCE_KPC * 3.085677581491367e19
JY_TO_W_HZ = 4.0 * np.pi * DISTANCE_M**2 * 1.0e-26

PRIMARY_CLASSES = {"Known", "True"}
CATALOGUE_SIGMA = 5.0
VISUAL_SIGMA = 3.0
N_RKM_BOOT = 500
N_PARAM_BOOT = 160
N_CUTOFF_BOOT = 300

COLORS = {
    "ink": "#17212B",
    "muted": "#65717E",
    "grid": "#D7DEE5",
    "primary": "#007C91",
    "primary_dark": "#005A6B",
    "visual": "#E28E2C",
    "canonical": "#8E4B9D",
    "best": "#D1495B",
    "band": "#9ED9E3",
    "grey": "#AEB7C2",
}


def finite_float(column) -> np.ndarray:
    """Masked-column-safe float conversion."""
    values = np.asarray(np.ma.getdata(column), dtype=float).copy()
    mask = np.ma.getmaskarray(column)
    if np.any(mask):
        values[mask] = np.nan
    return values


def text_array(column) -> np.ndarray:
    if hasattr(column, "filled"):
        column = column.filled("")
    return np.asarray([str(v).strip() for v in column], dtype="U80")


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


def fit_sample(sample_name, detected_mask, parent_mask, log_lum, log_limit, model_rows):
    det = detected_mask & parent_mask & np.isfinite(log_lum) & np.isfinite(log_limit)
    x = log_lum[det]
    u = log_limit[det]
    if len(x) < 20:
        raise ValueError(f"{sample_name}: only {len(x)} usable detections")

    print(f"\n  {sample_name}: {len(x)} detections / {int(parent_mask.sum())} parent objects")
    fits = []
    for model in MODEL_ORDER:
        fit = fit_model(model, x, u)
        fits.append(fit)
        print(
            f"    {model:<26s} logL={fit.loglike:8.2f}  "
            f"AICc={fit.aicc:8.2f}  BIC={fit.bic:8.2f}  PIT D={fit.ks_d:.3f}"
        )

    best_aicc = min(f.aicc for f in fits)
    best_bic = min(f.bic for f in fits)
    for fit in fits:
        model_rows.append(
            dict(
                sample=sample_name,
                model=fit.model,
                n_parent=int(parent_mask.sum()),
                n_detected=len(x),
                n_parameters=fit.k,
                log_likelihood=fit.loglike,
                AICc=fit.aicc,
                delta_AICc=fit.aicc - best_aicc,
                BIC=fit.bic,
                delta_BIC=fit.bic - best_bic,
                pit_KS_D=fit.ks_d,
                pit_KS_p_naive=fit.ks_p_naive,
                pit_KS_p_bootstrap=np.nan,
                parameter_names=", ".join(fit.names),
                parameters=pstring(fit.params, fit.names),
                fit_success=int(fit.success),
            )
        )
    return x, u, fits


def add_panel_label(ax, label):
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def tidy_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.7, alpha=0.55)


print("step07e: MeerKAT-only unbinned selection-aware radio PNLF")

for required in [MASTER_FILE, RMS_FILE]:
    if not required.exists():
        raise FileNotFoundError(required)
OUTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

print("\n[1] Loading MeerKAT master catalogue and local RMS map ...")
master = Table.read(MASTER_FILE, format="votable")
n_total = len(master)
ra = finite_float(master["RA"])
dec = finite_float(master["Dec"])
reid_class = text_array(master["reid_class"])
method = text_array(master["mkt_detection_method"])
det_meerkat = finite_float(master["det_meerkat"]).astype(int) == 1
flux_int = finite_float(master["mkt_int_flux_Jy"])
flux_peak = finite_float(master["mkt_peak_flux_Jy"])
flux_err = finite_float(master["mkt_err_int_flux_Jy"])
catalogue_detected = det_meerkat & (method == "catalogue") & np.isfinite(flux_int) & (flux_int > 0)
visual_detected = det_meerkat & (method == "visual") & np.isfinite(flux_int) & (flux_int > 0)

with fits.open(RMS_FILE, memmap=True) as hdul:
    hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
    rms_data = np.squeeze(hdu.data)
    rms_wcs = WCS(hdu.header, naxis=2)
    px, py = rms_wcs.all_world2pix(ra, dec, 0)
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    in_bounds = (ix >= 0) & (iy >= 0) & (ix < rms_data.shape[-1]) & (iy < rms_data.shape[-2])
    local_rms = np.full(n_total, np.nan)
    indices = np.where(in_bounds)[0]
    local_rms[indices] = rms_data[iy[indices], ix[indices]]

inside = in_bounds & np.isfinite(local_rms) & (local_rms > 0)
print(f"    Reid rows: {n_total}")
print(f"    Inside positive MeerKAT RMS footprint: {inside.sum()}")

# Aegean detects on peak S/N.  Translate peak threshold to an integrated-flux
# threshold with the robust median integrated/peak ratio of catalogue sources.
ratio_values = flux_int[catalogue_detected] / flux_peak[catalogue_detected]
ratio_values = ratio_values[np.isfinite(ratio_values) & (ratio_values > 0)]
int_peak_ratio = float(np.nanmedian(ratio_values))
print(f"    Median catalogue S_int/S_peak: {int_peak_ratio:.4f}")

flux_limit_primary = CATALOGUE_SIGMA * local_rms * int_peak_ratio
flux_limit_inclusive = VISUAL_SIGMA * local_rms * int_peak_ratio

# For a detected source the actual S_int/S_peak ratio gives the closest
# representation of the peak-selected Aegean boundary in integrated flux.
det_ratio = np.full(n_total, int_peak_ratio)
good_ratio = catalogue_detected & np.isfinite(flux_peak) & (flux_peak > 0)
det_ratio[good_ratio] = flux_int[good_ratio] / flux_peak[good_ratio]
flux_limit_for_fit = CATALOGUE_SIGMA * local_rms * det_ratio

good_visual_ratio = visual_detected & np.isfinite(flux_peak) & (flux_peak > 0)
visual_ratio = np.full(n_total, int_peak_ratio)
visual_ratio[good_visual_ratio] = flux_int[good_visual_ratio] / flux_peak[good_visual_ratio]
flux_limit_visual_fit = VISUAL_SIGMA * local_rms * visual_ratio

lum_w_hz = flux_int * JY_TO_W_HZ
lum_err_w_hz = flux_err * JY_TO_W_HZ
limit_w_hz = flux_limit_primary * JY_TO_W_HZ
limit_inclusive_w_hz = flux_limit_inclusive * JY_TO_W_HZ
limit_fit_w_hz = flux_limit_for_fit * JY_TO_W_HZ
limit_visual_fit_w_hz = flux_limit_visual_fit * JY_TO_W_HZ

log_lum = np.where(lum_w_hz > 0, np.log10(lum_w_hz), np.nan)
log_lum_err = np.where(
    (lum_w_hz > 0) & np.isfinite(lum_err_w_hz) & (lum_err_w_hz > 0),
    lum_err_w_hz / lum_w_hz / np.log(10.0),
    np.nan,
)
log_limit = np.where(limit_w_hz > 0, np.log10(limit_w_hz), np.nan)
log_limit_fit = np.where(limit_fit_w_hz > 0, np.log10(limit_fit_w_hz), np.nan)
log_limit_inclusive = np.where(limit_inclusive_w_hz > 0, np.log10(limit_inclusive_w_hz), np.nan)
log_limit_visual_fit = np.where(limit_visual_fit_w_hz > 0, np.log10(limit_visual_fit_w_hz), np.nan)

primary_parent = inside & np.isin(reid_class, list(PRIMARY_CLASSES))
known_parent = inside & (reid_class == "Known")
all_parent = inside.copy()
primary_detected = primary_parent & catalogue_detected
all_catalogue_detected = inside & catalogue_detected
inclusive_detected = primary_parent & (catalogue_detected | visual_detected)

fit_limits_primary = log_limit_fit.copy()
fit_limits_all = log_limit_fit.copy()
fit_limits_inclusive = np.where(visual_detected, log_limit_visual_fit, log_limit_fit)
# Diagnostic branch only: visual recoveries have no reproducible flux threshold.
fit_limits_inclusive[inclusive_detected] = np.minimum(
    fit_limits_inclusive[inclusive_detected],
    log_lum[inclusive_detected] - 1.0e-6,
)

print("\n[2] Pre-registered samples ...")
print(f"    Primary parent (Reid Known/True): {primary_parent.sum()}")
print(f"    Primary catalogue detections: {primary_detected.sum()}")
print(f"    Primary visual recoveries shown only diagnostically: {(primary_parent & visual_detected).sum()}")
print(f"    All Reid catalogue detections (robustness): {all_catalogue_detected.sum()}")

role = np.full(n_total, "outside_footprint", dtype="U40")
role[inside] = "outside_primary_class"
role[primary_parent] = "primary_upper_limit"
role[primary_parent & visual_detected] = "visual_recovery_diagnostic"
role[primary_detected] = "primary_catalogue_detection"

selection = Table()
selection["RP_ID"] = text_array(master["RP_ID"])
selection["reid_class"] = reid_class
selection["in_meerkat_footprint"] = inside.astype(np.int16)
selection["primary_parent"] = primary_parent.astype(np.int16)
selection["primary_catalogue_detection"] = primary_detected.astype(np.int16)
selection["visual_recovery_diagnostic"] = (primary_parent & visual_detected).astype(np.int16)
selection["pnlf_role"] = role
selection["mkt_detection_method"] = method
selection.add_column(Column(local_rms.astype("f8"), name="mkt_local_rms_Jy_beam", unit="Jy/beam"))
selection.add_column(Column(flux_int.astype("f8"), name="mkt_int_flux_Jy", unit="Jy"))
selection.add_column(Column(flux_err.astype("f8"), name="mkt_err_int_flux_Jy", unit="Jy"))
selection.add_column(Column(flux_limit_primary.astype("f8"), name="mkt_5sigma_int_limit_Jy", unit="Jy"))
selection.add_column(Column(lum_w_hz.astype("f8"), name="radio_luminosity_W_Hz", unit="W/Hz"))
selection.add_column(Column(log_lum.astype("f8"), name="log_radio_luminosity_W_Hz"))
selection.add_column(Column(log_lum_err.astype("f8"), name="log_radio_luminosity_err_dex"))
selection.add_column(Column(log_limit.astype("f8"), name="log_radio_luminosity_5sigma_upper_W_Hz"))
selection["luminosity_masked"] = np.zeros(n_total, dtype=np.int16)
selection.write(OUT_SELECTION, format="votable", overwrite=True)
print(f"    -> {OUT_SELECTION.name}")

print("\n[3] Fitting seven unbinned, selection-aware candidate shapes ...")
model_rows = []
primary_x, primary_u, primary_fits = fit_sample(
    "primary_known_true_catalogue",
    catalogue_detected,
    primary_parent,
    log_lum,
    fit_limits_primary,
    model_rows,
)
_, _, known_fits = fit_sample(
    "known_only_catalogue",
    catalogue_detected,
    known_parent,
    log_lum,
    fit_limits_primary,
    model_rows,
)
_, _, all_fits = fit_sample(
    "all_reid_catalogue",
    catalogue_detected,
    all_parent,
    log_lum,
    fit_limits_all,
    model_rows,
)
inclusive_x, inclusive_u, inclusive_fits = fit_sample(
    "known_true_plus_visual_diagnostic",
    catalogue_detected | visual_detected,
    primary_parent,
    log_lum,
    fit_limits_inclusive,
    model_rows,
)

primary_sorted = sorted(primary_fits, key=lambda fit: fit.aicc)
best = primary_sorted[0]
canonical = next(fit for fit in primary_fits if fit.model == "Canonical Ciardullo")
best_noncanonical = next(fit for fit in primary_sorted if fit.model != "Canonical Ciardullo")
print(f"\n    Primary AICc winner: {best.model}")
print(f"    Delta AICc (canonical - winner): {canonical.aicc - best.aicc:.2f}")
print(f"    Best non-canonical comparison: {best_noncanonical.model} (Delta AICc={best_noncanonical.aicc-best.aicc:.2f})")

print("\n[4] Calibrating distribution-level goodness of fit ...")
models_to_calibrate = {"Canonical Ciardullo", best_noncanonical.model}
for fit in primary_fits:
    if fit.model in models_to_calibrate:
        fit.ks_p_boot = calibrated_pit_pvalue(fit.model, fit, primary_x, primary_u, N_PARAM_BOOT)
        print(f"    {fit.model:<26s} bootstrap PIT p={fit.ks_p_boot:.4f}")

for row in model_rows:
    if row["sample"] == "primary_known_true_catalogue":
        match = next(f for f in primary_fits if f.model == row["model"])
        row["pit_KS_p_bootstrap"] = match.ks_p_boot

print("\n[5] Reverse Kaplan-Meier upper-limit inference ...")
parent_values = log_lum[primary_parent]
parent_limits = log_limit[primary_parent]
parent_events = catalogue_detected[primary_parent] & np.isfinite(parent_values)
finite_parent = np.isfinite(parent_limits)
parent_values = parent_values[finite_parent]
parent_limits = parent_limits[finite_parent]
parent_events = parent_events[finite_parent]

plot_min = float(min(np.nanpercentile(parent_limits, 1) - 0.15, np.nanmin(primary_x) - 0.2))
plot_max = float(np.nanmax(np.concatenate([primary_x, inclusive_x])) + 0.28)
grid = np.linspace(plot_min, plot_max, 700)
rkm = reverse_km_cdf(parent_values, parent_limits, parent_events, grid)
rkm_lo, rkm_med, rkm_hi = reverse_km_band(parent_values, parent_limits, parent_events, grid)

print("\n[6] Bootstrap bright-cutoff stability ...")
cutoff_models = [fit for fit in primary_fits if "Ciardullo" in fit.model or fit.model == "Radio expanding shell"]
cutoff_draws = {fit.model: [] for fit in cutoff_models}
for _ in range(N_CUTOFF_BOOT):
    take = RNG.integers(0, len(primary_x), len(primary_x))
    x_b = primary_x[take].copy()
    u_b = primary_u[take]
    # Include quoted integrated-flux uncertainty without changing the selection
    # model.  The perturbation is in log luminosity and cannot cross its own
    # selection limit.
    err_b = log_lum_err[primary_detected][take]
    err_b = np.where(np.isfinite(err_b), np.minimum(err_b, 0.30), 0.0)
    x_b += RNG.normal(0.0, err_b)
    x_b = np.maximum(x_b, u_b + 1.0e-6)
    for fit in cutoff_models:
        try:
            refit = fit_model(fit.model, x_b, u_b, fast=True, start=fit.params)
            cutoff_draws[fit.model].append(refit.params[0])
        except Exception:
            pass

cutoff_summary = {}
for model, draws in cutoff_draws.items():
    q = np.nanpercentile(np.asarray(draws), [16, 50, 84]) if draws else [np.nan] * 3
    cutoff_summary[model] = q
    print(f"    {model:<26s} logLmax={q[1]:.3f} (+{q[2]-q[1]:.3f}/-{q[1]-q[0]:.3f})")

model_table = Table(rows=model_rows)
model_table.write(OUT_MODELS, format="votable", overwrite=True)
print(f"    -> {OUT_MODELS.name}")

best_cutoff = cutoff_summary.get(best.model, np.array([np.nan, np.nan, np.nan]))
canonical_cutoff = cutoff_summary.get("Canonical Ciardullo", np.array([np.nan, np.nan, np.nan]))
gmm_fit = next(fit for fit in primary_fits if fit.model == "Gaussian mixture (2)")
gmm_w_faint = float(expit(gmm_fit.params[0]))
gmm_mu_faint = float(gmm_fit.params[1])
gmm_mu_bright = float(gmm_fit.params[1] + np.exp(gmm_fit.params[2]))
gmm_sigma_faint = float(np.exp(gmm_fit.params[3]))
gmm_sigma_bright = float(np.exp(gmm_fit.params[4]))
general_fit = next(fit for fit in primary_fits if fit.model == "Generalised Ciardullo")
summary = Table()
summary["analysis_frequency_MHz"] = [1295.0]
summary["distance_kpc"] = [DISTANCE_KPC]
summary["n_reid_total"] = [n_total]
summary["n_inside_meerkat"] = [int(inside.sum())]
summary["n_primary_parent"] = [int(primary_parent.sum())]
summary["n_primary_catalogue_detected"] = [int(primary_detected.sum())]
summary["n_primary_visual_diagnostic"] = [int((primary_parent & visual_detected).sum())]
summary["median_local_rms_uJy_beam"] = [float(np.nanmedian(local_rms[primary_parent]) * 1.0e6)]
summary["median_5sigma_logL_W_Hz"] = [float(np.nanmedian(log_limit[primary_parent]))]
summary["median_int_peak_ratio"] = [int_peak_ratio]
summary["primary_best_model_AICc"] = [best.model]
summary["primary_best_parameters"] = [pstring(best.params, best.names)]
summary["primary_best_noncanonical_model"] = [best_noncanonical.model]
summary["primary_best_noncanonical_delta_AICc"] = [best_noncanonical.aicc - best.aicc]
summary["canonical_delta_AICc"] = [canonical.aicc - best.aicc]
summary["canonical_delta_BIC"] = [canonical.bic - min(f.bic for f in primary_fits)]
summary["canonical_PIT_p_bootstrap"] = [canonical.ks_p_boot]
summary["best_noncanonical_PIT_p_bootstrap"] = [best_noncanonical.ks_p_boot]
summary["gmm_faint_component_fraction"] = [gmm_w_faint]
summary["gmm_bright_component_fraction"] = [1.0 - gmm_w_faint]
summary["gmm_faint_component_mu_logL"] = [gmm_mu_faint]
summary["gmm_bright_component_mu_logL"] = [gmm_mu_bright]
summary["gmm_faint_component_sigma_dex"] = [gmm_sigma_faint]
summary["gmm_bright_component_sigma_dex"] = [gmm_sigma_bright]
summary["generalised_ciardullo_faint_slope_a"] = [float(general_fit.params[1])]
summary["generalised_ciardullo_cutoff_g"] = [float(general_fit.params[2])]
summary["best_cutoff_logL_p16"] = [best_cutoff[0]]
summary["best_cutoff_logL_p50"] = [best_cutoff[1]]
summary["best_cutoff_logL_p84"] = [best_cutoff[2]]
summary["canonical_cutoff_logL_p16"] = [canonical_cutoff[0]]
summary["canonical_cutoff_logL_p50"] = [canonical_cutoff[1]]
summary["canonical_cutoff_logL_p84"] = [canonical_cutoff[2]]
summary["manual_luminosity_masks"] = [0]
summary["primary_selection_note"] = [
    "Reid Known/True inside RMS footprint; automatic MeerKAT catalogue detections; visual recoveries diagnostic only"
]
summary.write(OUT_SUMMARY, format="votable", overwrite=True)
print(f"    -> {OUT_SUMMARY.name}")

print("\n[7] Building the diagnostic PDF ...")
fit_by_name = {fit.model: fit for fit in primary_fits}
display_models = [best.model, best_noncanonical.model]
display_models = list(dict.fromkeys(display_models))
if "Radio expanding shell" not in display_models:
    display_models.append("Radio expanding shell")

with PdfPages(OUT_PDF) as pdf:
    # Page 1
    fig = plt.figure(figsize=(12.8, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.975, bottom=0.115, top=0.875, hspace=0.37, wspace=0.25)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_cdf = fig.add_subplot(gs[0, 1])
    ax_aic = fig.add_subplot(gs[1, 0])
    ax_pit = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "MeerKAT 1.295-GHz radio planetary-nebula luminosity function",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.075,
        0.918,
        f"Unbinned local-sensitivity likelihood  |  primary: {len(primary_x)} catalogue detections from "
        f"{primary_parent.sum()} Reid Known/True objects  |  no luminosity masks",
        ha="left",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    # A: observed values and selection-aware model count curves
    iqr = np.subtract(*np.percentile(primary_x, [75, 25]))
    bin_width = max(0.13, min(0.28, 2.0 * iqr / np.cbrt(len(primary_x))))
    edges = np.arange(np.floor(plot_min / bin_width) * bin_width, plot_max + bin_width, bin_width)
    counts, _ = np.histogram(primary_x, bins=edges)
    ax_hist.hist(
        primary_x,
        bins=edges,
        color=COLORS["primary"],
        alpha=0.78,
        edgecolor="white",
        lw=0.8,
        label="Primary catalogue detections",
        zorder=2,
    )
    visual_primary_x = log_lum[primary_parent & visual_detected]
    ax_hist.hist(
        visual_primary_x[np.isfinite(visual_primary_x)],
        bins=edges,
        histtype="step",
        color=COLORS["visual"],
        lw=1.8,
        hatch="////",
        label="Visual recoveries (diagnostic)",
        zorder=3,
    )
    line_styles = [
        (COLORS["best"], 2.6, "-"),
        (COLORS["canonical"], 2.1, "--"),
        (COLORS["ink"], 1.8, ":"),
    ]
    for model, (color, lw, ls) in zip(display_models, line_styles):
        fit = fit_by_name[model]
        density = average_conditional_density(model, fit.params, grid, primary_u)
        label = f"{model}" + (" (AICc best)" if model == best.model else "")
        ax_hist.plot(grid, density * len(primary_x) * bin_width, color=color, lw=lw, ls=ls, label=label, zorder=5)
    ax_hist.set_xlabel(r"$\log_{10}(L_{1.295\,\mathrm{GHz}} / \mathrm{W\,Hz^{-1}})$")
    ax_hist.set_ylabel(f"Objects per {bin_width:.2f} dex")
    ax_hist.set_xlim(plot_min, plot_max)
    ax_hist.set_ylim(0, max(counts.max() * 1.28, 5))
    ax_hist.legend(frameon=False, fontsize=8.2, loc="upper left")
    tidy_axis(ax_hist)
    add_panel_label(ax_hist, "A")

    # B: detected-sample ECDF and selection-aware predictions.  The reverse-KM
    # parent-population estimate is intentionally kept on page 2 because it is
    # a different estimand from the conditional distribution of detections.
    ecdf_x = np.sort(primary_x)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
    ax_cdf.step(
        ecdf_x,
        ecdf_y,
        where="post",
        color=COLORS["primary_dark"],
        lw=2.3,
        label="Observed catalogue-detection ECDF",
    )
    for model, (color, lw, ls) in zip(display_models, line_styles):
        fit = fit_by_name[model]
        pred = average_conditional_cdf(model, fit.params, grid, primary_u)
        ax_cdf.plot(grid, pred, color=color, lw=lw, ls=ls, label=model)
    ax_cdf.set_xlabel(r"$\log_{10}(L_{1.295\,\mathrm{GHz}} / \mathrm{W\,Hz^{-1}})$")
    ax_cdf.set_ylabel("Conditional cumulative fraction of detections")
    ax_cdf.set_xlim(plot_min, plot_max)
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.legend(frameon=False, fontsize=8.1, loc="lower right")
    tidy_axis(ax_cdf)
    add_panel_label(ax_cdf, "B")

    # C: penalised model comparison
    ranked = sorted(primary_fits, key=lambda fit: fit.aicc, reverse=True)
    delta = np.asarray([fit.aicc - best.aicc for fit in ranked])
    labels = [fit.model for fit in ranked]
    bar_colors = [COLORS["best"] if name == best.model else COLORS["grey"] for name in labels]
    bars = ax_aic.barh(np.arange(len(ranked)), delta, color=bar_colors, edgecolor="white", height=0.72)
    ax_aic.set_yticks(np.arange(len(ranked)), labels)
    ax_aic.set_xlabel(r"$\Delta$AICc from the preferred model")
    ax_aic.axvline(2, color=COLORS["muted"], lw=1.0, ls="--")
    ax_aic.axvline(10, color=COLORS["muted"], lw=1.0, ls=":")
    max_delta = max(float(np.nanmax(delta)), 1.0)
    ax_aic.set_xlim(0, max_delta * 1.18 + 0.5)
    for bar, value in zip(bars, delta):
        ax_aic.text(value + max_delta * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8.5)
    tidy_axis(ax_aic)
    add_panel_label(ax_aic, "C")

    # D: probability-integral transform; bin-free distribution check
    uniform_grid = np.linspace(0, 1, len(primary_x) + 1)
    ax_pit.plot([0, 1], [0, 1], color=COLORS["grey"], lw=1.4, ls="--", label="Expected if calibrated")
    pit_models = [
        ("Canonical Ciardullo", COLORS["canonical"], 2.4, "--"),
        (best_noncanonical.model, COLORS["best"], 2.4, "-"),
    ]
    for model, color, lw, ls in pit_models:
        fit = fit_by_name[model]
        pit = np.sort(conditional_cdf(model, fit.params, primary_x, primary_u))
        y = np.arange(1, len(pit) + 1) / len(pit)
        ptxt = f"{fit.ks_p_boot:.3f}" if np.isfinite(fit.ks_p_boot) else "not calibrated"
        ax_pit.step(pit, y, where="post", color=color, lw=lw, ls=ls, label=f"{model}  (bootstrap p={ptxt})")
    ax_pit.set_xlabel("Probability-integral transform")
    ax_pit.set_ylabel("Empirical cumulative fraction")
    ax_pit.set_xlim(0, 1)
    ax_pit.set_ylim(0, 1)
    ax_pit.legend(frameon=False, fontsize=8.3, loc="lower right")
    tidy_axis(ax_pit)
    add_panel_label(ax_pit, "D")

    footer = (
        "A histogram is shown only for visualisation; all fitting and ranking are unbinned.  "
        "Local truncation limits are derived from the MeerKAT RMS map and the catalogue's peak-S/N selection."
    )
    fig.text(0.075, 0.018, footer, ha="left", va="bottom", fontsize=8.7, color=COLORS["muted"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Page 2
    fig = plt.figure(figsize=(12.8, 9.2), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.975,
        bottom=0.13,
        top=0.88,
        hspace=0.36,
        wspace=0.27,
        height_ratios=[1.0, 1.08],
    )
    ax_heat = fig.add_subplot(gs[0, :])
    ax_noise = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Radio-PNLF robustness and physical interpretation",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.075,
        0.918,
        "Sample definitions are changed without masking any luminosity interval or individual bright source.",
        ha="left",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    # Robustness heatmap
    sample_sets = [
        ("Primary\nKnown+True, catalogue", primary_fits),
        ("Known only\ncatalogue", known_fits),
        ("All Reid classes\ncatalogue", all_fits),
        ("Known+True\n+ visual diagnostic", inclusive_fits),
    ]
    heat = np.zeros((len(MODEL_ORDER), len(sample_sets)))
    for j, (_, fits) in enumerate(sample_sets):
        amin = min(f.aicc for f in fits)
        mapping = {f.model: f.aicc - amin for f in fits}
        heat[:, j] = [mapping[name] for name in MODEL_ORDER]
    shown = np.clip(heat, 0, 30)
    cmap = LinearSegmentedColormap.from_list("evidence", ["#D1495B", "#F4D6A0", "#EEF1F4", "#AEB7C2"])
    im = ax_heat.imshow(shown, aspect="auto", cmap=cmap, vmin=0, vmax=30)
    ax_heat.set_xticks(np.arange(len(sample_sets)), [name for name, _ in sample_sets])
    ax_heat.set_yticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
    ax_heat.tick_params(axis="x", labelsize=9)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            text = ">30" if val > 30 else f"{val:.1f}"
            ax_heat.text(j, i, text, ha="center", va="center", fontsize=8.5, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.018)
    cbar.set_label(r"$\Delta$AICc (0 = preferred within each sample)")
    ax_heat.set_title("Model preference is tested across independent and inclusive sample definitions", pad=12, fontsize=12)
    add_panel_label(ax_heat, "A")

    # Non-parametric population CDF with detections and radio upper limits.
    limits_primary = log_limit[primary_parent]
    ax_noise.fill_between(
        grid,
        rkm_lo,
        rkm_hi,
        color=COLORS["band"],
        alpha=0.48,
        lw=0,
        label="95% row-bootstrap band",
    )
    ax_noise.step(
        grid,
        rkm,
        where="post",
        color=COLORS["primary_dark"],
        lw=2.3,
        label="Reverse Kaplan-Meier estimate",
    )
    rug = limits_primary[np.isfinite(limits_primary)]
    ax_noise.plot(rug, np.full_like(rug, 0.012), "|", color=COLORS["grey"], ms=6, alpha=0.28, label="Local 5-sigma upper limits")
    ax_noise.set_xlabel(r"$\log_{10}(L_{1.295\,\mathrm{GHz}} / \mathrm{W\,Hz^{-1}})$")
    ax_noise.set_ylabel(r"Parent cumulative fraction $P(\log L_\nu \leq x)$")
    ax_noise.set_xlim(plot_min, plot_max)
    ax_noise.set_ylim(-0.02, 1.02)
    ax_noise.legend(frameon=False, fontsize=8.5)
    ax_noise.text(
        0.065,
        0.97,
        "The distribution below the faintest constraining event is not fully identifiable",
        transform=ax_noise.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
        color=COLORS["muted"],
    )
    tidy_axis(ax_noise)
    add_panel_label(ax_noise, "B")

    # Interpretation panel, populated from the measured results
    ax_text.axis("off")
    delta_aicc = canonical.aicc - best.aicc
    if best.model == "Canonical Ciardullo":
        evidence = f"AICc best, but tied with {best_noncanonical.model} (Delta={best_noncanonical.aicc-best.aicc:.1f})"
    else:
        evidence = "not distinguishable" if delta_aicc < 2 else ("moderately disfavoured" if delta_aicc < 10 else "strongly disfavoured")
    canonical_p = canonical.ks_p_boot
    if np.isfinite(canonical_p):
        null_text = "not rejected" if canonical_p >= 0.05 else "rejected"
    else:
        null_text = "not calibrated"

    lines = [
        ("Primary result", 13, "bold", COLORS["ink"]),
        (f"Preferred shape: {best.model}", 11, "bold", COLORS["best"]),
        (f"Canonical optical form: {evidence}", 10, "normal", COLORS["ink"]),
        (f"Distribution-level null test: {null_text} (bootstrap p={canonical_p:.3f})", 10, "normal", COLORS["ink"]),
        (
            f"Canonical bright-edge parameter: log L = {canonical_cutoff[1]:.3f} "
            f"(+{canonical_cutoff[2]-canonical_cutoff[1]:.3f}/-{canonical_cutoff[1]-canonical_cutoff[0]:.3f}) W Hz$^{{-1}}$",
            9.5,
            "normal",
            COLORS["ink"],
        ),
        ("", 5, "normal", COLORS["ink"]),
        ("Why this inference is safer", 12, "bold", COLORS["ink"]),
        ("- Every catalogue luminosity is used; no bright-source deletion.", 9.5, "normal", COLORS["muted"]),
        ("- Faint incompleteness is modelled per sky position, not by one line.", 9.5, "normal", COLORS["muted"]),
        ("- Non-detections remain as upper limits in the non-parametric CDF.", 9.5, "normal", COLORS["muted"]),
        ("- Flexible ML density is penalised for extra parameters.", 9.5, "normal", COLORS["muted"]),
        ("- Visual recoveries cannot silently shape the primary fit.", 9.5, "normal", COLORS["muted"]),
        ("", 5, "normal", COLORS["ink"]),
        ("Physical caution", 12, "bold", COLORS["ink"]),
        ("The Ciardullo law is an optical [O III] null hypothesis, not a", 9.5, "normal", COLORS["muted"]),
        ("radio prediction. A radio LF traces free-free optical depth, density,", 9.5, "normal", COLORS["muted"]),
        ("ionised mass and expansion; universality requires another galaxy.", 9.5, "normal", COLORS["muted"]),
    ]
    y = 0.96
    for text_line, size, weight, color in lines:
        ax_text.text(0.055, y, text_line.replace("Delta", r"$\Delta$"), transform=ax_text.transAxes, ha="left", va="top", fontsize=size, fontweight=weight, color=color)
        if not text_line:
            y -= 0.022
        elif size >= 12:
            y -= 0.068
        else:
            y -= 0.050
    ax_text.add_patch(
        plt.Rectangle((0.0, 0.01), 0.99, 0.98, transform=ax_text.transAxes, fill=False, edgecolor=COLORS["grid"], lw=1.0)
    )
    add_panel_label(ax_text, "C")

    fig.text(
        0.075,
        0.025,
        "The visual-recovery column is an explicit sensitivity analysis because those sources were selected at known optical positions.",
        ha="left",
        va="bottom",
        fontsize=8.7,
        color=COLORS["muted"],
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"    -> {OUT_PDF.name}")
print("\nstep07e complete")
print(f"  Primary best model: {best.model}")
print(f"  Canonical Delta AICc: {canonical.aicc - best.aicc:.2f}")
print(f"  Canonical bootstrap PIT p: {canonical.ks_p_boot:.4f}")
print("  No optical PNLF constructed. No radio luminosity points manually masked.")
