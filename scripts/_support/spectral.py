"""Weighted power-law fit to a radio spectrum, and the reliability cuts.

Used twice: once for the PN detections, and once for the whole MeerKAT 5 sigma
catalogue when the PN spectral indices are compared against the general radio
field.  The comparison only means anything if both populations are fitted the
same way, so the fit lives here rather than being written out twice.

S_nu = 10^c nu^alpha, fitted as a straight line in log10 S against log10 nu
with the flux uncertainties propagated into logarithmic space:
sigma_logS = sigma_S / (S ln10).
"""

import numpy as np


def fit(frequencies, fluxes, errors):
    """Fit one row per source.

    frequencies, fluxes and errors are (n_sources, n_points) arrays; entries
    that are not finite, not positive, or have no usable error are dropped
    from that source's fit.  Returns alpha, intercept, alpha_err, chi2_nu and
    the number of points used.
    """
    frequencies = np.atleast_2d(np.asarray(frequencies, dtype=float))
    fluxes = np.atleast_2d(np.asarray(fluxes, dtype=float))
    errors = np.atleast_2d(np.asarray(errors, dtype=float))

    good = (np.isfinite(frequencies) & (frequencies > 0)
            & np.isfinite(fluxes) & (fluxes > 0)
            & np.isfinite(errors) & (errors > 0))

    x = np.where(good, np.log10(np.where(good, frequencies, 1.0)), 0.0)
    y = np.where(good, np.log10(np.where(good, fluxes, 1.0)), 0.0)
    sigma = np.where(good, errors / (np.where(good, fluxes, 1.0) * np.log(10.0)), np.inf)
    w = np.where(good, 1.0 / sigma ** 2, 0.0)

    n = good.sum(axis=1)
    sw = w.sum(axis=1)
    swx = (w * x).sum(axis=1)
    swy = (w * y).sum(axis=1)
    swxx = (w * x * x).sum(axis=1)
    swxy = (w * x * y).sum(axis=1)

    delta = sw * swxx - swx ** 2
    usable = (n >= 2) & (delta > 0)

    alpha = np.full(len(n), np.nan)
    intercept = np.full(len(n), np.nan)
    alpha_err = np.full(len(n), np.nan)
    chi2_nu = np.full(len(n), np.nan)

    d = np.where(usable, delta, 1.0)
    a = (sw * swxy - swx * swy) / d
    c = (swxx * swy - swx * swxy) / d
    alpha[usable] = a[usable]
    intercept[usable] = c[usable]
    alpha_err[usable] = np.sqrt(sw[usable] / delta[usable])

    residual = y - (a[:, None] * x + c[:, None])
    chi2 = (w * residual ** 2).sum(axis=1)
    dof = n - 2
    has_dof = usable & (dof > 0)
    chi2_nu[has_dof] = chi2[has_dof] / dof[has_dof]

    return alpha, intercept, alpha_err, chi2_nu, n.astype(int)


def reliable(n_points, alpha_err, chi2_nu, cuts):
    """The three quality cuts of Appendix A, applied together."""
    return (np.asarray(n_points) >= cuts["min_points_reliable"]) \
        & np.isfinite(alpha_err) & (np.asarray(alpha_err) < cuts["max_alpha_error"]) \
        & np.isfinite(chi2_nu) & (np.asarray(chi2_nu) < cuts["max_reduced_chisq"])


def classify(alpha, cuts):
    """thermal / uncertain / steep for indices that passed the cuts.

    Anything outside the physical free-free range is not one of the three
    classes and is returned as an empty string for the caller to record as
    having no reliable index.
    """
    alpha = np.asarray(alpha, dtype=float)
    thermal_low, thermal_high = cuts["thermal_range"]
    out = np.full(len(alpha), "", dtype="U9")
    out[alpha < cuts["steep_below"]] = "steep"
    out[(alpha >= cuts["steep_below"]) & (alpha < thermal_low)] = "uncertain"
    out[(alpha >= thermal_low) & (alpha <= thermal_high)] = "thermal"
    return out
