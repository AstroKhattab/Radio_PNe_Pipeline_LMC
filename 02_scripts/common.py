"""Shared helpers for the LMC radio PN pipeline.

Every step script imports this module.  It reads config.yaml, gives the input
and output paths, reads and writes the shared catalogue, writes the numbers
report and the LaTeX tables, and holds the few calculations that more than one
step needs (sky matching, the spectral fit, flux to magnitude).
"""

import logging
import os
import re
import warnings

import numpy as np
import yaml
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

import matplotlib
matplotlib.use("Agg")

warnings.simplefilter("ignore", AstropyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("fontTools").setLevel(logging.ERROR)   # font-embedding chatter in PDF output

ROOT = os.path.dirname(os.path.abspath(__file__))


# --- configuration and paths -------------------------------------------------

def load_config():
    """Read config.yaml, or the file named by $PN_CONFIG when it is set."""
    path = os.environ.get("PN_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


CFG = load_config()


def _dir(key):
    path = os.path.expanduser(CFG["paths"][key])
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    os.makedirs(path, exist_ok=True)
    return path


def data(key):
    return os.path.join(_dir("data"), CFG["inputs"][key])


def figure(name):
    return os.path.join(_dir("figures"), name)


def table_path(name):
    return os.path.join(_dir("tables"), name)


CATALOGUE = os.path.join(_dir("output"), "catalogue.fits")
NUMBERS = os.path.join(_dir("notes"), "numbers.txt")


# --- the shared catalogue ----------------------------------------------------

def read_catalogue():
    if not os.path.exists(CATALOGUE):
        raise SystemExit("04_catalogues/catalogue.fits not found: run step0_data.py first")
    cat = Table.read(CATALOGUE)
    for name in cat.colnames:
        if cat[name].dtype.kind == "S":
            cat[name] = np.char.strip(cat[name].astype(str))
    return cat


def write_catalogue(cat):
    cat.write(CATALOGUE, overwrite=True)


def set_column(cat, name, values):
    if name in cat.colnames:
        cat.remove_column(name)
    cat[name] = values


def read_input(key):
    tab = Table.read(data(key))
    return tab


def numeric(tab, name):
    """Column as a float array, with masked or blank values as NaN."""
    col = np.ma.asarray(tab[name])
    try:
        return np.asarray(col.astype(float).filled(np.nan), dtype=float)
    except (ValueError, TypeError):
        out = np.full(len(col), np.nan)
        for i, v in enumerate(col):
            try:
                out[i] = float(v)
            except (ValueError, TypeError):
                pass
        return out


def text(tab, name):
    col = np.ma.asarray(tab[name])
    mask = np.ma.getmaskarray(col)
    return np.array(["" if m else str(v).strip() for v, m in zip(col.data, mask)])


# --- numbers report ----------------------------------------------------------

def write_numbers(step, lines):
    """Replace this step's block in 07_notes/numbers.txt.

    `lines` is a list of (label, value) pairs.  Each step owns one block, so
    the steps can be run in any order and rerun alone.
    """
    blocks = {}
    if os.path.exists(NUMBERS):
        current = None
        for line in open(NUMBERS):
            match = re.match(r"^\[(.+)\]$", line.strip())
            if match:
                current = match.group(1)
                blocks[current] = []
            elif current and line.strip():
                blocks[current].append(line.rstrip("\n"))
    blocks[step] = [f"{label:<52s} {value}" for label, value in lines]
    with open(NUMBERS, "w") as fh:
        for key in sorted(blocks):
            fh.write(f"[{key}]\n")
            fh.write("\n".join(blocks[key]) + "\n\n")


# --- LaTeX tables ------------------------------------------------------------

def write_tex(name, columns, rows, align=None, note=None):
    """Write a bare tabular environment; the float and caption stay in the paper."""
    align = align or "l" * len(columns)
    with open(table_path(name), "w") as fh:
        fh.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n")
        fh.write(" & ".join(columns) + " \\\\\n\\midrule\n")
        for row in rows:
            if row == "midrule":
                fh.write("\\midrule\n")
            else:
                fh.write(" & ".join(str(c) for c in row) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
        if note:
            fh.write(f"\\\\[2pt]\n{{\\footnotesize {note}}}\n")


# Names of the Step 4 fit tiers, kept distinct from the Step 6 levels (High, Possible, Weak).
TIER_NAMES = {"high": "High_fit", "medium": "Medium_fit", "low": "Low_fit"}


def tex_escape(value):
    return str(value).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def fmt(value, digits=2, dash="---"):
    if value is None or value is np.ma.masked or not np.isfinite(float(value)):
        return dash
    return f"{float(value):.{digits}f}"


# --- plotting ----------------------------------------------------------------

def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.5, "legend.fontsize": 9, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6,
    })
    return plt


COLOURS = {"thermal": "#2166AC", "uncertain": "#F2A541", "steep": "#C23B33",
           "none": "#B8BEC8"}


# --- calculations ------------------------------------------------------------

def nearest_match(ra1, dec1, ra2, dec2, radius_arcsec):
    """Nearest source in list 2 for every source in list 1.

    Returns (index into list 2 or -1, separation in arcsec).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    good1 = np.isfinite(ra1) & np.isfinite(dec1)
    good2 = np.where(np.isfinite(ra2) & np.isfinite(dec2))[0]
    index = np.full(len(ra1), -1, dtype=int)
    sep = np.full(len(ra1), np.nan)
    c1 = SkyCoord(ra1[good1] * u.deg, dec1[good1] * u.deg)
    c2 = SkyCoord(ra2[good2] * u.deg, dec2[good2] * u.deg)
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    sep[good1] = d2d.to_value(u.arcsec)
    hit = sep[good1] <= radius_arcsec
    full = np.where(good1)[0]
    index[full[hit]] = good2[idx[hit]]
    return index, sep


def spectral_fit(freq, flux, err):
    """Weighted least-squares power law, S = A nu^alpha, in log space.

    Each row is one source; NaN slots are ignored.  Returns alpha, log10 A,
    the error on alpha, reduced chi-square and the number of points.
    """
    freq, flux, err = (np.atleast_2d(np.asarray(a, dtype=float)) for a in (freq, flux, err))
    good = np.isfinite(freq) & np.isfinite(flux) & np.isfinite(err) & (flux > 0) & (err > 0)
    x = np.where(good, np.log10(np.where(good, freq, 1.0)), 0.0)
    y = np.where(good, np.log10(np.where(good, flux, 1.0)), 0.0)
    sy = np.where(good, err / (np.where(good, flux, 1.0) * np.log(10)), 1.0)
    w = np.where(good, 1.0 / sy ** 2, 0.0)
    n = good.sum(axis=1)
    s, sx, sxx = w.sum(1), (w * x).sum(1), (w * x * x).sum(1)
    sxy, syy = (w * x * y).sum(1), (w * y).sum(1)
    delta = s * sxx - sx ** 2
    ok = (n >= 2) & (delta > 0)
    alpha = np.full(len(n), np.nan)
    inter = np.full(len(n), np.nan)
    aerr = np.full(len(n), np.nan)
    chi2nu = np.full(len(n), np.nan)
    alpha[ok] = (s[ok] * sxy[ok] - sx[ok] * syy[ok]) / delta[ok]
    inter[ok] = (sxx[ok] * syy[ok] - sx[ok] * sxy[ok]) / delta[ok]
    aerr[ok] = np.sqrt(s[ok] / delta[ok])
    model = inter[:, None] + alpha[:, None] * x
    chi2 = np.where(good, w * (y - model) ** 2, 0.0).sum(1)
    more = ok & (n > 2)
    chi2nu[more] = chi2[more] / (n[more] - 2)
    return alpha, inter, aerr, chi2nu, n


def distance_cm():
    return CFG["distance_kpc"] * 3.0856776e21


def magnitude(flux_jy):
    """Radio absolute magnitude, M = -2.5 log10(L / L_ref), L = 4 pi d^2 S."""
    lum = 4 * np.pi * distance_cm() ** 2 * np.asarray(flux_jy, dtype=float) * 1e-23
    return -2.5 * np.log10(lum / CFG["pnlf"]["reference_luminosity"])


def flux_at_magnitude(mag):
    """Inverse of magnitude(): flux density in Jy."""
    lum = CFG["pnlf"]["reference_luminosity"] * 10 ** (-np.asarray(mag) / 2.5)
    return lum / (4 * np.pi * distance_cm() ** 2) * 1e23


def spectral_indices(cat):
    """Fit alpha for every radio detection in the catalogue.

    Uses the MeerKAT sub-band flux densities of the catalogue source the PN was
    matched to, plus the ASKAP 888 MHz point where there is one.  The flux-scale
    term of each survey is added in quadrature to every point.  Returns a dict
    of arrays: alpha, alpha_err, chi2nu, n_points, n_meerkat, intercept.
    """
    mk = CFG["meerkat"]
    sub = read_input("meerkat_subbands")
    col = CFG["columns"]["meerkat_subbands"]
    chans = list(mk["channels"].items())
    n = len(cat)

    # Link only MeerKAT catalogue detections, by their radio position.
    linked = np.asarray(cat["mkt_method"], str) == "catalogue"
    idx, _ = nearest_match(np.where(linked, np.asarray(cat["mkt_ra"], float), np.nan),
                           np.where(linked, np.asarray(cat["mkt_dec"], float), np.nan),
                           numeric(sub, col["ra"]), numeric(sub, col["dec"]),
                           CFG["crossmatch"]["radius_arcsec"])
    has = idx >= 0

    slots = len(chans) + 1
    freq = np.full((n, slots), np.nan)
    flux = np.full((n, slots), np.nan)
    err = np.full((n, slots), np.nan)
    for k, (ch, mhz) in enumerate(chans):
        fname = next(c for c in (f"{ch}_int_flux_1", f"{ch}_int_flux") if c in sub.colnames)
        s = numeric(sub, fname) * 1e3
        e = numeric(sub, f"{ch}_err_int_flux") * 1e3
        e = np.where(np.isfinite(e) & (e > 0), e, mk["default_error_fraction"] * s)
        e = np.hypot(e, mk["flux_scale"] * s)
        freq[has, k] = mhz / 1e3
        flux[has, k] = s[idx[has]]
        err[has, k] = e[idx[has]]
    flux[flux <= 0] = np.nan
    n_meerkat = np.isfinite(flux[:, :-1]).sum(1)

    af = np.asarray(cat["askap_flux_mJy"], float)
    ae = np.asarray(cat["askap_err_mJy"], float)
    ok = np.isfinite(af) & (af > 0)
    freq[ok, -1] = CFG["askap"]["frequency_ghz"]
    flux[ok, -1] = af[ok]
    err[ok, -1] = ae[ok]

    alpha, inter, aerr, chi2nu, npts = spectral_fit(freq, flux, err)
    return dict(alpha=alpha, alpha_err=aerr, chi2nu=chi2nu, n_points=npts,
                n_meerkat=n_meerkat, intercept=inter, freq=freq, flux=flux, err=err)


def spectral_class(alpha):
    si = CFG["spectral_index"]
    lo, hi = si["thermal"]
    cls = np.full(len(alpha), "none", dtype="U10")
    fin = np.isfinite(alpha)
    cls[fin & (alpha < si["steep_below"])] = "steep"
    cls[fin & (alpha >= si["steep_below"]) & (alpha < lo)] = "uncertain"
    cls[fin & (alpha >= lo) & (alpha <= hi)] = "thermal"
    cls[fin & (alpha > hi)] = "inverted"
    return cls
