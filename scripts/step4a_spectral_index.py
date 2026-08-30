"""
step4a_spectral_index.py
========================
Fit one spectral energy distribution per Reid source using every available
independent radio-frequency detection:

    * up to 12 MeerKAT subchannels (908.037--1656.2 MHz; channels 8/9 excluded)
    * the ASKAP 888 MHz integrated flux, when the source is detected by ASKAP

The MeerKAT broadband/central-frequency flux is shown on inspection plots as a
reference point but is not fitted because it is derived from the same data as
the subchannels and would otherwise be counted twice.

Physical spectral-index classification
--------------------------------------
Applied only after the fit passes the quality cuts:
    thermal    : -0.2 <= alpha <= +2.0
    uncertain  : -0.5 <= alpha < -0.2  (intermediate physical interval)
    steep      : alpha < -0.5

Fits that fail the quality cuts, sources without a fit, and the physically
unexpected alpha > +2 category are labelled ``no_reliable_alpha`` rather than
being mixed into the physical uncertain interval.  Every detected Reid source
still gets an inspection page.  ASKAP points are orange and MeerKAT points are
blue.

Inputs
------
    03_Outputs/step3d_multisurvey_detected_union.vot
    01_Data/MeerKAT_Final_5sigma.vot

Outputs
-------
    03_Outputs/step4a_spectral_index.vot
    05_Figures/step4a_spectral_index_distribution.pdf
    04_Inspect/figures/step4a_sed_thermal.pdf
    04_Inspect/figures/step4a_sed_uncertain.pdf
    04_Inspect/figures/step4a_sed_steep.pdf
    04_Inspect/figures/step4a_sed_no_reliable_alpha.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import warnings

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

warnings.filterwarnings("ignore")
apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "05_Figures")
INSP = os.path.join(BASE, "04_Inspect", "figures")

DETECTED_FILE = os.path.join(OUTS, "step3d_multisurvey_detected_union.vot")
MKT5_FILE = os.path.join(DATA, "MeerKAT_Final_5sigma.vot")
OUT_VOT = os.path.join(OUTS, "step4a_spectral_index.vot")
FIG_HIST = os.path.join(FIGS, "step4a_spectral_index_distribution.pdf")

LOOKUP_RAD = 4.5
STEEP_BOUNDARY = -0.5
THERMAL_BOUNDARY = -0.2
THERMAL_MAXIMUM = 2.0
MIN_FIT_POINTS = 2
MIN_RELIABLE_POINTS = 4
MAX_REDUCED_CHISQ = 10.0
MAX_ALPHA_ERROR = 0.5
ASKAP_FREQ_GHZ = 0.888
MEERKAT_CENTRAL_FREQ_GHZ = 1.295

# Channels 8 and 9 were omitted from the published spectral fits because of
# RFI.  Astropy appends _1 to the integrated-flux FIELD names in this
# VOTable (the XML contains duplicate identifiers), so the resolver below
# accepts both the Astropy names and their unsuffixed equivalents.
CHANNEL_FREQUENCIES_MHZ = {
    "ch2": 908.037,
    "ch3": 952.342,
    "ch4": 996.646,
    "ch5": 1043.46,
    "ch6": 1092.78,
    "ch7": 1144.61,
    "ch10": 1317.23,
    "ch11": 1381.18,
    "ch12": 1448.05,
    "ch13": 1519.94,
    "ch14": 1593.92,
    "ch15": 1656.2,
}

os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
os.makedirs(INSP, exist_ok=True)

print("step4a: fitting spectral indices")

print("\n[1] Loading catalogues...")
detected = Table.read(DETECTED_FILE, format="votable")
mkt5 = Table.read(MKT5_FILE, format="votable")
print(f"    Multisurvey detections : {len(detected)}")
print(f"    MeerKAT 5sigma table   : {len(mkt5)}")

def find_col(table, candidates):
    for candidate in candidates:
        if candidate in table.colnames:
            return candidate
    raise KeyError(f"None of {candidates} found")

ra5_col = find_col(mkt5, ["RAJ2000_deg", "dRAJ2000", "ra"])
dec5_col = find_col(mkt5, ["DECJ2000_deg", "dDECJ2000", "dec"])
sp_col = find_col(mkt5, ["spindex", "new_alpha_odr"])
spe_col = find_col(mkt5, ["spindex_err", "new_err_alpha_odr"])
rchi_col = find_col(mkt5, ["redchisq", "new_red_chisq_odr"])
npt_col = find_col(mkt5, ["n_points", "new_n_points"])

available_channels = {}
for channel, frequency in CHANNEL_FREQUENCIES_MHZ.items():
    flux_candidates = [f"{channel}_int_flux_1", f"{channel}_int_flux"]
    error_candidates = [f"{channel}_err_int_flux", f"{channel}_err_int_flux_1"]
    flux_column = next((c for c in flux_candidates if c in mkt5.colnames), None)
    error_column = next((c for c in error_candidates if c in mkt5.colnames), None)
    if flux_column is not None:
        available_channels[channel] = (frequency, flux_column, error_column)

print(f"    Available MeerKAT subchannels: {len(available_channels)}")
if len(available_channels) != 12:
    print("    WARNING: expected 12 MeerKAT subchannels")

coords_mkt5 = SkyCoord(
    np.asarray(mkt5[ra5_col], dtype=float) * u.deg,
    np.asarray(mkt5[dec5_col], dtype=float) * u.deg,
)

det_meerkat = np.asarray(detected["det_meerkat"], dtype=int) == 1
det_askap = np.asarray(detected["det_askap"], dtype=int) == 1
ra_opt = np.asarray(detected["RA"], dtype=float)
dec_opt = np.asarray(detected["Dec"], dtype=float)
ra_radio = np.asarray(detected["mkt_ra"], dtype=float)
dec_radio = np.asarray(detected["mkt_dec"], dtype=float)
fallback = ~np.isfinite(ra_radio) | ~np.isfinite(dec_radio)
ra_radio[fallback] = ra_opt[fallback]
dec_radio[fallback] = dec_opt[fallback]

# Link only actual MeerKAT detections to the 5sigma multichannel table.  This
# prevents an unrelated nearby MeerKAT source being attached to an ASKAP-only PN.
mkt5_idx = np.full(len(detected), -1, dtype=int)
for k in np.where(det_meerkat)[0]:
    if not (np.isfinite(ra_radio[k]) and np.isfinite(dec_radio[k])):
        continue
    coord = SkyCoord(ra_radio[k] * u.deg, dec_radio[k] * u.deg)
    index, separation, _ = coord.match_to_catalog_sky(coords_mkt5)
    if separation.to_value(u.arcsec) <= LOOKUP_RAD:
        mkt5_idx[k] = int(index)

def extract_meerkat_channels(row):
    frequencies, fluxes, errors = [], [], []
    for frequency_mhz, flux_column, error_column in available_channels.values():
        flux_jy = float(row[flux_column])
        error_jy = float(row[error_column]) if error_column else np.nan
        if not (np.isfinite(flux_jy) and flux_jy > 0):
            continue
        if not (np.isfinite(error_jy) and error_jy > 0):
            error_jy = 0.10 * flux_jy
        frequencies.append(frequency_mhz / 1000.0)
        fluxes.append(flux_jy * 1000.0)
        errors.append(error_jy * 1000.0)
    return (np.asarray(frequencies), np.asarray(fluxes), np.asarray(errors))

def askap_point(k):
    if not det_askap[k]:
        return None
    flux = float(detected["askap_int_flux_mJy"][k])
    error = float(detected["askap_err_int_flux_total_mJy"][k])
    if not (np.isfinite(flux) and flux > 0):
        return None
    if not (np.isfinite(error) and error > 0):
        error = 0.08 * flux
    return ASKAP_FREQ_GHZ, flux, error

def combined_sed(k):
    if mkt5_idx[k] >= 0:
        mk_freq, mk_flux, mk_error = extract_meerkat_channels(mkt5[mkt5_idx[k]])
    else:
        mk_freq = np.array([]); mk_flux = np.array([]); mk_error = np.array([])
    ap = askap_point(k)
    if ap is None:
        fit_freq = mk_freq.copy(); fit_flux = mk_flux.copy(); fit_error = mk_error.copy()
    else:
        fit_freq = np.append(mk_freq, ap[0])
        fit_flux = np.append(mk_flux, ap[1])
        fit_error = np.append(mk_error, ap[2])
    order = np.argsort(fit_freq)
    return (mk_freq, mk_flux, mk_error, ap,
            fit_freq[order], fit_flux[order], fit_error[order])

def weighted_spectral_fit(frequencies, fluxes, errors):
    """Weighted linear fit to log10(S) = alpha*log10(nu) + intercept."""
    valid = (
        np.isfinite(frequencies) & (frequencies > 0) &
        np.isfinite(fluxes) & (fluxes > 0) &
        np.isfinite(errors) & (errors > 0)
    )
    frequencies = frequencies[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]
    if len(frequencies) < MIN_FIT_POINTS:
        return np.nan, np.nan, np.nan, np.nan
    x = np.log10(frequencies)
    y = np.log10(fluxes)
    sigma_y = errors / (fluxes * np.log(10.0))
    weights = 1.0 / sigma_y ** 2
    design = np.column_stack([x, np.ones_like(x)])
    normal = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    parameters = covariance @ (design.T @ (weights * y))
    alpha, intercept = parameters
    residuals = y - (alpha * x + intercept)
    chi2 = float(np.sum((residuals / sigma_y) ** 2))
    dof = len(x) - 2
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan
    alpha_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    return float(alpha), float(intercept), alpha_error, reduced_chi2

n_sources = len(detected)
alpha = np.full(n_sources, np.nan)
intercept = np.full(n_sources, np.nan)
alpha_error = np.full(n_sources, np.nan)
reduced_chi2 = np.full(n_sources, np.nan)
n_fit_points = np.zeros(n_sources, dtype=np.int16)
n_meerkat_points = np.zeros(n_sources, dtype=np.int16)
has_askap_point = np.zeros(n_sources, dtype=np.int16)
mkt_ch4_flux_jy = np.full(n_sources, np.nan)
mkt_ch4_error_jy = np.full(n_sources, np.nan)

print("\n[2] Fitting all available MeerKAT + ASKAP detections...")
for k in range(n_sources):
    mk_f, mk_s, mk_e, ap, fit_f, fit_s, fit_e = combined_sed(k)
    n_meerkat_points[k] = len(mk_f)
    has_askap_point[k] = int(ap is not None)
    n_fit_points[k] = len(fit_f)
    alpha[k], intercept[k], alpha_error[k], reduced_chi2[k] = weighted_spectral_fit(
        fit_f, fit_s, fit_e
    )
    if mkt5_idx[k] >= 0 and "ch4" in available_channels:
        _, flux_column, error_column = available_channels["ch4"]
        row = mkt5[mkt5_idx[k]]
        try:
            ch4_flux = float(row[flux_column])
            if np.isfinite(ch4_flux) and ch4_flux > 0:
                mkt_ch4_flux_jy[k] = ch4_flux
                ch4_error = float(row[error_column]) if error_column else np.nan
                if np.isfinite(ch4_error) and ch4_error > 0:
                    mkt_ch4_error_jy[k] = ch4_error
        except (TypeError, ValueError):
            pass

# Preserve the published MeerKAT-only fit for comparison.
alpha_mkt_catalogue = np.full(n_sources, np.nan)
alpha_mkt_catalogue_error = np.full(n_sources, np.nan)
alpha_mkt_catalogue_chi2 = np.full(n_sources, np.nan)
alpha_mkt_catalogue_npoints = np.zeros(n_sources, dtype=np.int16)
for k in np.where(mkt5_idx >= 0)[0]:
    row = mkt5[mkt5_idx[k]]
    alpha_mkt_catalogue[k] = float(row[sp_col])
    alpha_mkt_catalogue_error[k] = float(row[spe_col])
    alpha_mkt_catalogue_chi2[k] = float(row[rchi_col])
    alpha_mkt_catalogue_npoints[k] = int(row[npt_col])

# Three exclusive physical categories are assigned only to reliable fits.
# Measurement/fit failures are kept separate from the physical intermediate
# interval so that "uncertain" retains its intended alpha-range meaning.
sp_class = np.full(n_sources, "no_reliable_alpha", dtype="U20")
fit_status = np.full(n_sources, "no_fit", dtype="U24")
for k in range(n_sources):
    if not np.isfinite(alpha[k]):
        continue
    if n_fit_points[k] < MIN_RELIABLE_POINTS:
        fit_status[k] = "insufficient_points"
        continue
    if not (np.isfinite(alpha_error[k]) and alpha_error[k] < MAX_ALPHA_ERROR):
        fit_status[k] = "large_alpha_error"
        continue
    if not (np.isfinite(reduced_chi2[k]) and reduced_chi2[k] < MAX_REDUCED_CHISQ):
        fit_status[k] = "poor_fit"
        continue
    fit_status[k] = "reliable"
    if alpha[k] < STEEP_BOUNDARY:
        sp_class[k] = "steep"
    elif alpha[k] < THERMAL_BOUNDARY:
        sp_class[k] = "uncertain"
    elif alpha[k] <= THERMAL_MAXIMUM:
        sp_class[k] = "thermal"
    else:
        sp_class[k] = "no_reliable_alpha"
        fit_status[k] = "alpha_above_physical_range"

n_thermal = int(np.sum(sp_class == "thermal"))
n_uncertain = int(np.sum(sp_class == "uncertain"))
n_steep = int(np.sum(sp_class == "steep"))
n_no_reliable = int(np.sum(sp_class == "no_reliable_alpha"))
n_with_fit = int(np.sum(np.isfinite(alpha)))

print(f"    With a fitted alpha : {n_with_fit}/{n_sources}")
print(f"    Thermal             : {n_thermal}")
print(f"    Uncertain           : {n_uncertain}")
print(f"    Steep               : {n_steep}")
print(f"    No reliable alpha   : {n_no_reliable}")

print("\n[3] Writing spectral catalogue...")
replace_columns = [
    "alpha", "alpha_err", "alpha_redchisq", "alpha_n_points", "sp_class",
    "alpha_combined", "alpha_combined_err", "alpha_combined_redchisq",
    "alpha_combined_n_points", "alpha_n_meerkat_points", "alpha_has_askap",
    "alpha_fit_status", "alpha_meerkat_catalogue", "alpha_meerkat_catalogue_err",
    "alpha_meerkat_catalogue_redchisq", "alpha_meerkat_catalogue_n_points",
    "mkt_ch4_996p646_flux_Jy", "mkt_ch4_996p646_err_Jy",
]
for name in replace_columns:
    if name in detected.colnames:
        detected.remove_column(name)

for column in [
    Column(alpha.astype("f4"), name="alpha"),
    Column(alpha_error.astype("f4"), name="alpha_err"),
    Column(reduced_chi2.astype("f4"), name="alpha_redchisq"),
    Column(n_fit_points, name="alpha_n_points"),
    Column(alpha.astype("f4"), name="alpha_combined"),
    Column(alpha_error.astype("f4"), name="alpha_combined_err"),
    Column(reduced_chi2.astype("f4"), name="alpha_combined_redchisq"),
    Column(n_fit_points, name="alpha_combined_n_points"),
    Column(n_meerkat_points, name="alpha_n_meerkat_points"),
    Column(has_askap_point, name="alpha_has_askap"),
    Column(fit_status, name="alpha_fit_status"),
    Column(sp_class, name="sp_class"),
    Column(alpha_mkt_catalogue.astype("f4"), name="alpha_meerkat_catalogue"),
    Column(alpha_mkt_catalogue_error.astype("f4"), name="alpha_meerkat_catalogue_err"),
    Column(alpha_mkt_catalogue_chi2.astype("f4"), name="alpha_meerkat_catalogue_redchisq"),
    Column(alpha_mkt_catalogue_npoints, name="alpha_meerkat_catalogue_n_points"),
    Column(mkt_ch4_flux_jy.astype("f4"), name="mkt_ch4_996p646_flux_Jy"),
    Column(mkt_ch4_error_jy.astype("f4"), name="mkt_ch4_996p646_err_Jy"),
]:
    detected.add_column(column)

detected.write(OUT_VOT, format="votable", overwrite=True)
print(f"    {OUT_VOT}")

print("\n[4] Generating spectral-index distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(-1.5, 2.05, 0.10)
for label, color, display in [
    ("steep", "#C23B33", "Steep / non-thermal"),
    ("uncertain", "#F2A541", "Uncertain / intermediate"),
    ("thermal", "#2166AC", "Thermal"),
]:
    values = alpha[(sp_class == label) & np.isfinite(alpha)]
    ax.hist(values, bins=bins, color=color, edgecolor="white", lw=0.6,
            alpha=1.0, label=f"{display} (N={len(values)})")
ax.axvline(STEEP_BOUNDARY, color="#333333", lw=1.4, ls="--")
ax.axvline(THERMAL_BOUNDARY, color="#333333", lw=1.4, ls="--")
ylim_top = ax.get_ylim()[1]
ax.text(STEEP_BOUNDARY - 0.02, ylim_top * 0.94, r"$-0.5$",
        fontsize=9, ha="right", va="top")
ax.text(THERMAL_BOUNDARY + 0.02, ylim_top * 0.94, r"$-0.2$",
        fontsize=9, ha="left", va="top")
ax.text(0.98, 0.70, f"No reliable $\\alpha$: N={n_no_reliable}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bbbbbb"))
ax.set_xlabel(r"Combined spectral index ($S_\nu \propto \nu^\alpha$)")
ax.set_ylabel("Number of reliable fits")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_HIST, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()

def safe_value(table, column, index, default=np.nan):
    if column not in table.colnames:
        return default
    try:
        value = float(table[column][index])
        return value if np.isfinite(value) else default
    except Exception:
        return default

def write_sed_pdf(output_path, indices, category):
    with PdfPages(output_path) as pdf:
        for k in indices:
            mk_freq, mk_flux, mk_error, ap, fit_freq, fit_flux, fit_error = combined_sed(k)
            rpid = str(detected["RP_ID"][k]).strip() or f"row_{k}"
            reid_class = str(detected["reid_classification"][k]).strip()
            surveys = str(detected["detection_surveys"][k]).strip()

            fig, ax = plt.subplots(figsize=(8, 6))
            if len(mk_freq):
                ax.errorbar(mk_freq, mk_flux, yerr=mk_error, fmt="o", capsize=4,
                            color="#2166AC", markersize=6, label="MeerKAT subchannels", zorder=5)
            if ap is not None:
                ax.errorbar([ap[0]], [ap[1]], yerr=[ap[2]], fmt="s", capsize=5,
                            color="#E67E22", markersize=7, label="ASKAP 888 MHz", zorder=6)

            # Show, but do not fit, the MeerKAT broadband central-frequency point.
            central_flux = safe_value(detected, "mkt_int_flux_Jy", k) * 1000.0
            central_error = safe_value(detected, "mkt_err_int_flux_Jy", k) * 1000.0
            if np.isfinite(central_flux) and central_flux > 0:
                yerr = [central_error] if np.isfinite(central_error) and central_error > 0 else None
                ax.errorbar([MEERKAT_CENTRAL_FREQ_GHZ], [central_flux], yerr=yerr,
                            fmt="D", mfc="none", mec="#2166AC", ecolor="#2166AC",
                            markersize=6, label="MeerKAT 1.295 GHz broadband (not fitted)")

            if np.isfinite(alpha[k]) and len(fit_freq) >= MIN_FIT_POINTS:
                x_min = max(0.75, fit_freq.min() * 0.92)
                x_max = min(1.85, fit_freq.max() * 1.08)
                frequency_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)
                model_flux = 10 ** (alpha[k] * np.log10(frequency_grid) + intercept[k])
                ax.plot(frequency_grid, model_flux, color="#222222", lw=1.8,
                        label=(rf"Combined fit: $\alpha={alpha[k]:.2f}\pm{alpha_error[k]:.2f}$"
                               + "\n" + rf"$\chi^2_\nu={reduced_chi2[k]:.2f}$; "
                               + f"N={n_fit_points[k]}"))
            else:
                ax.text(0.5, 0.55, "Insufficient independent frequency points for a fit",
                        transform=ax.transAxes, ha="center", va="center")

            positive_flux = fit_flux[np.isfinite(fit_flux) & (fit_flux > 0)]
            if np.isfinite(central_flux) and central_flux > 0:
                positive_flux = np.append(positive_flux, central_flux)
            if len(positive_flux):
                ax.set_yscale("log")
                ax.set_ylim(positive_flux.min() * 0.45, positive_flux.max() * 2.2)
            ax.set_xscale("log")
            ax.set_xlim(0.82, 1.78)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Integrated flux density (mJy)")
            ax.set_title(f"{rpid}  |  Reid: {reid_class}  |  {surveys}  |  {category}")
            ax.grid(True, which="both", alpha=0.2)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            if len(positive_flux):
                ax.yaxis.set_major_locator(LogLocator(base=10.0, subs="auto", numticks=10))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_major_formatter(ScalarFormatter())
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=9, loc="best")
            plt.tight_layout()
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)

print("\n[5] Generating one SED page for every detected Reid source...")
pdf_outputs = []
for category in ["thermal", "uncertain", "steep", "no_reliable_alpha"]:
    indices = np.where(sp_class == category)[0]
    output = os.path.join(INSP, f"step4a_sed_{category}.pdf")
    write_sed_pdf(output, indices, category)
    pdf_outputs.append(output)
    print(f"    {os.path.basename(output)}")

print("\nstep4a complete")
print(f"    Catalogue : {OUT_VOT}")
print(f"    Histogram : {FIG_HIST}")
for output in pdf_outputs:
    print(f"    SED PDF   : {output}")
