"""
step02b_crossmatch_askap.py
===========================
Step 2b (ASKAP) — Cross-match the unified Reid LMC PN catalogue against the published ASKAP-EMU
888 MHz LMC source catalogue (Pennock et al. 2021).

Matching policy
---------------
    separation <= 4.5 arcsec : accepted automatically
    4.5 < separation <= 6.0  : manual review
    6.0 < separation <= 10.0 : wide-review candidate; not accepted

The 10 arcsec outer search follows the initial radius used by Pennock et al.,
while their paper cautions that offsets above 6 arcsec are often doubtful.

Outputs
-------
    03_Outputs/step02b_askap_matched.vot
    03_Outputs/step02b_askap_unmatched.vot
    03_Outputs/step02b_askap_review.vot
    04_Figures/step02b_askap_offsets.pdf

ASKAP catalogue fitting errors and the published 8 per cent absolute
calibration term are stored separately.  The total integrated-flux error is
their quadrature sum.

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

warnings.filterwarnings("ignore")
apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

ASKAP_FILE = os.path.join(DATA, "ASKAP_LMC_888MHz_catalogue_full.vot")
REID_FILE = os.path.join(OUTS, "step01a_reid_master.vot")
OUT_MATCHED = os.path.join(OUTS, "step02b_askap_matched.vot")
OUT_UNMATCHED = os.path.join(OUTS, "step02b_askap_unmatched.vot")
OUT_REVIEW = os.path.join(OUTS, "step02b_askap_review.vot")
FIG_OFFSETS = os.path.join(FIGS, "step02b_askap_offsets.pdf")

ACCEPT_RAD = 4.5
REVIEW_RAD = 6.0
OUTER_RAD = 10.0
ASKAP_FREQ_MHZ = 888.0
ASKAP_CAL_FRAC = 0.08
ASKAP_AREA_DEG2 = 120.0

os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

print("step02b: cross-matching Reid against ASKAP-EMU 888 MHz")

print("\n[1] Loading catalogues...")
reid = Table.read(REID_FILE, format="votable")
askap = Table.read(ASKAP_FILE, format="votable")
print(f"    Reid  : {len(reid)} sources")
print(f"    ASKAP : {len(askap)} sources at {ASKAP_FREQ_MHZ:.0f} MHz")

required = {
    "RAJ2000", "DEJ2000", "PeakFlux", "e_PeakFlux", "IntFlux",
    "e_IntFlux", "LocalRMS", "SourceList", "EMU-ID", "Flags",
}
missing = sorted(required - set(askap.colnames))
if missing:
    raise KeyError(f"ASKAP full catalogue is missing required fields: {missing}")

ra_reid = np.asarray(reid["RA"], dtype=float)
dec_reid = np.asarray(reid["Dec"], dtype=float)
valid_reid = np.isfinite(ra_reid) & np.isfinite(dec_reid)
reid_valid_idx = np.where(valid_reid)[0]

ra_askap = np.asarray(askap["RAJ2000"], dtype=float)
dec_askap = np.asarray(askap["DEJ2000"], dtype=float)
valid_askap = np.isfinite(ra_askap) & np.isfinite(dec_askap)
askap_valid_idx = np.where(valid_askap)[0]

coords_reid = SkyCoord(ra_reid[valid_reid] * u.deg, dec_reid[valid_reid] * u.deg)
coords_askap = SkyCoord(ra_askap[valid_askap] * u.deg, dec_askap[valid_askap] * u.deg)

nearest_valid, sep, _ = coords_reid.match_to_catalog_sky(coords_askap)
nearest_rows = askap_valid_idx[nearest_valid]
sep_arcsec_valid = sep.to_value(u.arcsec)

nearest_full = np.full(len(reid), -1, dtype=int)
sep_full = np.full(len(reid), np.nan)
nearest_full[reid_valid_idx] = nearest_rows
sep_full[reid_valid_idx] = sep_arcsec_valid

accepted = valid_reid & (sep_full <= ACCEPT_RAD)
review = valid_reid & (sep_full > ACCEPT_RAD) & (sep_full <= OUTER_RAD)
borderline = valid_reid & (sep_full > ACCEPT_RAD) & (sep_full <= REVIEW_RAD)
wide_review = valid_reid & (sep_full > REVIEW_RAD) & (sep_full <= OUTER_RAD)

print("\n[2] Match counts...")
print(f"    Accepted       (<= {ACCEPT_RAD:.1f}\") : {accepted.sum()}")
print(f"    Borderline ({ACCEPT_RAD:.1f}-{REVIEW_RAD:.1f}\") : {borderline.sum()}")
print(f"    Wide review ({REVIEW_RAD:.1f}-{OUTER_RAD:.1f}\") : {wide_review.sum()}")

# Approximate background expectation for a uniformly distributed catalogue.
surface_density = len(askap) / ASKAP_AREA_DEG2
chance_per_reid = np.pi * (ACCEPT_RAD / 3600.0) ** 2 * surface_density
print(f"    Uniform-background expectation at {ACCEPT_RAD:.1f}\": "
      f"{chance_per_reid * len(reid):.2f} matches")

def values_for_matches(column_name, fill=np.nan, dtype=float):
    out = np.full(len(reid), fill, dtype=dtype)
    rows = np.where(accepted | review)[0]
    src = nearest_full[rows]
    out[rows] = np.asarray(askap[column_name][src], dtype=dtype)
    return out

def strings_for_matches(column_name, width):
    out = np.full(len(reid), "", dtype=f"U{width}")
    rows = np.where(accepted | review)[0]
    src = nearest_full[rows]
    out[rows] = np.asarray(askap[column_name][src], dtype=str)
    return out

peak = values_for_matches("PeakFlux")
err_peak_fit = values_for_matches("e_PeakFlux")
int_flux = values_for_matches("IntFlux")
err_int_fit = values_for_matches("e_IntFlux")
local_rms = values_for_matches("LocalRMS")

# VizieR uses negative sentinel values for unavailable fitted uncertainties.
err_peak_fit[err_peak_fit < 0] = np.nan
err_int_fit[err_int_fit < 0] = np.nan

err_int_total = np.sqrt(err_int_fit ** 2 + (ASKAP_CAL_FRAC * int_flux) ** 2)
snr = np.divide(peak, local_rms, out=np.full(len(reid), np.nan), where=local_rms > 0)

match_status = np.full(len(reid), "unmatched", dtype="U16")
match_status[accepted] = "accepted"
match_status[borderline] = "review"
match_status[wide_review] = "wide_review"

emu_id = strings_for_matches("EMU-ID", 32)
source_type = np.full(len(reid), "", dtype="U12")
source_type[np.char.find(emu_id, " ES ") >= 0] = "isolated"
source_type[np.char.find(emu_id, " EC ") >= 0] = "component"

all_rows = reid.copy()
columns = [
    Column(np.full(len(reid), ASKAP_FREQ_MHZ, dtype="f4"), name="askap_frequency_MHz", unit="MHz"),
    Column(sep_full.astype("f4"), name="askap_separation_arcsec", unit="arcsec"),
    Column(match_status, name="askap_match_status"),
    Column(emu_id, name="askap_id"),
    Column(strings_for_matches("SourceList", 8), name="askap_source_list"),
    Column(values_for_matches("RAJ2000"), name="askap_ra", unit="deg"),
    Column(values_for_matches("DEJ2000"), name="askap_dec", unit="deg"),
    Column(peak.astype("f4"), name="askap_peak_flux_mJy_beam", unit="mJy/beam"),
    Column(err_peak_fit.astype("f4"), name="askap_err_peak_flux_fit_mJy_beam", unit="mJy/beam"),
    Column(int_flux.astype("f4"), name="askap_int_flux_mJy", unit="mJy"),
    Column(err_int_fit.astype("f4"), name="askap_err_int_flux_fit_mJy", unit="mJy"),
    Column(np.full(len(reid), ASKAP_CAL_FRAC, dtype="f4"), name="askap_calibration_fraction"),
    Column(err_int_total.astype("f4"), name="askap_err_int_flux_total_mJy", unit="mJy"),
    Column(local_rms.astype("f4"), name="askap_local_rms_mJy_beam", unit="mJy/beam"),
    Column(snr.astype("f4"), name="askap_snr"),
    Column(values_for_matches("a").astype("f4"), name="askap_a_arcsec", unit="arcsec"),
    Column(values_for_matches("b").astype("f4"), name="askap_b_arcsec", unit="arcsec"),
    Column(values_for_matches("PA").astype("f4"), name="askap_pa_deg", unit="deg"),
    Column(strings_for_matches("Flags", 4), name="askap_fit_flag"),
    Column(source_type, name="askap_source_type"),
]
for column in columns:
    all_rows.add_column(column)

all_rows[accepted].write(OUT_MATCHED, format="votable", overwrite=True)
all_rows[~accepted].write(OUT_UNMATCHED, format="votable", overwrite=True)
all_rows[review].write(OUT_REVIEW, format="votable", overwrite=True)

print("\n[3] Writing catalogues...")
print(f"    {os.path.basename(OUT_MATCHED)}   : {accepted.sum()}")
print(f"    {os.path.basename(OUT_UNMATCHED)} : {(~accepted).sum()}")
print(f"    {os.path.basename(OUT_REVIEW)}    : {review.sum()}")

# Offset diagnostic for accepted and review candidates.
plot_mask = accepted | review
plot_rows = np.where(plot_mask)[0]
ask_rows = nearest_full[plot_rows]
d_ra = (ra_askap[ask_rows] - ra_reid[plot_rows]) * np.cos(np.deg2rad(dec_reid[plot_rows])) * 3600.0
d_dec = (dec_askap[ask_rows] - dec_reid[plot_rows]) * 3600.0

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].scatter(d_ra[accepted[plot_rows]], d_dec[accepted[plot_rows]], s=22,
                color="#E67E22", label="Accepted", alpha=0.85)
axes[0].scatter(d_ra[review[plot_rows]], d_dec[review[plot_rows]], s=32,
                facecolors="none", edgecolors="#333333", label="Review")
for radius, style in [(ACCEPT_RAD, "-"), (REVIEW_RAD, "--"), (OUTER_RAD, ":")]:
    axes[0].add_patch(plt.Circle((0, 0), radius, fill=False, color="#666666", ls=style, lw=1))
axes[0].set_xlabel(r"$\Delta$RA cos(Dec) [arcsec]")
axes[0].set_ylabel(r"$\Delta$Dec [arcsec]")
axes[0].set_aspect("equal")
axes[0].legend(fontsize=9)

finite_sep = sep_full[np.isfinite(sep_full) & (sep_full <= OUTER_RAD)]
axes[1].hist(finite_sep, bins=np.arange(0, OUTER_RAD + 0.5, 0.5),
             color="#E67E22", edgecolor="white")
axes[1].axvline(ACCEPT_RAD, color="#333333", ls="-", label="Accepted limit")
axes[1].axvline(REVIEW_RAD, color="#666666", ls="--", label="Review limit")
axes[1].set_xlabel("Nearest ASKAP separation [arcsec]")
axes[1].set_ylabel("Number of Reid sources")
axes[1].legend(fontsize=9)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIG_OFFSETS, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Figure: {FIG_OFFSETS}")

print("\nstep02b complete")
