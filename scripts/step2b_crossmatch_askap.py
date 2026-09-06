"""
step2b_crossmatch_askap.py
==========================
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
    03_Outputs/step2b_askap_matched.vot
    03_Outputs/step2b_askap_unmatched.vot
    03_Outputs/step2b_askap_review.vot
    05_Figures/step2b_askap_offsets.pdf

ASKAP catalogue fitting errors and the published 8 per cent absolute
calibration term are stored separately.  The total integrated-flux error is
their quadrature sum.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table
from astropy.utils.exceptions import AstropyWarning

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import AnchoredText

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config
from _support.plot_style import apply_paper_style

warnings.simplefilter("ignore", AstropyWarning)
apply_paper_style()

cfg = config.load()
ASKAP_FILE = cfg.data("askap_catalogue")
REID_FILE = cfg.out("step1_parent_catalogue.vot")
OUT_MATCHED = cfg.out("step2b_askap_matched.vot")
OUT_UNMATCHED = cfg.out("step2b_askap_unmatched.vot")
OUT_REVIEW = cfg.out("step2b_askap_review.vot")
OUT_STATS = cfg.out("step2b_askap_match_stats.csv")
FIG_OFFSETS = cfg.fig("step2b_askap_offsets.pdf")

ACCEPT_RAD = cfg["crossmatch"]["accept_arcsec"]
REVIEW_RAD = cfg["crossmatch"]["askap_review_arcsec"]
OUTER_RAD = cfg["crossmatch"]["askap_outer_arcsec"]
ASKAP_FREQ_MHZ = cfg["askap"]["frequency_mhz"]
ASKAP_CAL_FRAC = cfg["askap"]["calibration_fraction"]
ASKAP_AREA_DEG2 = cfg["crossmatch"]["askap_area_deg2"]

print("step2b: cross-matching the parent catalogue against ASKAP-EMU 888 MHz")

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

# The accepted matches must be one-to-one; nearest neighbour alone does not
# guarantee it, and a shared counterpart would double-count a detection.
used = nearest_full[accepted]
assert len(set(used)) == len(used), "one ASKAP source matched to two PNe"

# Chance coincidences expected if the radio sources were distributed at random.
surface_density = len(askap) / ASKAP_AREA_DEG2
chance = np.pi * (ACCEPT_RAD / 3600.0) ** 2 * surface_density * len(reid)
print(f"    Expected by chance at {ACCEPT_RAD:.1f}\": {chance:.2f} over the parent sample")

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

# Offset diagnostic, drawn in exactly the same style as the MeerKAT panel in
# step2a so the two surveys can be compared by eye without the reader having to
# re-learn the plot.  Only accepted matches are shown; the wider review cases
# are quoted in the text rather than plotted, because including them would
# stretch the axes and make the two figures incomparable.
STYLE  = "ggplot"
HEXBIN = {"gridsize": 30, "cmap": "plasma", "mincnt": 1}
HIST   = {"bins": "auto", "color": "#D32F2F", "alpha": 0.55}
XLIM   = (-5.0, 5.0)
YLIM   = (-5.0, 5.0)
FONTS  = {"axes_label": 13, "axes_title": 13, "tick": 11, "cbar": 11, "stats": 10}

plot_rows = np.where(accepted)[0]
ask_rows = nearest_full[plot_rows]
c_opt = SkyCoord(ra=ra_reid[plot_rows] * u.deg, dec=dec_reid[plot_rows] * u.deg)
c_rad = SkyCoord(ra=ra_askap[ask_rows] * u.deg, dec=dec_askap[ask_rows] * u.deg)
dra_raw, ddec_raw = c_opt.spherical_offsets_to(c_rad)
x = dra_raw.to(u.arcsec).value
y = ddec_raw.to(u.arcsec).value
finite = np.isfinite(x) & np.isfinite(y)
x = x[finite]; y = y[finite]

plt.style.use(STYLE)
fig = plt.figure(figsize=(7, 7))
gs_outer = GridSpec(1, 1, figure=fig)
gs_inner = GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_outer[0],
    width_ratios=[6, 1, 0.35], height_ratios=[1, 6],
    wspace=0.10, hspace=0.10,
)
ax_histx = fig.add_subplot(gs_inner[0, 0])
ax_main  = fig.add_subplot(gs_inner[1, 0])
ax_histy = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)
ax_cbar  = fig.add_subplot(gs_inner[1, 2])

hb = ax_main.hexbin(x, y, **HEXBIN)
ax_main.set_xlabel("ΔRA · cos(δ)  (arcsec)", fontsize=FONTS["axes_label"])
ax_main.set_ylabel("Δδ  (arcsec)",            fontsize=FONTS["axes_label"])
ax_main.set_xlim(*XLIM); ax_main.set_ylim(*YLIM)
ax_main.axhline(0, color="black", lw=1.5)
ax_main.axvline(0, color="black", lw=1.5)
ax_main.grid(axis="both", linestyle="--", alpha=0.3)

ax_histx.hist(x, **HIST)
ax_histx.axvline(np.mean(x), color="black", ls="--", lw=1.2)
ax_histx.set_ylabel("Counts", fontsize=FONTS["tick"])
ax_histx.set_xlim(*XLIM)
plt.setp(ax_histx.get_xticklabels(), visible=False)

ax_histy.hist(y, **HIST, orientation="horizontal")
ax_histy.axhline(np.mean(y), color="black", ls="--", lw=1.2)
ax_histy.set_xlabel("Counts", fontsize=FONTS["tick"])
ax_histy.set_ylim(*YLIM)
plt.setp(ax_histy.get_yticklabels(), visible=False)

cbar = fig.colorbar(hb, cax=ax_cbar)
cbar.set_label("Counts per bin", fontsize=FONTS["cbar"])
cbar.ax.tick_params(labelsize=FONTS["tick"])

stats_txt = "\n".join([
    fr"$\langle\Delta\mathrm{{RA}}\rangle = {np.mean(x):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\mathrm{{RA}}) = {np.std(x):.2f}^{{\prime\prime}}$",
    fr"$\langle\Delta\delta\rangle = {np.mean(y):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\delta) = {np.std(y):.2f}^{{\prime\prime}}$",
    f"$N = {len(x)}$",
])
at = AnchoredText(stats_txt, prop=dict(size=FONTS["stats"]),
                  frameon=True, loc="upper left")
at.patch.set_boxstyle("round,pad=0.3"); at.patch.set_alpha(0.9)
ax_main.add_artist(at)

# NO title — added in Overleaf
plt.savefig(FIG_OFFSETS, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    Figure: {FIG_OFFSETS}")

stats = Table()
stats["quantity"] = ["n_parent", "n_accepted", "n_borderline_4p5_to_6",
                     "n_wide_review_6_to_10", "median_offset_arcsec",
                     "mean_dra_cosdec_arcsec", "mean_ddec_arcsec",
                     "std_dra_cosdec_arcsec", "std_ddec_arcsec",
                     "expected_chance_matches"]
stats["value"] = [len(reid), int(accepted.sum()), int(borderline.sum()),
                  int(wide_review.sum()), float(np.nanmedian(sep_full[accepted])),
                  float(np.mean(x)), float(np.mean(y)),
                  float(np.std(x)), float(np.std(y)), float(chance)]
stats.write(OUT_STATS, format="ascii.csv", overwrite=True)
print(f"    {os.path.basename(OUT_STATS)}")

print("\nstep2b complete")
print(f"  Accepted      : {int(accepted.sum())}")
print(f"  Median offset : {np.nanmedian(sep_full[accepted]):.2f} arcsec")
