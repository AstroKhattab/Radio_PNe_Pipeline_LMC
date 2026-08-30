"""
step4b_mir_radio_ratio.py
=========================
Step 4b — MIR/radio diagnostic at the two adopted radio frequencies.

Cohen et al. (2011, MNRAS 413, 514) did not define the diagnostic at
1.2 GHz.  They combined MGPS-2 measurements at 843 MHz and NVSS
measurements at 1.4 GHz and treated the two surveys as a common frequency
near 1 GHz.  Their median 8-micron/radio ratio for PNe is 4.7 +/- 1.1.

This step therefore reports the ratio separately using:
    1. ASKAP integrated flux at 888 MHz;
    2. MeerKAT broadband integrated flux at 1295 MHz.

Only these two are used.  An earlier version also computed the ratio from the
MeerKAT 996.646 MHz sub-band, on the grounds that it sits closest to the
~1 GHz convention of Cohen et al., and preferred it over the other two.  That
has been dropped: a single sub-band carries roughly a twelfth of the bandwidth
and therefore several times the noise of the broadband measurement, so it made
the adopted ratio noisier than it needed to be for no gain in frequency
fidelity.  The adopted diagnostic now uses ASKAP 888 MHz where available and
the MeerKAT 1295 MHz broadband flux otherwise.  No radio
frequency scaling is hidden in these columns: every ratio uses the measured
flux at the frequency stated in its name.

Inputs
------
    03_Outputs/step4a_spectral_index.vot

Outputs
-------
    03_Outputs/step4b_mir_radio_ratio.vot
    05_Figures/step4b_mir_radio_ratio.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import warnings

import matplotlib
import numpy as np
from astropy.table import Column, Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style

warnings.filterwarnings("ignore")
apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "05_Figures")

INPUT_FILE = os.path.join(OUTS, "step4a_spectral_index.vot")
OUT_VOT = os.path.join(OUTS, "step4b_mir_radio_ratio.vot")
FIG_RATIO = os.path.join(FIGS, "step4b_mir_radio_ratio.pdf")

ZP_8UM_JY = 64.13
COHEN_PN_MEDIAN = 4.7
PN_SCREEN_LOW = 0.5
PN_SCREEN_HIGH = 10.0
HII_LIKE_LOW = 20.0


def numeric(table, column):
    """Return a float array, converting masks and invalid values to NaN."""
    values = np.ma.asarray(table[column], dtype=float)
    return np.asarray(values.filled(np.nan), dtype=float)


def ratio_with_error(ir_flux, ir_error, radio_flux, radio_error):
    ratio = np.full(len(ir_flux), np.nan)
    error = np.full(len(ir_flux), np.nan)
    valid = (
        np.isfinite(ir_flux) & (ir_flux > 0) &
        np.isfinite(radio_flux) & (radio_flux > 0)
    )
    ratio[valid] = ir_flux[valid] / radio_flux[valid]
    valid_error = (
        valid & np.isfinite(ir_error) & (ir_error >= 0) &
        np.isfinite(radio_error) & (radio_error >= 0)
    )
    error[valid_error] = ratio[valid_error] * np.sqrt(
        (ir_error[valid_error] / ir_flux[valid_error]) ** 2 +
        (radio_error[valid_error] / radio_flux[valid_error]) ** 2
    )
    return ratio, error


def diagnostic_class(ratios):
    labels = np.full(len(ratios), "unavailable", dtype="U20")
    valid = np.isfinite(ratios)
    labels[valid & (ratios < PN_SCREEN_LOW)] = "radio_excess"
    labels[valid & (ratios >= PN_SCREEN_LOW) & (ratios <= PN_SCREEN_HIGH)] = "PN_like"
    labels[valid & (ratios > PN_SCREEN_HIGH) & (ratios < HII_LIKE_LOW)] = "intermediate_high"
    labels[valid & (ratios >= HII_LIKE_LOW)] = "HII_like"
    return labels


os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

print("step4b: mid-infrared to radio ratio diagnostic")

print("\n[1] Loading the combined spectral catalogue...")
detected = Table.read(INPUT_FILE, format="votable")
print(f"    Sources: {len(detected)}")

print("\n[2] Converting SAGE [8.0] magnitude to flux density...")
mag_8 = numeric(detected, "__8.0_")
mag_8_error = numeric(detected, "e__8.0_")
flux_8um_jy = np.full(len(detected), np.nan)
flux_8um_error_jy = np.full(len(detected), np.nan)
has_8 = np.isfinite(mag_8)
flux_8um_jy[has_8] = ZP_8UM_JY * 10 ** (-mag_8[has_8] / 2.5)
has_8_error = has_8 & np.isfinite(mag_8_error) & (mag_8_error >= 0)
flux_8um_error_jy[has_8_error] = (
    flux_8um_jy[has_8_error] * np.log(10.0) / 2.5 * mag_8_error[has_8_error]
)
print(f"    Sources with [8.0] photometry: {int(has_8.sum())}")

print("\n[3] Computing the three requested MIR/radio ratios...")
askap_flux_jy = numeric(detected, "askap_int_flux_mJy") / 1000.0
askap_error_jy = numeric(detected, "askap_err_int_flux_total_mJy") / 1000.0

mkt_central_flux_jy = numeric(detected, "mkt_int_flux_Jy")
mkt_central_error_jy = numeric(detected, "mkt_err_int_flux_Jy")

ratio_askap, ratio_askap_error = ratio_with_error(
    flux_8um_jy, flux_8um_error_jy, askap_flux_jy, askap_error_jy
)
ratio_central, ratio_central_error = ratio_with_error(
    flux_8um_jy, flux_8um_error_jy, mkt_central_flux_jy, mkt_central_error_jy
)

# Best available measurement closest to Cohen et al.'s common ~1-GHz
# convention: ASKAP 888 MHz first, then the MeerKAT 1295 MHz broadband flux.
n_sources = len(detected)
ratio_near1 = np.full(n_sources, np.nan)
ratio_near1_error = np.full(n_sources, np.nan)
radio_frequency_mhz = np.full(n_sources, np.nan)
radio_survey = np.full(n_sources, "none", dtype="U24")

for ratio, error, frequency, label in [
    (ratio_askap, ratio_askap_error, 888.0, "ASKAP"),
    (ratio_central, ratio_central_error, 1295.0, "MeerKAT_broadband"),
]:
    use = ~np.isfinite(ratio_near1) & np.isfinite(ratio)
    ratio_near1[use] = ratio[use]
    ratio_near1_error[use] = error[use]
    radio_frequency_mhz[use] = frequency
    radio_survey[use] = label

near1_class = diagnostic_class(ratio_near1)

for label, values in [
    ("ASKAP 888 MHz", ratio_askap),
    ("MeerKAT 1295 MHz", ratio_central),
    ("Adopted near 1 GHz", ratio_near1),
]:
    valid = np.isfinite(values)
    median = np.nanmedian(values) if np.any(valid) else np.nan
    print(f"    {label:<23}: N={int(valid.sum()):3d}, median={median:7.2f}")

print("\n[4] Writing Step 6a catalogue...")
new_columns = {
    "flux_8um_Jy": flux_8um_jy.astype("f4"),
    "flux_8um_err_Jy": flux_8um_error_jy.astype("f4"),
    "cohen_ratio_askap_888": ratio_askap.astype("f4"),
    "cohen_ratio_askap_888_err": ratio_askap_error.astype("f4"),
    "cohen_ratio_mkt_1295": ratio_central.astype("f4"),
    "cohen_ratio_mkt_1295_err": ratio_central_error.astype("f4"),
    "cohen_ratio_near1GHz": ratio_near1.astype("f4"),
    "cohen_ratio_near1GHz_err": ratio_near1_error.astype("f4"),
    "cohen_radio_frequency_MHz": radio_frequency_mhz.astype("f4"),
    "cohen_radio_measurement": radio_survey,
    "cohen_diagnostic": near1_class,
}
for name, values in new_columns.items():
    if name in detected.colnames:
        detected.remove_column(name)
    detected.add_column(Column(values, name=name))

# Compatibility aliases for the existing downstream PNLF script.  They refer
# explicitly to the MeerKAT broadband measurement, never to ASKAP-only flux.
for name in ["radio_flux_Jy", "cohen_ratio"]:
    if name in detected.colnames:
        detected.remove_column(name)
detected.add_column(Column(mkt_central_flux_jy.astype("f4"), name="radio_flux_Jy"))
detected.add_column(Column(ratio_near1.astype("f4"), name="cohen_ratio"))

detected.write(OUT_VOT, format="votable", overwrite=True)
print(f"    {OUT_VOT}")

print("\n[5] Generating the two-frequency comparison figure...")
# The 996.646 MHz sub-band panel has been removed; see the module docstring.
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
panels = [
    (ratio_askap, "ASKAP 888 MHz", "#E67E22"),
    (ratio_central, "MeerKAT 1295 MHz", "#5E3C99"),
]
bins = np.arange(0, 30.5, 1.0)
for ax, (values, title, color) in zip(axes, panels):
    valid = values[np.isfinite(values)]
    shown = valid[valid <= 30]
    median = np.nanmedian(valid) if len(valid) else np.nan
    ax.axvspan(PN_SCREEN_LOW, PN_SCREEN_HIGH, color="#70A1D7", alpha=0.15,
               label="Broad PN-like band (0.5–10)")
    ax.hist(shown, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(COHEN_PN_MEDIAN, color="#222222", linestyle="--", linewidth=1.5,
               label="Cohen PN median (4.7)")
    if np.isfinite(median):
        ax.axvline(median, color="#C23B33", linewidth=1.5,
                   label=f"This sample median ({median:.1f})")
    ax.set_title(title)
    ax.set_xlabel(r"$F_{8\,\mu m}/S_{radio}$")
    ax.text(0.97, 0.95, f"N={len(valid)}\n>30: {int(np.sum(valid > 30))}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Number of sources")
# axes[0].legend(fontsize=8, loc="upper right")
# changed 2026-08-30: the upper-right legend sat on top of the N annotation at (0.97, 0.95)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Cohen MIR/radio diagnostic at ASKAP 888 MHz and MeerKAT 1295 MHz")
plt.tight_layout()
plt.savefig(FIG_RATIO, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    {FIG_RATIO}")

print("\nstep4b complete")
print("    Cohen et al. reference: MGPS-2 843 MHz + NVSS 1.4 GHz, treated as ~1 GHz")
print("    There is no exact 1.2-GHz Cohen ratio in that paper.")
