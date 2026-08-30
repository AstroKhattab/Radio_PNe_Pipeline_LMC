"""
step04b_build_multisurvey_catalogue.py
======================================
Build a one-row-per-Reid master catalogue containing independent MeerKAT and
ASKAP measurements, then select the union detected by either survey.

This script does not require a source to be detected by both surveys.  Survey
fluxes remain in separate, explicitly named columns.

Inputs
------
    03_Outputs/step01a_reid_master.vot
    03_Outputs/step02a_meerkat_matched.vot
    03_Outputs/step03b_meerkat_visual.vot
    03_Outputs/step04a_meerkat_detected.vot
    03_Outputs/step02b_askap_matched.vot

Outputs
-------
    03_Outputs/step04b_multisurvey_master.vot  -- all 679 Reid rows
    03_Outputs/step04b_multisurvey_detected_union.vot -- MeerKAT OR ASKAP
    03_Outputs/step04b_multisurvey_nondetected.vot -- neither survey
    04_Figures/step04b_multisurvey_summary.pdf

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
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

REID_FILE = os.path.join(OUTS, "step01a_reid_master.vot")
MKT_MATCHED_FILE = os.path.join(OUTS, "step02a_meerkat_matched.vot")
MKT_VISUAL_FILE = os.path.join(OUTS, "step03b_meerkat_visual.vot")
MKT_DETECTED_FILE = os.path.join(OUTS, "step04a_meerkat_detected.vot")
ASKAP_MATCHED_FILE = os.path.join(OUTS, "step02b_askap_matched.vot")
OUT_MASTER = os.path.join(OUTS, "step04b_multisurvey_master.vot")
OUT_DETECTED = os.path.join(OUTS, "step04b_multisurvey_detected_union.vot")
OUT_NONDETECTED = os.path.join(OUTS, "step04b_multisurvey_nondetected.vot")
OUT_FIG = os.path.join(FIGS, "step04b_multisurvey_summary.pdf")

os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

print("step04b: building the multi-survey master catalogue")

reid = Table.read(REID_FILE, format="votable")
mkt_matched = Table.read(MKT_MATCHED_FILE, format="votable")
mkt_visual = Table.read(MKT_VISUAL_FILE, format="votable") if os.path.exists(MKT_VISUAL_FILE) else None
mkt_detected = Table.read(MKT_DETECTED_FILE, format="votable")
askap_matched = Table.read(ASKAP_MATCHED_FILE, format="votable")

print("\n[1] Loaded catalogues")
print(f"    Reid master       : {len(reid)}")
print(f"    MeerKAT catalogue : {len(mkt_matched)}")
print(f"    MeerKAT visual    : {len(mkt_visual) if mkt_visual is not None else 0}")
print(f"    MeerKAT union     : {len(mkt_detected)}")
print(f"    ASKAP accepted    : {len(askap_matched)}")

def row_key(table, index):
    rpid = str(table["RP_ID"][index]).strip() if "RP_ID" in table.colnames else ""
    if rpid:
        return "RP:" + rpid
    return "COORD:{:.8f},{:.8f}".format(float(table["RA"][index]), float(table["Dec"][index]))

def key_map(table):
    return {row_key(table, i): i for i in range(len(table))}

reid_keys = [row_key(reid, i) for i in range(len(reid))]
mkt_det_map = key_map(mkt_detected)
mkt_match_map = key_map(mkt_matched)
mkt_visual_map = key_map(mkt_visual) if mkt_visual is not None else {}
askap_map = key_map(askap_matched)

n = len(reid)
det_meerkat = np.array([key in mkt_det_map for key in reid_keys], dtype=bool)
det_askap = np.array([key in askap_map for key in reid_keys], dtype=bool)
det_union = det_meerkat | det_askap

survey = np.full(n, "none", dtype="U16")
survey[det_meerkat & ~det_askap] = "MeerKAT"
survey[~det_meerkat & det_askap] = "ASKAP"
survey[det_meerkat & det_askap] = "both"

master = reid.copy()
master.add_column(Column(np.asarray(master["reid_class"], dtype=str), name="reid_classification"))
master.add_column(Column(det_meerkat.astype(np.int16), name="det_meerkat"))
master.add_column(Column(det_askap.astype(np.int16), name="det_askap"))
master.add_column(Column(survey, name="detection_surveys"))

def blank_float():
    return np.full(n, np.nan, dtype=float)

def blank_text(width=20):
    return np.full(n, "", dtype=f"U{width}")

# Catalogue fluxes are already in Jy; the visual extractions come in mJy.
mkt_method = blank_text(20)
mkt_ra = blank_float(); mkt_dec = blank_float(); mkt_sep = blank_float()
mkt_peak = blank_float(); mkt_int = blank_float(); mkt_err_int = blank_float()
mkt_rms = blank_float(); mkt_snr = blank_float(); mkt_err_method = blank_text(24)

for i, key in enumerate(reid_keys):
    if key in mkt_match_map:
        j = mkt_match_map[key]
        mkt_method[i] = "catalogue"
        mkt_ra[i] = float(mkt_matched["mkt_ra"][j])
        mkt_dec[i] = float(mkt_matched["mkt_dec"][j])
        mkt_sep[i] = float(mkt_matched["separation_arcsec"][j])
        mkt_peak[i] = float(mkt_matched["mkt_peak_flux_Jy"][j])
        mkt_int[i] = float(mkt_matched["mkt_int_flux_Jy"][j])
        mkt_err_int[i] = float(mkt_matched["mkt_err_int_flux_Jy"][j])
        mkt_rms[i] = float(mkt_matched["mkt_local_rms_Jy"][j])
        mkt_snr[i] = float(mkt_matched["mkt_snr"][j])
        mkt_err_method[i] = "catalogue_fit"
    elif key in mkt_visual_map:
        j = mkt_visual_map[key]
        mkt_method[i] = "visual"
        mkt_ra[i] = float(mkt_visual["mkt_ra"][j])
        mkt_dec[i] = float(mkt_visual["mkt_dec"][j])
        mkt_sep[i] = float(mkt_visual["offset_arcsec"][j])
        mkt_peak[i] = float(mkt_visual["mkt_peak_flux_mJy"][j]) / 1000.0
        mkt_int[i] = float(mkt_visual["mkt_int_flux_mJy"][j]) / 1000.0
        mkt_rms[i] = float(mkt_visual["mkt_local_rms_mJy"][j]) / 1000.0
        mkt_snr[i] = float(mkt_visual["mkt_snr"][j])
        # The visual script does not fit an integrated-flux uncertainty.
        # Preserve a transparent local-rms proxy instead of inventing a fit error.
        mkt_err_int[i] = mkt_rms[i]
        mkt_err_method[i] = "local_rms_proxy"

for column in [
    Column(mkt_method, name="mkt_detection_method"),
    Column(mkt_ra, name="mkt_ra", unit="deg"),
    Column(mkt_dec, name="mkt_dec", unit="deg"),
    Column(mkt_sep.astype("f4"), name="mkt_separation_arcsec", unit="arcsec"),
    Column(mkt_peak.astype("f4"), name="mkt_peak_flux_Jy", unit="Jy"),
    Column(mkt_int.astype("f4"), name="mkt_int_flux_Jy", unit="Jy"),
    Column(mkt_err_int.astype("f4"), name="mkt_err_int_flux_Jy", unit="Jy"),
    Column(mkt_err_method, name="mkt_err_int_method"),
    Column(mkt_rms.astype("f4"), name="mkt_local_rms_Jy", unit="Jy/beam"),
    Column(mkt_snr.astype("f4"), name="mkt_snr"),
]:
    master.add_column(column)

# Copy every prefixed ASKAP field from the accepted-match table.
askap_columns = [name for name in askap_matched.colnames if name.startswith("askap_")]
for name in askap_columns:
    src_col = askap_matched[name]
    if src_col.dtype.kind in "fiu":
        arr = blank_float()
        for i, key in enumerate(reid_keys):
            if key in askap_map:
                arr[i] = float(src_col[askap_map[key]])
        new_col = Column(arr.astype("f4"), name=name, unit=getattr(src_col, "unit", None))
    else:
        arr = blank_text(max(16, int(src_col.dtype.itemsize / 4) if src_col.dtype.kind == "U" else 32))
        for i, key in enumerate(reid_keys):
            if key in askap_map:
                arr[i] = str(src_col[askap_map[key]])
        new_col = Column(arr, name=name)
    master.add_column(new_col)

master.write(OUT_MASTER, format="votable", overwrite=True)
master[det_union].write(OUT_DETECTED, format="votable", overwrite=True)
master[~det_union].write(OUT_NONDETECTED, format="votable", overwrite=True)

n_both = int(np.sum(det_meerkat & det_askap))
n_mkt_only = int(np.sum(det_meerkat & ~det_askap))
n_ask_only = int(np.sum(~det_meerkat & det_askap))

print("\n[2] Multisurvey union")
print(f"    Both surveys : {n_both}")
print(f"    MeerKAT only : {n_mkt_only}")
print(f"    ASKAP only   : {n_ask_only}")
print(f"    Union        : {det_union.sum()}")
print(f"    Neither      : {(~det_union).sum()}")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
labels = ["Both", "MeerKAT only", "ASKAP only", "Neither"]
counts = [n_both, n_mkt_only, n_ask_only, int((~det_union).sum())]
colors = ["#6A3D9A", "#2166AC", "#E67E22", "#BDBDBD"]
bars = ax.bar(labels, counts, color=colors, edgecolor="white")
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.015,
            str(count), ha="center", va="bottom")
ax.set_ylabel("Number of Reid sources")
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, max(counts) * 1.12)
plt.tight_layout()
plt.savefig(OUT_FIG, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()

print("\n[3] Outputs")
print(f"    {OUT_MASTER}")
print(f"    {OUT_DETECTED}")
print(f"    {OUT_NONDETECTED}")
print(f"    {OUT_FIG}")
print("\nstep04b complete")
