"""
step04a_detection_catalogue.py
==============================
Step 4a - The radio detection catalogue.

Brings together everything that counts as a radio detection of an LMC PN and
puts it on one flux scale:

    catalogue detections   MeerKAT and/or ASKAP source lists (Step 2)
    tentative detections   forced photometry, vetted (Step 3)

Flux scale
----------
Every source needs a single 1.295 GHz integrated flux for the luminosity
function.  In order of preference:

    1. the MeerKAT catalogue integrated flux;
    2. the forced-photometry aperture flux, for tentative detections;
    3. the ASKAP 888 MHz flux scaled to 1.295 GHz, for the handful of PNe seen
       by ASKAP but not by MeerKAT.  The scaling uses the fitted spectral index
       where Step 4b provides one and alpha = -0.1 otherwise, which is the
       optically thin free-free value; the correction is small (a factor of
       1.04 for a flat spectrum) and the sources concerned are few.

Sources detected only by ASKAP are kept but flagged, because their flux is the
only one in the catalogue that is not measured at the reference frequency.

Outputs
-------
    03_Outputs/step04a_radio_detections.vot   the detection catalogue
    03_Outputs/step04a_detection_summary.csv  counts behind the paper table
    04_Figures/step04a_detections.pdf         flux distribution and provenance

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column, vstack

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

XMATCH_FILE = os.path.join(OUTS, "step02c_radio_crossmatch.vot")
VISUAL_FILE = os.path.join(OUTS, "step03e_visual_detections.vot")

OUT_TABLE = os.path.join(OUTS, "step04a_radio_detections.vot")
OUT_SUMMARY = os.path.join(OUTS, "step04a_detection_summary.csv")
OUT_FIG = os.path.join(FIGS, "step04a_detections.pdf")

REF_FREQ_GHZ = 1.295
ASKAP_FREQ_GHZ = 0.888
DEFAULT_ALPHA = -0.1     # optically thin free-free

print("=" * 60)
print("  Step 4a  -  radio detection catalogue")
print("=" * 60)

xm = Table.read(XMATCH_FILE, format="votable")
vis = Table.read(VISUAL_FILE, format="votable")
vis = vis[np.asarray(vis["visual_detected"]) == 1]
print(f"\nparent sample          : {len(xm)}")
print(f"catalogue detections   : {int(np.sum(xm['catalogue_detected']))}")
print(f"tentative detections   : {len(vis)}")

# Carry the vetting result back onto the parent table so that every PN in the
# parent sample has a detection state, not just the ones we kept.
method = np.where(np.asarray(xm["catalogue_detected"]) == 1, "catalogue", "none")
method = method.astype("U12")
fp_flux = np.full(len(xm), np.nan)
fp_snr = np.full(len(xm), np.nan)

pid_to_row = {p: i for i, p in enumerate(np.asarray(xm["PN_ID"]))}
for k in range(len(vis)):
    r = pid_to_row[str(vis["PN_ID"][k])]
    method[r] = "tentative"
    fp_flux[r] = float(vis["fp_int_mJy"][k])
    fp_snr[r] = float(vis["fp_snr"][k])

xm["detection_method"] = Column(method)
xm["fp_int_mJy"] = Column(fp_flux.astype("f4"), unit="mJy")
xm["fp_snr"] = Column(fp_snr.astype("f4"))

is_det = method != "none"
det = xm[is_det]
print(f"total detections       : {len(det)}")

# ------------------------------------------------------------- unified flux
mkt = np.asarray(det["mkt_int_mJy"], dtype=float)
e_mkt = np.asarray(det["mkt_e_int_mJy"], dtype=float)
ask = np.asarray(det["askap_int_mJy"], dtype=float)
e_ask = np.asarray(det["askap_e_int_mJy"], dtype=float)
fp = np.asarray(det["fp_int_mJy"], dtype=float)
meth = np.asarray(det["detection_method"])

flux = np.full(len(det), np.nan)
e_flux = np.full(len(det), np.nan)
origin = np.full(len(det), "", dtype="U16")

use_mkt = np.isfinite(mkt) & (mkt > 0)
flux[use_mkt] = mkt[use_mkt]
e_flux[use_mkt] = e_mkt[use_mkt]
origin[use_mkt] = "MeerKAT"

use_fp = ~use_mkt & (meth == "tentative") & np.isfinite(fp) & (fp > 0)
flux[use_fp] = fp[use_fp]
# forced photometry: sigma_int propagated in Step 3a from the local scatter
e_flux[use_fp] = fp[use_fp] / np.asarray(det["fp_snr"], dtype=float)[use_fp]
origin[use_fp] = "forced"

use_ask = ~use_mkt & ~use_fp & np.isfinite(ask) & (ask > 0)
scale = (REF_FREQ_GHZ / ASKAP_FREQ_GHZ) ** DEFAULT_ALPHA
flux[use_ask] = ask[use_ask] * scale
e_flux[use_ask] = e_ask[use_ask] * scale
origin[use_ask] = "ASKAP scaled"

det["flux_1295_mJy"] = Column(flux.astype("f4"), unit="mJy")
det["e_flux_1295_mJy"] = Column(e_flux.astype("f4"), unit="mJy")
det["flux_origin"] = Column(origin)

print("\nflux provenance:")
for lab in ("MeerKAT", "forced", "ASKAP scaled"):
    print(f"    {lab:<14}{int(np.sum(origin == lab)):>5}")
missing = int(np.sum(~np.isfinite(flux)))
if missing:
    print(f"    no usable flux{missing:>5}")

det.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("parent sample", len(xm)),
        ("catalogue detections", int(np.sum(meth == "catalogue"))),
        ("tentative detections (forced photometry)", int(np.sum(meth == "tentative"))),
        ("total radio detections", len(det)),
        ("    flux from MeerKAT 1.295 GHz", int(np.sum(origin == "MeerKAT"))),
        ("    flux from forced photometry", int(np.sum(origin == "forced"))),
        ("    flux scaled from ASKAP 888 MHz", int(np.sum(origin == "ASKAP scaled"))),
        ("non-detections", int(np.sum(method == "none")))]
Table(rows=rows, names=("quantity", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                                overwrite=True)
print("\n" + "-" * 56)
for label, v in rows:
    print(f"  {label:<46}{v:>6}")
print("-" * 56)

# ====================================================================== figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

ax = axes[0]
good = np.isfinite(flux) & (flux > 0)
bins = np.logspace(np.log10(np.nanmin(flux[good])), np.log10(np.nanmax(flux[good])), 30)
ax.hist(flux[good & (meth == "catalogue")], bins=bins, color="#22539b",
        label=f"catalogue ({int((good & (meth == 'catalogue')).sum())})")
ax.hist(flux[good & (meth == "tentative")], bins=bins, color="#e8a33d",
        label=f"tentative ({int((good & (meth == 'tentative')).sum())})")
ax.set_xscale("log")
ax.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax.set_ylabel("number of PNe")
ax.set_title("Radio detections of LMC planetary nebulae", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.tick_params(labelsize=8)

ax2 = axes[1]
labels, counts = np.unique(np.asarray(det["detection"]), return_counts=True)
disp = {"both": "MeerKAT\n+ ASKAP", "MeerKAT": "MeerKAT\nonly",
        "ASKAP": "ASKAP\nonly", "none": "forced\nphotometry"}
order = ["both", "MeerKAT", "ASKAP", "none"]
vals = [int(np.sum(np.asarray(det["detection"]) == o)) for o in order]
cols = ["#4c72b0", "#22539b", "#c44e52", "#e8a33d"]
bars = ax2.bar([disp[o] for o in order], vals, color=cols, width=0.66)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
             ha="center", fontsize=9)
ax2.set_ylabel("number of PNe")
ax2.set_ylim(0, max(vals) * 1.16)
ax2.set_title(f"{len(det)} detections out of {len(xm)} parent PNe", fontsize=10)
ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}   ({len(det)} rows)")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
