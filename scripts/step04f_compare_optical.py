"""
step04f_compare_optical.py
==========================
Step 4f - The radio verdict against the optical classification.

Nothing in Steps 4b to 4e used the optical class, so the radio verdict and the
optical one are independent statements about the same objects and can be laid
side by side.  Two groups are worth arguing about in the paper:

  * optically uncertain, radio confident.  An object HASH calls only a
    probable PN, or Reid & Parker call Possible, which passes every radio
    criterion we can measure.  The radio data promote it.

  * optically certain, radio poor.  An object catalogued as a true PN whose
    radio counterpart fails the criteria - typically a steep spectrum, or a
    flux above what an LMC PN can produce.  Either the radio source is not the
    PN, in which case the match is a chance alignment, or the optical
    classification deserves another look.  Both possibilities are worth
    naming, and this script produces the list to name them from.

Outputs
-------
    03_Outputs/step04f_optical_comparison.vot   the cross-tabulation, row by row
    03_Outputs/step04f_promoted.csv             optically weak, radio strong
    03_Outputs/step04f_contested.csv            optically strong, radio weak
    03_Outputs/step04f_contingency.csv          the table for the paper
    04_Figures/step04f_optical_comparison.pdf   the same, as a figure

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

IN_FILE = os.path.join(OUTS, "step04e_classified.vot")
OUT_TABLE = os.path.join(OUTS, "step04f_optical_comparison.vot")
OUT_PROMOTED = os.path.join(OUTS, "step04f_promoted.csv")
OUT_CONTESTED = os.path.join(OUTS, "step04f_contested.csv")
OUT_CONTINGENCY = os.path.join(OUTS, "step04f_contingency.csv")
OUT_FIG = os.path.join(FIGS, "step04f_optical_comparison.pdf")

RADIO_ORDER = ["confirmed", "probable", "possible", "rejected"]
RADIO_COLOUR = {"confirmed": "#22539b", "probable": "#4c9bd6",
                "possible": "#e8a33d", "rejected": "#c44e52"}
HASH_LABEL = {"T": "true PN (HASH T)", "P": "probable PN (HASH P)"}

print("=" * 60)
print("  Step 4f  -  radio verdict against optical classification")
print("=" * 60)

det = Table.read(IN_FILE, format="votable")
n = len(det)
radio = np.asarray(det["radio_class"])
hashstat = np.asarray(det["PNstat"])
reid = np.asarray(det["reid_class"])
n_meas = np.asarray(det["n_criteria_measurable"], dtype=int)
n_pass = np.asarray(det["n_criteria_passed"], dtype=int)
print(f"\nradio detections : {n}")

# ------------------------------------------------------- contingency, HASH
hash_states = ["T", "P"]
grid = np.zeros((len(hash_states), len(RADIO_ORDER)), dtype=int)
for i, h in enumerate(hash_states):
    for j, r in enumerate(RADIO_ORDER):
        grid[i, j] = int(np.sum((hashstat == h) & (radio == r)))

print("\nHASH status against radio class:")
head = f"    {'':<20}" + "".join(f"{r:>12}" for r in RADIO_ORDER) + f"{'total':>8}"
print(head)
for i, h in enumerate(hash_states):
    print(f"    {HASH_LABEL[h]:<20}" + "".join(f"{v:>12d}" for v in grid[i])
          + f"{grid[i].sum():>8d}")
print(f"    {'total':<20}" + "".join(f"{v:>12d}" for v in grid.sum(axis=0))
      + f"{grid.sum():>8d}")

cont = Table()
cont["optical_status"] = Column([HASH_LABEL[h] for h in hash_states] + ["total"])
for j, r in enumerate(RADIO_ORDER):
    cont[r] = Column(list(grid[:, j]) + [int(grid[:, j].sum())])
cont["total"] = Column([int(g.sum()) for g in grid] + [int(grid.sum())])
cont.write(OUT_CONTINGENCY, format="ascii.csv", overwrite=True)

# ---------------------------------------------- the two interesting groups
promoted = (radio == "confirmed") & ((hashstat == "P") | (reid == "Possible"))
contested = (hashstat == "T") & (n_meas > 0) & (n_pass <= 1) & (n_pass < n_meas)

print(f"\noptically uncertain but radio-confirmed : {int(promoted.sum())}")
print(f"optically true PN but failing the radio criteria : {int(contested.sum())}")

show = ["PN_ID", "hash_name", "PNstat", "reid_class", "flux_1295_mJy", "alpha",
        "sp_class", "mir_ratio", "mir_class", "ceiling_class",
        "n_criteria_passed", "n_criteria_measurable", "radio_class",
        "detection_method"]
show = [c for c in show if c in det.colnames]

det[show][promoted].write(OUT_PROMOTED, format="ascii.csv", overwrite=True)
det[show][contested].write(OUT_CONTESTED, format="ascii.csv", overwrite=True)

if contested.sum():
    print("\ncontested sources (HASH true PN, radio criteria fail):")
    print(f"    {'ID':<11}{'flux':>7}{'alpha':>8}  {'spectrum':<10}"
          f"{'MIR':>7}  {'passed':<8}")
    for k in np.where(contested)[0]:
        a = det["alpha"][k]
        mr = det["mir_ratio"][k]
        print(f"    {str(det['PN_ID'][k]):<11}{det['flux_1295_mJy'][k]:>7.2f}"
              f"{a if np.isfinite(a) else float('nan'):>8.2f}  "
              f"{str(det['sp_class'][k]):<10}"
              f"{mr if np.isfinite(mr) else float('nan'):>7.1f}  "
              f"{n_pass[k]}/{n_meas[k]}")

det["optically_weak_radio_strong"] = Column(promoted.astype(np.int16))
det["optically_strong_radio_weak"] = Column(contested.astype(np.int16))
det.write(OUT_TABLE, format="votable", overwrite=True)

# ====================================================================== figure
fig = plt.figure(figsize=(14.4, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.1], wspace=0.34)

# panel 1 - the contingency table as stacked bars
ax = fig.add_subplot(gs[0, 0])
xs = np.arange(len(hash_states))
bottom = np.zeros(len(hash_states))
for j, r in enumerate(RADIO_ORDER):
    ax.bar(xs, grid[:, j], bottom=bottom, color=RADIO_COLOUR[r], width=0.6,
           label=r)
    for i in range(len(hash_states)):
        if grid[i, j] >= 6:
            ax.text(xs[i], bottom[i] + grid[i, j] / 2, str(grid[i, j]),
                    ha="center", va="center", fontsize=8, color="w")
    bottom += grid[:, j]
ax.set_xticks(xs)
ax.set_xticklabels(["true PN\n(HASH T)", "probable PN\n(HASH P)"], fontsize=8.5)
ax.set_ylabel("number of PNe")
ax.set_title("Radio verdict within each optical class", fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.9)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# panel 2 - the same as a fraction, which is what the argument rests on
ax2 = fig.add_subplot(gs[0, 1])
bottom = np.zeros(len(hash_states))
tot = grid.sum(axis=1).astype(float)
for j, r in enumerate(RADIO_ORDER):
    f = grid[:, j] / np.where(tot > 0, tot, 1) * 100
    ax2.bar(xs, f, bottom=bottom, color=RADIO_COLOUR[r], width=0.6, label=r)
    for i in range(len(hash_states)):
        if f[i] >= 7:
            ax2.text(xs[i], bottom[i] + f[i] / 2, f"{f[i]:.0f}%", ha="center",
                     va="center", fontsize=8, color="w")
    bottom += f
ax2.set_xticks(xs)
ax2.set_xticklabels(["true PN\n(HASH T)", "probable PN\n(HASH P)"], fontsize=8.5)
ax2.set_ylabel("per cent of detections")
ax2.set_ylim(0, 100)
ax2.set_title("The same, as fractions", fontsize=10)
ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# panel 3 - where the disputed objects sit
ax3 = fig.add_subplot(gs[0, 2])
flux = np.asarray(det["flux_1295_mJy"], dtype=float)
alpha = np.asarray(det["alpha"], dtype=float)
rest = ~promoted & ~contested
ax3.scatter(flux[rest], alpha[rest], s=13, c="0.8", edgecolors="none",
            label="other detections")
ax3.scatter(flux[promoted], alpha[promoted], s=42, c="#22539b",
            edgecolors="k", linewidths=0.4, marker="o",
            label=f"optically weak, radio strong ({int(promoted.sum())})")
ax3.scatter(flux[contested], alpha[contested], s=52, c="#c44e52",
            edgecolors="k", linewidths=0.4, marker="D",
            label=f"optically strong, radio weak ({int(contested.sum())})")
ax3.axhline(-0.5, color="0.4", ls="--", lw=0.9)
ax3.axhline(-0.2, color="0.4", ls="--", lw=0.9)
ax3.axvline(2.2, color="#c44e52", lw=1.2)
ax3.set_xscale("log")
ax3.set_ylim(-3.0, 2.5)
ax3.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax3.set_ylabel(r"spectral index $\alpha$")
ax3.set_title("The objects worth arguing about", fontsize=10)
ax3.legend(fontsize=7, framealpha=0.9, loc="lower right")
ax3.tick_params(labelsize=8)

fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_CONTINGENCY)}")
print(f"wrote {os.path.basename(OUT_PROMOTED)}   ({int(promoted.sum())} rows)")
print(f"wrote {os.path.basename(OUT_CONTESTED)}  ({int(contested.sum())} rows)")
print(f"wrote {os.path.basename(OUT_FIG)}")
