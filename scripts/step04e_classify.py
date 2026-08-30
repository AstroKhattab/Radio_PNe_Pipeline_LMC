"""
step04e_classify.py
===================
Step 4e - Putting the criteria together.

Three independent criteria have been applied to every radio detection:

    Step 4b   thermal spectral index
    Step 4c   mid-infrared to radio ratio in the PN range
    Step 4d   flux below the physical ceiling for an LMC PN

Each returns one of three answers for each source: pass, fail, or nothing to
say.  "Nothing to say" is not a failure, and this is the point on which the
earlier version of this classification went wrong.  A PN with no IRAC
photometry and a spectrum too faint to fit is not a weaker candidate than one
with three measurements - it is simply a source we know less about.  Scoring
out of a fixed total of three punishes the first kind of source for the state
of the ancillary data rather than for anything about the object.

So each source is scored out of the criteria that could actually be evaluated
for it, and both numbers are carried: how many it passed, and how many were
measurable at all.

    confirmed      every measurable criterion passed, at least two measurable
    probable       most measurable criteria passed
    possible       at least one passed, but not most
    rejected       every measurable criterion failed, at least one measurable
    unconstrained  nothing measurable

The optical classification from Reid & Parker and from HASH does not enter
this at any point.  That is deliberate: the whole purpose is to have a radio
verdict that can be compared against the optical one, and a classification
that used the optical class as an input could not be compared against it.
Step 4f does that comparison.

The confirmed sample is the strong sample used as the second luminosity
function in Step 5.

Outputs
-------
    03_Outputs/step04e_classified.vot
    03_Outputs/step04e_class_summary.csv
    04_Figures/step04e_classification.pdf

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

IN_FILE = os.path.join(OUTS, "step04d_flux_ceiling.vot")
OUT_TABLE = os.path.join(OUTS, "step04e_classified.vot")
OUT_SUMMARY = os.path.join(OUTS, "step04e_class_summary.csv")
OUT_FIG = os.path.join(FIGS, "step04e_classification.pdf")

CLASS_ORDER = ["confirmed", "probable", "possible", "unconstrained", "rejected"]
CLASS_COLOUR = {"confirmed": "#22539b", "probable": "#4c9bd6",
                "possible": "#e8a33d", "unconstrained": "0.72",
                "rejected": "#c44e52"}

print("=" * 60)
print("  Step 4e  -  radio classification")
print("=" * 60)

det = Table.read(IN_FILE, format="votable")
n = len(det)
print(f"\nradio detections : {n}")

sp = np.asarray(det["sp_class"])
mir = np.asarray(det["mir_class"])
ceil = np.asarray(det["ceiling_class"])

# pass = 1, fail = 0, not measurable = -1
c_alpha = np.full(n, -1, dtype=np.int16)
c_alpha[sp == "thermal"] = 1
c_alpha[sp == "steep"] = 0
# "uncertain" spans the gap between the thermal and synchrotron populations
# and does not decide anything either way, so it stays unmeasured.

c_mir = np.full(n, -1, dtype=np.int16)
c_mir[mir == "PN_like"] = 1
c_mir[(mir == "radio_excess") | (mir == "HII_like")] = 0
# "intermediate" is the same kind of in-between value and is treated the same.

c_ceil = np.full(n, -1, dtype=np.int16)
c_ceil[ceil == "below_ceiling"] = 1
c_ceil[ceil == "above_ceiling"] = 0

crit = np.vstack([c_alpha, c_mir, c_ceil])
measurable = (crit >= 0).sum(axis=0)
passed = (crit == 1).sum(axis=0)
frac = np.divide(passed, measurable, out=np.full(n, np.nan, dtype=float),
                 where=measurable > 0)

verdict = np.full(n, "unconstrained", dtype="U14")
has = measurable > 0
verdict[has & (frac > 0.5)] = "probable"
verdict[has & (frac > 0) & (frac <= 0.5)] = "possible"
verdict[has & (frac == 0)] = "rejected"
verdict[(measurable >= 2) & (frac == 1.0)] = "confirmed"

det["crit_alpha"] = Column(c_alpha)
det["crit_mir"] = Column(c_mir)
det["crit_ceiling"] = Column(c_ceil)
det["n_criteria_measurable"] = Column(measurable.astype(np.int16))
det["n_criteria_passed"] = Column(passed.astype(np.int16))
det["criteria_fraction"] = Column(frac.astype("f4"))
det["radio_class"] = Column(verdict)
det["strong_sample"] = Column((verdict == "confirmed").astype(np.int16))
det.write(OUT_TABLE, format="votable", overwrite=True)

print("\ncriterion by criterion:")
print(f"    {'':<22}{'pass':>7}{'fail':>7}{'n/a':>7}")
for lab, c in (("thermal spectrum", c_alpha),
               ("MIR/radio ratio", c_mir),
               ("below flux ceiling", c_ceil)):
    print(f"    {lab:<22}{int((c == 1).sum()):>7}{int((c == 0).sum()):>7}"
          f"{int((c == -1).sum()):>7}")

rows = [("radio detections", n)]
for v in CLASS_ORDER:
    rows.append((v, int((verdict == v).sum())))
rows.append(("strong sample (confirmed)", int((verdict == "confirmed").sum())))
Table(rows=rows, names=("class", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                             overwrite=True)
print("\n" + "-" * 44)
for label, v in rows:
    print(f"  {label:<34}{v:>6}")
print("-" * 44)

print("\nhow many criteria were measurable per source:")
for k in range(4):
    print(f"    {k} : {int((measurable == k).sum()):>4}")

# ====================================================================== figure
fig = plt.figure(figsize=(14.6, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.2], wspace=0.3)

ax = fig.add_subplot(gs[0, 0])
vals = [int((verdict == v).sum()) for v in CLASS_ORDER]
bars = ax.bar(range(len(CLASS_ORDER)), vals,
              color=[CLASS_COLOUR[v] for v in CLASS_ORDER], width=0.68)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
            ha="center", fontsize=9)
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("number of PNe")
ax.set_ylim(0, max(vals) * 1.16)
ax.set_title(f"Radio classification of {n} detections", fontsize=10)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# how the verdict depends on how much we could measure
ax2 = fig.add_subplot(gs[0, 1])
xs = np.arange(4)
bottom = np.zeros(4)
for v in CLASS_ORDER:
    h = np.array([int(((verdict == v) & (measurable == k)).sum()) for k in xs])
    ax2.bar(xs, h, bottom=bottom, color=CLASS_COLOUR[v], width=0.66, label=v)
    bottom += h
ax2.set_xticks(xs)
ax2.set_xlabel("number of criteria that could be evaluated")
ax2.set_ylabel("number of PNe")
ax2.set_title("Verdict against how much we could measure", fontsize=10)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

# where the classes sit in the flux distribution
ax3 = fig.add_subplot(gs[0, 2])
flux = np.asarray(det["flux_1295_mJy"], dtype=float)
good = np.isfinite(flux) & (flux > 0)
bins = np.logspace(np.log10(np.nanmin(flux[good])),
                   np.log10(np.nanmax(flux[good]) * 1.1), 30)
bottom = np.zeros(len(bins) - 1)
for v in CLASS_ORDER:
    h, _ = np.histogram(flux[good & (verdict == v)], bins=bins)
    ax3.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
            color=CLASS_COLOUR[v], label=v)
    bottom += h
ax3.set_xscale("log")
ax3.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax3.set_ylabel("number of PNe")
ax3.set_title("Classification against brightness", fontsize=10)
ax3.legend(fontsize=7.5, framealpha=0.9)
ax3.tick_params(labelsize=8)

fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
