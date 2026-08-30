"""
step5d_criteria_tally.py
========================
Step 5e - Score every detection on the three criteria alone.

The evidence grade produced in Step 5a uses the Reid & Parker optical class as
part of its rule set: a source cannot reach the highest grade unless it is
already classified Known or True optically.  That is defensible as a summary of
everything we know, but it makes the grade unusable for the one question we
most want to ask, which is whether the radio data agree with the optical
classification.  A grade that takes the optical class as an input cannot be
compared against it.

This step therefore scores each detection on the three radio criteria and
nothing else:

    4a  thermal spectral index      pass: thermal      fail: steep
    4b  mid-infrared to radio ratio pass: PN-like      fail: radio excess or
                                                             H II-like
    4c  radio flux ceiling          pass: below 2.2 mJy  fail: above

Each returns pass, fail, or nothing to say.  "Nothing to say" is not a failure.
A nebula with no 8 micron photometry and a spectrum too faint to fit is not a
weaker candidate than one with three measurements; it is an object we know less
about.  Scoring out of a fixed total of three would penalise it for the state
of the ancillary data rather than for anything about the source, so each object
is scored out of the criteria that could actually be evaluated for it, and both
numbers are carried.

    Radio_Rejected   the object fails a hard physical test: its flux exceeds
                     the 2.2 mJy ceiling, or its radio position lies more than
                     4.5 arcsec from the optical one.  These are not weak
                     candidates, they are objects the radio data rule out.
    Radio_High       every measurable criterion passed, at least two measurable
    Radio_Possible   more than half of the measurable criteria passed
    Radio_Weak       everything else: half or fewer passed, or nothing
                     measurable at all

The class names deliberately avoid the words the optical catalogues use.
HASH and Reid & Parker both classify objects as "True", "Probable", "Likely"
and "Possible", and reusing those words for a radio verdict makes any
comparison between the two impossible to read.

Nothing here uses the optical classification, so the cross-tabulations at the
end are a genuine comparison of two independent statements about the same
objects.

Outputs
-------
    03_Outputs/step5d_criteria_tally.vot
    03_Outputs/step5d_tally_summary.csv
    03_Outputs/step5d_tally_vs_hash.csv
    03_Outputs/step5d_tally_vs_reid.csv
    03_Outputs/step5d_promoted.csv     optically weak, radio strong
    03_Outputs/step5d_contested.csv    optically true, radio rejected
    05_Figures/step5d_criteria_tally.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

warnings.filterwarnings("ignore")

BASE = os.environ.get("PNBASE", os.path.expanduser("~/Desktop/Research/PN LMC Paper"))
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "05_Figures")

IN_FILE = os.path.join(OUTS, "step5b_spectral_index_recut.vot")
OUT_VOT = os.path.join(OUTS, "step5d_criteria_tally.vot")
OUT_SUMMARY = os.path.join(OUTS, "step5d_tally_summary.csv")
OUT_HASH = os.path.join(OUTS, "step5d_tally_vs_hash.csv")
OUT_REID = os.path.join(OUTS, "step5d_tally_vs_reid.csv")
OUT_PROM = os.path.join(OUTS, "step5d_promoted.csv")
OUT_CONT = os.path.join(OUTS, "step5d_contested.csv")
OUT_FIG = os.path.join(FIGS, "step5d_criteria_tally.pdf")

ORDER = ["Radio_High", "Radio_Possible", "Radio_Weak", "Radio_Rejected"]
COLOUR = {"Radio_High": "#1f4e9c", "Radio_Possible": "#4c9bd6",
          "Radio_Weak": "#e8a33d", "Radio_Rejected": "#c0392b"}
MATCH_RAD = 4.5   # arcsec; the acceptance radius used in Step 2

MIR_LOW, MIR_HIGH = 0.5, 10.0     # Cohen et al. 2011 PN-like band
MIR_HII = 20.0
CEILING_MJY = 2.2                  # Filipovic et al. 2009


def text(table, name):
    return np.array([str(x).strip() for x in table[name]])


print("=" * 62)
print("  Step 5e  -  criteria tally, independent of the optical class")
print("=" * 62)

t = Table.read(IN_FILE, format="votable")
n = len(t)
print(f"\nradio detections : {n}")

# ------------------------------------------------------- criterion 1: alpha
sp = text(t, "sp_class_recut")
c_alpha = np.full(n, -1, dtype=np.int16)
c_alpha[sp == "thermal"] = 1
c_alpha[sp == "steep"] = 0
# 'uncertain' spans the gap between the thermal and synchrotron populations
# and decides nothing either way, so it counts as not measurable.

# ------------------------------------------------- criterion 2: MIR / radio
ratio = np.asarray(t["cohen_ratio_near1GHz"], dtype=float)
c_mir = np.full(n, -1, dtype=np.int16)
ok = np.isfinite(ratio)
c_mir[ok & (ratio >= MIR_LOW) & (ratio <= MIR_HIGH)] = 1
c_mir[ok & ((ratio < MIR_LOW) | (ratio >= MIR_HII))] = 0
# the intermediate band, 10 to 20, is the same kind of in-between value

# --------------------------------------------------- criterion 3: flux ceiling
flux_mjy = np.asarray(t["mkt_int_flux_Jy"], dtype=float) * 1e3
c_ceil = np.full(n, -1, dtype=np.int16)
meas = np.isfinite(flux_mjy) & (flux_mjy > 0)
c_ceil[meas & (flux_mjy <= CEILING_MJY)] = 1
c_ceil[meas & (flux_mjy > CEILING_MJY)] = 0

crit = np.vstack([c_alpha, c_mir, c_ceil])
measurable = (crit >= 0).sum(axis=0)
passed = (crit == 1).sum(axis=0)
frac = np.divide(passed, measurable, out=np.full(n, np.nan), where=measurable > 0)

# Hard rejections first.  These are physical, not statistical: a source above
# the ceiling is too luminous to be a PN at this distance, and one further than
# the acceptance radius from the optical position is not the same object.  In
# practice the positional test never fires, because Step 2 only accepts matches
# within MATCH_RAD in the first place; it is written down anyway so that the
# rule is complete and so that it would catch such a source if the matching
# radius were ever widened.
sep = np.asarray(t["mkt_separation_arcsec"], dtype=float) \
    if "mkt_separation_arcsec" in t.colnames else np.full(n, np.nan)
bad_position = np.isfinite(sep) & (sep > MATCH_RAD)
rejected = (c_ceil == 0) | bad_position
print(f"\nhard rejections: {int(rejected.sum())} "
      f"(above the ceiling: {int((c_ceil == 0).sum())}, "
      f"offset > {MATCH_RAD} arcsec: {int(bad_position.sum())})")

verdict = np.full(n, "Radio_Weak", dtype="U16")
has = measurable > 0
verdict[has & (frac > 0.5)] = "Radio_Possible"
verdict[(measurable >= 2) & (frac == 1.0)] = "Radio_High"
verdict[rejected] = "Radio_Rejected"

print("\ncriterion by criterion:")
print(f"    {'':<26}{'pass':>7}{'fail':>7}{'n/a':>7}")
for lab, c in (("thermal spectrum (4a)", c_alpha),
               ("MIR/radio ratio (4b)", c_mir),
               ("below flux ceiling (4c)", c_ceil)):
    print(f"    {lab:<26}{int((c == 1).sum()):>7}"
          f"{int((c == 0).sum()):>7}{int((c == -1).sum()):>7}")

print("\nhow many criteria could be evaluated:")
for k in range(4):
    print(f"    {k} of 3 : {int((measurable == k).sum()):>4}")

print("\nscore out of what was measurable:")
for m in range(1, 4):
    for p in range(m + 1):
        cnt = int(((measurable == m) & (passed == p)).sum())
        if cnt:
            print(f"    {p} of {m} passed : {cnt:>4}")

t["crit_thermal"] = Column(c_alpha)
t["crit_mir"] = Column(c_mir)
t["crit_ceiling"] = Column(c_ceil)
t["n_criteria_measurable"] = Column(measurable.astype(np.int16))
t["n_criteria_passed"] = Column(passed.astype(np.int16))
t["radio_only_class"] = Column(verdict)

rows = [("radio detections", n)]
for v in ORDER:
    rows.append((v, int((verdict == v).sum())))
Table(rows=rows, names=("class", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                             overwrite=True)
print("\n" + "-" * 44)
for lab, v in rows:
    print(f"  {lab:<32}{v:>6}")
print("-" * 44)


def contingency(labels, states, path, title, display=None):
    g = np.zeros((len(states), len(ORDER)), dtype=int)
    for i, s in enumerate(states):
        for j, v in enumerate(ORDER):
            g[i, j] = int(np.sum((labels == s) & (verdict == v)))
    tab = Table()
    tab["optical_class"] = Column([(display or {}).get(s, s) for s in states] + ["total"])
    for j, v in enumerate(ORDER):
        tab[v] = Column(list(g[:, j]) + [int(g[:, j].sum())])
    tab["total"] = Column([int(r.sum()) for r in g] + [int(g.sum())])
    tab.write(path, format="ascii.csv", overwrite=True)
    print(f"\n{title}")
    print("    " + f"{'':<26}" + "".join(f"{v:>16}" for v in ORDER) + f"{'total':>8}")
    for i, s in enumerate(states):
        lab = (display or {}).get(s, s)
        print("    " + f"{lab:<26}" + "".join(f"{x:>16d}" for x in g[i])
              + f"{g[i].sum():>8d}")
    print("    " + f"{'total':<26}" + "".join(f"{x:>16d}" for x in g.sum(axis=0))
          + f"{g.sum():>8d}")
    return g


hs = text(t, "hash_status_label")
rc = text(t, "reid_classification")
g_hash = contingency(hs, ["Listed as True in HASH", "Listed as Possible in HASH"],
                     OUT_HASH, "Criteria tally against HASH classification:",
                     {"Listed as True in HASH": "true PN (HASH T)",
                      "Listed as Possible in HASH": "probable PN (HASH P)"})
g_reid = contingency(rc, ["True", "Known", "Likely", "Possible"], OUT_REID,
                     "Criteria tally against Reid & Parker class:")

promoted = (verdict == "Radio_High") & ((hs == "Listed as Possible in HASH") | (rc == "Possible"))
contested = (hs == "Listed as True in HASH") & (verdict == "Radio_Rejected")
print(f"\noptically weak but Radio_High       : {int(promoted.sum())}")
print(f"HASH true PN but Radio_Rejected     : {int(contested.sum())}")

show = [c for c in ["RP_ID", "Name", "reid_classification", "hash_status_label",
                    "mkt_int_flux_Jy", "alpha_combined", "sp_class_recut",
                    "cohen_ratio_near1GHz", "n_criteria_passed",
                    "n_criteria_measurable", "radio_only_class"]
        if c in t.colnames]
t[show][promoted].write(OUT_PROM, format="ascii.csv", overwrite=True)
t[show][contested].write(OUT_CONT, format="ascii.csv", overwrite=True)
t.write(OUT_VOT, format="votable", overwrite=True)

# ====================================================================== figure
fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))

ax = axes[0]
vals = [int((verdict == v).sum()) for v in ORDER]
bars = ax.bar(range(len(ORDER)), vals, color=[COLOUR[v] for v in ORDER], width=0.66)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
            ha="center", fontsize=9)
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER, rotation=28, ha="right", fontsize=8.5)
ax.set_ylabel("Number of PNe")
ax.set_ylim(0, max(vals) * 1.15)
ax.set_title("Our classification, radio criteria only", fontsize=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax3 = axes[1]
xs3 = np.arange(2)
bottom = np.zeros(2)
tot = g_hash.sum(axis=1).astype(float)
for j, v in enumerate(ORDER):
    f = g_hash[:, j] / np.where(tot > 0, tot, 1) * 100
    ax3.bar(xs3, f, bottom=bottom, color=COLOUR[v], width=0.6)
    for i in range(2):
        if f[i] >= 7:
            ax3.text(xs3[i], bottom[i] + f[i] / 2, f"{f[i]:.0f}%", ha="center",
                     va="center", fontsize=8, color="w")
    bottom += f
ax3.set_xticks(xs3)
ax3.set_xticklabels(["true PN\n(HASH T)", "probable PN\n(HASH P)"], fontsize=9)
ax3.set_ylabel("Per cent of detections")
ax3.set_ylim(0, 100)
ax3.set_title("Radio verdict within each optical class", fontsize=10)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
print(f"\nwrote {os.path.basename(OUT_VOT)} and 5 tables + {os.path.basename(OUT_FIG)}")
