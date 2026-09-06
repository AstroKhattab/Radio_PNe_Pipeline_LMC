"""
step5d_criteria_tally.py
========================
Step 5 - Score every detection on the three criteria alone.

Earlier versions of this analysis also carried an evidence grade that used the
Reid & Parker optical class as part of its rule set: a source could not reach
the highest grade unless it was already classified Known or True optically.
That grade cannot be compared against the optical classification, because it
takes it as an input, so it has been removed rather than reported alongside.

This step scores each detection on the three radio criteria and nothing else:

    4a  thermal spectral index      pass: thermal      fail: steep

The spectral class comes from Step 4a, where the reliability cuts of
Appendix A are applied (n_pts >= 4, delta alpha < 0.5, chi2_nu < 10).  An
index that failed one of those is not a measurement of a flat or a steep
spectrum, so criterion 4a is simply unavailable for it.  The cuts are applied
in exactly one place; a second, looser copy of them here is how the criterion
came to be scored on indices the paper says are unusable.
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

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings

import numpy as np
from astropy.table import Table, Column
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter("ignore", AstropyWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()
IN_FILE = cfg.out("step4c_flux_ceiling.vot")
OUT_VOT = cfg.out("step5d_criteria_tally.vot")
OUT_SUMMARY = cfg.out("step5d_tally_summary.csv")
OUT_CRIT = cfg.out("step5d_criteria_outcomes.csv")
OUT_SCORE = cfg.out("step5d_passed_vs_measurable.csv")
OUT_HASH = cfg.out("step5d_tally_vs_hash.csv")
OUT_REID = cfg.out("step5d_tally_vs_reid.csv")
OUT_PROM = cfg.out("step5d_promoted.csv")
OUT_CONT = cfg.out("step5d_contested.csv")

ORDER = ["Radio_High", "Radio_Possible", "Radio_Weak", "Radio_Rejected"]
MATCH_RAD = cfg["crossmatch"]["accept_arcsec"]
MIN_MEASURABLE_HIGH = cfg["scoring"]["min_measurable_for_high"]

MIR_LOW, MIR_HIGH = cfg["mir_radio"]["pn_band"]
MIR_HII = cfg["mir_radio"]["hii_above"]
CEILING_MJY = cfg["flux_ceiling"]["ceiling_mjy"]


def text(table, name):
    return np.array([str(x).strip() for x in table[name]])


print("=" * 62)
print("  Step 5  -  criteria tally, independent of the optical class")
print("=" * 62)

t = Table.read(IN_FILE, format="votable")
n = len(t)
print(f"\nradio detections : {n}")

# ------------------------------------------------------- criterion 1: alpha
sp = text(t, "sp_class")
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
verdict[(measurable >= MIN_MEASURABLE_HIGH) & (frac == 1.0)] = "Radio_High"
verdict[rejected] = "Radio_Rejected"

print("\ncriterion by criterion:")
print(f"    {'':<26}{'pass':>7}{'fail':>7}{'n/a':>7}")
crit_rows = []
for lab, name, c in (("thermal spectrum (4a)", "4a_spectral_index", c_alpha),
                     ("MIR/radio ratio (4b)", "4b_mir_radio_ratio", c_mir),
                     ("below flux ceiling (4c)", "4c_flux_ceiling", c_ceil)):
    crit_rows.append((name, int((c == 1).sum()), int((c == 0).sum()),
                      int((c == -1).sum())))
    print(f"    {lab:<26}{int((c == 1).sum()):>7}"
          f"{int((c == 0).sum()):>7}{int((c == -1).sum()):>7}")
Table(rows=crit_rows, names=("criterion", "pass", "fail", "not_measurable")).write(
    OUT_CRIT, format="ascii.csv", overwrite=True)

print("\nhow many criteria could be evaluated:")
for k in range(4):
    print(f"    {k} of 3 : {int((measurable == k).sum()):>4}")

print("\nscore out of what was measurable:")
score_rows = []
for m in range(1, 4):
    for p in range(m + 1):
        cnt = int(((measurable == m) & (passed == p)).sum())
        if cnt:
            score_rows.append((p, m, cnt))
            print(f"    {p} of {m} passed : {cnt:>4}")
score_rows.append((-1, 0, int((measurable == 0).sum())))   # -1: nothing measurable
Table(rows=score_rows, names=("passed", "measurable", "N")).write(
    OUT_SCORE, format="ascii.csv", overwrite=True)
assert sum(r[2] for r in score_rows) == n, "the score tally does not sum to the detections"

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
                    "mkt_int_flux_Jy", "alpha_combined", "sp_class",
                    "cohen_ratio_near1GHz", "n_criteria_passed",
                    "n_criteria_measurable", "radio_only_class"]
        if c in t.colnames]
t[show][promoted].write(OUT_PROM, format="ascii.csv", overwrite=True)
t[show][contested].write(OUT_CONT, format="ascii.csv", overwrite=True)
t.write(OUT_VOT, format="votable", overwrite=True)

# The figure this step used to draw was removed from the paper: the same
# information is in the tables, and a figure that only restates a table earns
# no space.  The tables below are still written.
print(f"\nwrote {os.path.basename(OUT_VOT)} and 7 tables")
