"""Check the identities that have to hold between the pipeline's outputs.

These are not unit tests of the code; they are arithmetic that must close if
the chain is consistent.  Every one of them was written because it either
caught something or would have caught something:

  - the four detection categories must sum to the parent sample;
  - each step must read the same number of rows the step before it wrote;
  - the four spectral categories must sum to the detections, which is the
    identity the manuscript's own sentence violated;
  - a criterion's pass, fail and not-measurable counts must sum to the
    detections;
  - the score tally must sum to the detections;
  - the four radio classes must sum to the detections;
  - the flux ceiling and the PNLF must agree about which sources are above it;
  - derived constants must equal what they are derived from.

    python3 tests/check_bookkeeping.py
"""

import os
import sys
import warnings

import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
from _support import config

warnings.simplefilter("ignore", AstropyWarning)
cfg = config.load()

failures = []


def check(description, condition, detail=""):
    if condition:
        print(f"  ok    {description}")
    else:
        print(f"  FAIL  {description}  {detail}")
        failures.append(description)


def vot(name):
    return Table.read(cfg.out(name), format="votable")


def csv(name):
    return Table.read(cfg.out(name), format="ascii.csv")


def text(table, key):
    return np.array([str(x).strip() for x in table[key]])


print("bookkeeping checks")

parent = vot("step1_parent_catalogue.vot")
n_parent = len(parent)
nonpn = vot("step1_lmc_nonpn.vot")
check("parent RP_ID is unique", len(set(text(parent, "RP_ID"))) == n_parent)
check("every parent source has PNstat T or P",
      set(text(parent, "hash_pnstat")) <= set(cfg["parent"]["pn_status"]))

matched = vot("step2a_meerkat_matched.vot")
unmatched = vot("step2a_meerkat_unmatched.vot")
check("step 2a matched + unmatched = parent",
      len(matched) + len(unmatched) == n_parent,
      f"{len(matched)} + {len(unmatched)} != {n_parent}")
check("no MeerKAT source is matched to two PNe",
      len(set(zip(np.asarray(matched["mkt_ra"], float),
                  np.asarray(matched["mkt_dec"], float)))) == len(matched))

askap = vot("step2b_askap_matched.vot")
check("no ASKAP source is matched to two PNe",
      len(set(zip(np.asarray(askap["askap_ra"], float),
                  np.asarray(askap["askap_dec"], float)))) == len(askap))
check("every accepted ASKAP match is within the acceptance radius",
      np.all(np.asarray(askap["askap_separation_arcsec"], float)
             <= cfg["crossmatch"]["accept_arcsec"]))

outside = vot("step3a_meerkat_outside.vot")
to_check = vot("step3a_meerkat_to_check.vot")
check("step 3a inside + outside = the step 2a unmatched",
      len(outside) + len(to_check) == len(unmatched),
      f"{len(outside)} + {len(to_check)} != {len(unmatched)}")

visual = vot("step3b_meerkat_visual.vot")
check("every visual detection came from the inspection queue",
      set(text(visual, "RP_ID")) <= set(text(to_check, "RP_ID")))
check("the aperture correction was applied",
      np.allclose(np.asarray(visual["enclosed_flux_fraction"], float),
                  cfg.enclosed_fraction()))

detected = vot("step3c_meerkat_detected.vot")
nondetected = vot("step3c_meerkat_nondetected.vot")
check("step 3c detected = cross-matched + visual",
      len(detected) == len(matched) + len(visual),
      f"{len(detected)} != {len(matched)} + {len(visual)}")
check("step 3c detected + non-detected + outside = parent",
      len(detected) + len(nondetected) + len(outside) == n_parent,
      f"{len(detected)} + {len(nondetected)} + {len(outside)} != {n_parent}")
check("the detection catalogue kept its flux column",
      "mkt_int_flux_Jy" in detected.colnames)

master = vot("step3d_multisurvey_master.vot")
det_mkt = np.asarray(master["det_meerkat"], int) == 1
det_ask = np.asarray(master["det_askap"], int) == 1
n_both = int((det_mkt & det_ask).sum())
n_mkt_only = int((det_mkt & ~det_ask).sum())
n_ask_only = int((~det_mkt & det_ask).sum())
n_neither = int((~det_mkt & ~det_ask).sum())
check("both + MeerKAT-only + ASKAP-only + neither = parent",
      n_both + n_mkt_only + n_ask_only + n_neither == n_parent,
      f"{n_both} + {n_mkt_only} + {n_ask_only} + {n_neither} != {n_parent}")
check("the MeerKAT detections agree with step 3c",
      n_both + n_mkt_only == len(detected))
check("the ASKAP detections agree with step 2b",
      n_both + n_ask_only == len(askap))

union = vot("step3d_multisurvey_detected_union.vot")
check("the union table holds the union",
      len(union) == n_both + n_mkt_only + n_ask_only)

spectral = vot("step4a_spectral_index.vot")
check("step 4a reads the union", len(spectral) == len(union))
sp = text(spectral, "sp_class")
n_det = len(spectral)
classes = {name: int((sp == name).sum())
           for name in ("thermal", "uncertain", "steep", "no_reliable_alpha")}
check("the four spectral categories sum to the detections",
      sum(classes.values()) == n_det, f"{classes} sums to {sum(classes.values())}, not {n_det}")
check("the spectral categories are exclusive",
      set(sp) <= set(classes), f"unexpected classes {set(sp) - set(classes)}")

reliable = text(spectral, "alpha_fit_status") == "reliable"
check("every classified source passed the reliability cuts",
      np.all(reliable[np.isin(sp, ["thermal", "uncertain", "steep"])]))
check("no source that failed a cut carries a spectral class",
      not np.any(np.isin(sp[~reliable], ["thermal", "uncertain", "steep"])))
alpha = np.asarray(spectral["alpha_combined"], float)
alpha_err = np.asarray(spectral["alpha_combined_err"], float)
chi2 = np.asarray(spectral["alpha_combined_redchisq"], float)
npts = np.asarray(spectral["alpha_combined_n_points"], int)
cuts = cfg["spectral_index"]
check("the reliable set obeys the three cuts exactly",
      np.all((npts[reliable] >= cuts["min_points_reliable"])
             & (alpha_err[reliable] < cuts["max_alpha_error"])
             & (chi2[reliable] < cuts["max_reduced_chisq"])))
check("the ASKAP point is used only where ASKAP detected the source",
      np.all(np.asarray(spectral["alpha_has_askap"], int)
             <= np.asarray(spectral["det_askap"], int)))

tally = vot("step5d_criteria_tally.vot")
check("step 5 reads every detection", len(tally) == n_det)
criteria = csv("step5d_criteria_outcomes.csv")
for row in criteria:
    total = int(row["pass"]) + int(row["fail"]) + int(row["not_measurable"])
    check(f"criterion {row['criterion']} sums to the detections", total == n_det,
          f"{total} != {n_det}")

score = csv("step5d_passed_vs_measurable.csv")
check("the score tally sums to the detections",
      int(np.sum(score["N"])) == n_det, f"{int(np.sum(score['N']))} != {n_det}")
check("nothing passed more criteria than were measurable",
      np.all(np.asarray(tally["n_criteria_passed"], int)
             <= np.asarray(tally["n_criteria_measurable"], int)))

verdict = text(tally, "radio_only_class")
counts = {name: int((verdict == name).sum()) for name in
          ("Radio_High", "Radio_Possible", "Radio_Weak", "Radio_Rejected")}
check("the four radio classes sum to the detections",
      sum(counts.values()) == n_det, f"{counts}")
measurable = np.asarray(tally["n_criteria_measurable"], int)
passed = np.asarray(tally["n_criteria_passed"], int)
high = verdict == "Radio_High"
check("Radio_High passed everything measurable, at least two of them",
      np.all((passed[high] == measurable[high])
             & (measurable[high] >= cfg["scoring"]["min_measurable_for_high"])))
rejected = verdict == "Radio_Rejected"
flux_mjy = np.asarray(tally["mkt_int_flux_Jy"], float) * 1e3
separation = np.asarray(tally["mkt_separation_arcsec"], float)
check("every rejection is a hard physical failure",
      np.all((flux_mjy[rejected] > cfg["flux_ceiling"]["ceiling_mjy"])
             | (separation[rejected] > cfg["crossmatch"]["accept_arcsec"])))
# The claim that the score uses no optical information is worth testing rather
# than asserting: rebuild the verdict from the three criterion columns alone
# and check it reproduces the published one for every source.
criteria_only = np.vstack([np.asarray(tally[name], int)
                           for name in ("crit_thermal", "crit_mir", "crit_ceiling")])
rebuilt_measurable = (criteria_only >= 0).sum(axis=0)
rebuilt_passed = (criteria_only == 1).sum(axis=0)
fraction = np.divide(rebuilt_passed, rebuilt_measurable,
                     out=np.full(n_det, np.nan), where=rebuilt_measurable > 0)
rebuilt = np.full(n_det, "Radio_Weak", dtype="U16")
rebuilt[(rebuilt_measurable > 0) & (fraction > 0.5)] = "Radio_Possible"
rebuilt[(rebuilt_measurable >= cfg["scoring"]["min_measurable_for_high"])
        & (fraction == 1.0)] = "Radio_High"
rebuilt[(criteria_only[2] == 0)
        | (np.isfinite(separation) & (separation > cfg["crossmatch"]["accept_arcsec"]))] = "Radio_Rejected"
check("the verdict is a function of the three criteria and nothing else",
      np.array_equal(rebuilt, verdict),
      f"{int((rebuilt != verdict).sum())} sources differ")

for column in ("reid_classification", "hash_pn_status"):
    check(f"{column} is carried but not used by the score", column in tally.colnames)

pnlf = vot("step6a_pnlf.vot")
included = np.asarray(pnlf["pnlf_included"], int) == 1
above = np.asarray(tally["crit_ceiling"], int) == 0
check("the PNLF excludes exactly the ASKAP-only and the over-ceiling sources",
      int(included.sum()) == n_det - int(above.sum()) - n_ask_only
      - int(((np.asarray(tally["crit_ceiling"], int) == -1)
             & (np.asarray(tally["det_meerkat"], int) == 1)).sum()),
      f"{int(included.sum())} included")
check("no ASKAP-only source entered the PNLF",
      not np.any(included & (np.asarray(pnlf["det_meerkat"], int) == 0)))

high_sample = vot("step6g_pnlf_high.vot")
check("the high-confidence PNLF sample is exactly Radio_High",
      len(high_sample) == counts["Radio_High"],
      f"{len(high_sample)} != {counts['Radio_High']}")

# Derived constants must equal what they are derived from.
ceiling_mag = cfg.magnitude(cfg["flux_ceiling"]["ceiling_mjy"] * 1e-3)
check("the ceiling magnitude follows from the ceiling and the distance",
      np.isclose(cfg.ceiling_magnitude(), ceiling_mag))
check("an aperture of one beam half-width encloses half the flux",
      np.isclose(cfg.enclosed_fraction(cfg["meerkat"]["beam_fwhm_arcsec"] / 2), 0.5))

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)
print("all checks passed")
