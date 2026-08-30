"""Re-cut the spectral-index quality tiers and spectral classes.

Regenerates the recut catalogue that step07f, step07g and step08a read.

The 4.5-sigma detection work left the raw two-point and multi-point fits in
step05a untouched, so the quality cuts have to be re-applied downstream rather
than inside the fitting script.  This module applies them and writes a single
catalogue carrying every column the later steps expect.

Inputs
    03_Outputs/step4a_spectral_index.vot   fitted alpha_combined and its
                                            error, reduced chi-square and
                                            number of flux points (225 rows).
    03_Outputs/step5a_confidence.vot       joined by RP_ID for the Cohen
                                            mid-IR/radio ratio columns, the
                                            8 um fluxes and the confidence /
                                            evidence columns.  These are not
                                            recomputed here, only carried
                                            through so the recut file is
                                            self-contained.

Output
    03_Outputs/step5b_spectral_index_recut.vot  all input columns plus
    sp_class_recut, alpha_tier_recut and alpha_flag_reason.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import numpy as np
from astropy.table import Table

# Repository root, two levels up from 02_Scripts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "03_Outputs")

SPECTRAL_INDEX_FILE = os.path.join(OUTDIR, "step4a_spectral_index.vot")
CONFIDENCE_FILE = os.path.join(OUTDIR, "step5a_confidence.vot")
OUTPUT_FILE = os.path.join(OUTDIR, "step5b_spectral_index_recut.vot")

# A power law through fewer than four flux points is not constrained enough to
# quote a class from.
MIN_POINTS = 4
# Beyond this the 1-sigma error spans more than the full steep/thermal range.
MAX_ALPHA_ERR = 0.7
# Reduced chi-square above this means the power law is not describing the SED.
MAX_REDCHISQ = 10.0

# Class boundaries in alpha (S_nu ~ nu^alpha).
STEEP_MAX = -0.5
UNCERTAIN_MAX = -0.2
THERMAL_MAX = 2.0


def classify_alpha(alpha):
    if alpha < STEEP_MAX:
        return "steep"
    if alpha < UNCERTAIN_MAX:
        return "uncertain"
    if alpha <= THERMAL_MAX:
        return "thermal"
    # alpha > 2 is steeper than optically thick free-free and is not a class we
    # quote; no source in the current catalogue reaches here with a usable fit.
    return "no_fit"


def main():
    spec = Table.read(SPECTRAL_INDEX_FILE)
    conf = Table.read(CONFIDENCE_FILE)

    out = spec.copy()

    # step06b is step05a plus the mid-IR and confidence columns; bring across
    # only the columns step05a does not already have, matched on RP_ID.
    key = {str(rp): i for i, rp in enumerate(conf["RP_ID"])}
    order = np.array([key[str(rp)] for rp in out["RP_ID"]])
    for name in conf.colnames:
        if name not in out.colnames:
            out[name] = conf[name][order]

    alpha = np.asarray(out["alpha_combined"], dtype=float)
    alpha_err = np.asarray(out["alpha_combined_err"], dtype=float)
    redchisq = np.asarray(out["alpha_combined_redchisq"], dtype=float)
    n_points = np.asarray(out["alpha_combined_n_points"], dtype=float)

    tier = np.empty(len(out), dtype="U16")
    reason = np.empty(len(out), dtype="U19")
    sp_class = np.empty(len(out), dtype="U9")

    for i in range(len(out)):
        if not np.isfinite(alpha[i]):
            tier[i] = "no_fit"
            reason[i] = "no_fit"
            sp_class[i] = "no_fit"
            continue

        if not (n_points[i] >= MIN_POINTS):
            tier[i] = "no_fit"
            reason[i] = "insufficient_points"
            sp_class[i] = "no_fit"
            continue

        bad_chisq = not (redchisq[i] < MAX_REDCHISQ)
        bad_err = not (alpha_err[i] <= MAX_ALPHA_ERR)

        if bad_chisq:
            tier[i] = "flagged_poor_fit"
            reason[i] = "high_redchisq"
        elif bad_err:
            tier[i] = "flagged_poor_fit"
            reason[i] = "large_alpha_error"
        else:
            tier[i] = "reliable"
            reason[i] = "pass"

        # Flagged fits keep a class; they are simply not quoted without the flag.
        sp_class[i] = classify_alpha(alpha[i])

    out["sp_class_recut"] = sp_class
    out["alpha_tier_recut"] = tier
    out["alpha_flag_reason"] = reason

    out.write(OUTPUT_FILE, format="votable")

    print("wrote", OUTPUT_FILE, len(out), "rows")
    for col in ("alpha_tier_recut", "alpha_flag_reason", "sp_class_recut"):
        values, counts = np.unique(out[col], return_counts=True)
        print(col, dict(zip(values.tolist(), counts.tolist())))
    reliable = out["alpha_tier_recut"] == "reliable"
    values, counts = np.unique(out["sp_class_recut"][reliable], return_counts=True)
    print("sp_class_recut (reliable only)", dict(zip(values.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
