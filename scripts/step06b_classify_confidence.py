"""
step06b_classify_confidence.py
===============================
Step 6b — Transparent, evidence-based PN confidence classification.

The catalogue keeps Reid & Parker's classification unchanged in
``reid_classification`` and reports this work's result separately in
``our_classification``.  The result is an ordinal evidence grade, not a
statistical posterior probability.

Positive evidence units (one each)
----------------------------------
    * a secure MeerKAT catalogue association;
    * independent detection in both MeerKAT and ASKAP;
    * a reliable thermal-like radio spectrum (-0.2 <= alpha <= +2.0);
    * a near-1-GHz MIR/radio ratio in the broad PN-like screening band.

Base classification
-------------------
    High-confidence PN : Reid Known/True and >=3 positive units
    Probable PN        : Reid Known/True and >=1 positive unit, or Reid
                         Likely with >=2, or Reid Possible with >=3
    Possible PN        : all remaining sources

Caution rules
-------------
    * one reliable steep spectrum (alpha < -0.5) or one extreme MIR/radio diagnostic caps
      the result at Possible PN;
    * ASKAP-only detections, ASKAP Bronze/components, or two independent
      contradictory diagnostics are Questionable/review.

The MIR/radio ratio is deliberately not allowed to reject a candidate by
itself.  Cohen et al. measured integrated nebular IRAC fluxes, whereas the
current input is SAGE catalogue photometry; visual/aperture validation is
required before giving that diagnostic full evidential weight.

Inputs
------
    03_Outputs/step06a_mir_radio_ratio.vot

Outputs
-------
    03_Outputs/step06b_confidence.vot
    04_Figures/step06b_confidence_summary.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import warnings
from collections import Counter

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
FIGS = os.path.join(BASE, "04_Figures")
INPUT_FILE = os.path.join(OUTS, "step06a_mir_radio_ratio.vot")
OUT_VOT = os.path.join(OUTS, "step06b_confidence.vot")
FIGURE = os.path.join(FIGS, "step06b_confidence_summary.pdf")

os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

print("step06b: assigning PN confidence grades")

catalogue = Table.read(INPUT_FILE, format="votable")
n_sources = len(catalogue)
print(f"\n[1] Loaded {n_sources} multisurvey Reid sources")

our_classification = np.full(n_sources, "Possible PN", dtype="U24")
positive_units = np.zeros(n_sources, dtype=np.int16)
conflict_units = np.zeros(n_sources, dtype=np.int16)
review_required = np.zeros(n_sources, dtype=np.int16)
association_evidence = np.full(n_sources, "", dtype="U48")
spectral_evidence = np.full(n_sources, "", dtype="U40")
mir_evidence = np.full(n_sources, "", dtype="U48")
classification_basis = np.full(n_sources, "", dtype="U512")

print("\n[2] Applying the documented evidence rules...")
for k, row in enumerate(catalogue):
    reid = str(row["reid_classification"]).strip()
    surveys = str(row["detection_surveys"]).strip()
    mkt_method = str(row["mkt_detection_method"]).strip()
    spectral = str(row["sp_class"]).strip()
    fit_status = str(row["alpha_fit_status"]).strip()
    mir = str(row["cohen_diagnostic"]).strip()
    askap_list = str(row["askap_source_list"]).strip()
    askap_type = str(row["askap_source_type"]).strip()

    reasons = [f"Reid={reid}"]
    positive = 0
    conflicts = 0

    if mkt_method == "catalogue":
        positive += 1
        association_evidence[k] = "secure_MeerKAT_catalogue"
        reasons.append("secure MeerKAT catalogue association")
    elif mkt_method == "visual":
        association_evidence[k] = "MeerKAT_visual"
        reasons.append("MeerKAT visual extraction")
    elif surveys == "ASKAP":
        association_evidence[k] = "ASKAP_only_requires_review"
        reasons.append("ASKAP-only association")
    else:
        association_evidence[k] = "radio_association_unclear"

    if surveys == "both":
        positive += 1
        association_evidence[k] += "+ASKAP_confirmation"
        reasons.append("independent ASKAP confirmation")

    if spectral == "thermal" and fit_status == "reliable":
        positive += 1
        spectral_evidence[k] = "reliable_thermal_-0.2_to_+2.0"
        reasons.append("reliable thermal-like spectrum")
    elif spectral == "steep" and fit_status == "reliable":
        conflicts += 1
        spectral_evidence[k] = "reliable_steep_alpha_lt_-0.5"
        reasons.append("reliable steep spectrum: caution")
    elif spectral == "uncertain" and fit_status == "reliable":
        spectral_evidence[k] = "intermediate_alpha_-0.5_to_-0.2"
        reasons.append("intermediate spectral index: physically uncertain")
    else:
        spectral_evidence[k] = f"no_reliable_alpha_{fit_status}"
        reasons.append(f"no reliable spectral classification ({fit_status})")

    if mir == "PN_like":
        positive += 1
        mir_evidence[k] = "PN_like_screening_ratio_0.5_to_10"
        reasons.append("PN-like MIR/radio screening ratio")
    elif mir in {"HII_like", "radio_excess"}:
        conflicts += 1
        mir_evidence[k] = f"{mir}_screening_only"
        reasons.append(f"{mir} MIR/radio screening result: caution")
    elif mir == "intermediate_high":
        mir_evidence[k] = "intermediate_high_screening_ratio"
        reasons.append("intermediate-high MIR/radio screening ratio")
    else:
        mir_evidence[k] = "unavailable"
        reasons.append("no MIR/radio ratio")

    positive_units[k] = positive
    conflict_units[k] = conflicts

    if reid in {"Known", "True"} and positive >= 3:
        result = "High-confidence PN"
    elif (
        (reid in {"Known", "True"} and positive >= 1) or
        (reid == "Likely" and positive >= 2) or
        (reid == "Possible" and positive >= 3)
    ):
        result = "Probable PN"
    else:
        result = "Possible PN"

    association_issue = (
        surveys == "ASKAP" or askap_list == "BRONZE" or askap_type == "component"
    )
    if association_issue or conflicts >= 2:
        result = "Questionable/review"
        review_required[k] = 1
        if association_issue:
            reasons.append("radio association/ASKAP quality requires visual review")
        if conflicts >= 2:
            reasons.append("two independent contradictory diagnostics")
    elif conflicts == 1:
        result = "Possible PN"
        review_required[k] = 1
        reasons.append("single contradictory diagnostic caps confidence at Possible")

    our_classification[k] = result
    classification_basis[k] = "; ".join(reasons)

new_columns = {
    "our_classification": our_classification,
    "confidence_positive_units": positive_units,
    "confidence_conflict_units": conflict_units,
    "confidence_review_required": review_required,
    "association_evidence": association_evidence,
    "spectral_evidence": spectral_evidence,
    "mir_evidence": mir_evidence,
    "classification_basis": classification_basis,
}
for name, values in new_columns.items():
    if name in catalogue.colnames:
        catalogue.remove_column(name)
    catalogue.add_column(Column(values, name=name))

catalogue.write(OUT_VOT, format="votable", overwrite=True)
print(f"\n[3] Wrote {OUT_VOT}")

order = ["High-confidence PN", "Probable PN", "Possible PN", "Questionable/review"]
counts = Counter(our_classification)
colors = ["#2A9D8F", "#457B9D", "#E9C46A", "#C23B33"]

fig, ax = plt.subplots(figsize=(8.5, 5))
x = np.arange(len(order))
values = [counts[label] for label in order]
bars = ax.bar(x, values, color=colors, edgecolor="white")
ax.set_xticks(x, order, rotation=12, ha="right")
ax.set_ylabel("Number of sources")
ax.set_title("Evidence-based PN confidence classification")
ax.spines[["top", "right"]].set_visible(False)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center")
plt.tight_layout()
plt.savefig(FIGURE, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()

print(f"    Figure: {FIGURE}")
for label in order:
    print(f"    {label:<21}: {counts[label]}")
print("\nstep06b complete")
