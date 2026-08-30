#!/usr/bin/env python3
"""Cross-match the 225 radio-detected Reid sources against HASH V/163.

This is a catalogue-status check, not an independent confirmation step.  It
adds the current HASH status to every MeerKAT/ASKAP-detected Reid row but does
not use HASH to change the radio/MIR confidence score assigned in Step 06b.

Input
-----
    03_Outputs/step04b_multisurvey_detected_union.vot
    01_Data/HASH_V163_full.vot

Outputs
-------
    03_Outputs/step04c_hash_status_all.vot
    03_Outputs/step04c_hash_crossmatch.vot
    03_Outputs/step04c_hash_unmatched.vot
    04_Figures/step04c_hash_status_summary.pdf
    04_Figures/step04c_hash_positional_diagnostics.pdf

Method
------
The official CDS/VizieR HASH release V/163 is restricted to rows whose
``Domain`` is LMC.  Candidate pairs within 4.5 arcsec are sorted by angular
separation and accepted greedily as a one-to-one match.  Ambiguity fields are
retained even though the present result has no multiple candidates.  HASH
``T`` is reported as "listed as True in HASH"; it is not called an independent
confirmation because almost every matched record points back to Reid & Parker.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Column, Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle, FancyBboxPatch

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from _support.plot_style import apply_paper_style


warnings.filterwarnings("ignore")
apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

RADIO_FILE = os.path.join(OUTS, "step04b_multisurvey_detected_union.vot")
HASH_FILE = os.path.join(DATA, "HASH_V163_full.vot")
OUT_ALL = os.path.join(OUTS, "step04c_hash_status_all.vot")
OUT_MATCHED = os.path.join(OUTS, "step04c_hash_crossmatch.vot")
OUT_UNMATCHED = os.path.join(OUTS, "step04c_hash_unmatched.vot")
FIG_SUMMARY = os.path.join(FIGS, "step04c_hash_status_summary.pdf")
FIG_POSITION = os.path.join(FIGS, "step04c_hash_positional_diagnostics.pdf")

MATCH_RADIUS_ARCSEC = 4.5
N_SHIFT_TRIALS = 1000
SHIFT_MIN_ARCSEC = 60.0
SHIFT_MAX_ARCSEC = 300.0
SHIFT_SEED = 163045
HASH_RELEASE = "CDS/VizieR V/163 (pnmain.dat, 25 June 2026)"
HASH_RETRIEVED = "12 August 2026"

HASH_TRUE = "#238B7A"
HASH_POSSIBLE = "#E5A11A"
HASH_UNMATCHED = "#C85757"
INK = "#243447"
MUTED = "#657686"

os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)


def clean_text(value):
    """Return a VOTable string without masked-value placeholders."""
    if np.ma.is_masked(value):
        return ""
    text = str(value).strip()
    return "" if text in {"--", "nan", "None"} else text


def numeric(table, name):
    values = np.ma.asarray(table[name], dtype=float)
    return np.asarray(values.filled(np.nan), dtype=float)


def status_label(status):
    return {
        "T": "Listed as True in HASH",
        "L": "Listed as Likely in HASH",
        "P": "Listed as Possible in HASH",
    }.get(status, f"HASH {status}" if status else "No HASH match")


def add_column(table, values, name, *, unit=None, description=None):
    if name in table.colnames:
        table.remove_column(name)
    column = Column(values, name=name, unit=unit)
    if description:
        column.description = description
    table.add_column(column)


def add_card(axis, number, label, colour, detail):
    axis.set_axis_off()
    axis.add_patch(FancyBboxPatch(
        (0.02, 0.06), 0.96, 0.88,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        transform=axis.transAxes, facecolor="white", edgecolor=colour,
        linewidth=1.6,
    ))
    axis.add_patch(FancyBboxPatch(
        (0.02, 0.06), 0.035, 0.88,
        boxstyle="round,pad=0.0,rounding_size=0.04",
        transform=axis.transAxes, facecolor=colour, edgecolor=colour,
        linewidth=0,
    ))
    axis.text(0.10, 0.60, f"{number}", transform=axis.transAxes,
              ha="left", va="center", fontsize=20, fontweight="bold",
              color=INK)
    axis.text(0.10, 0.35, label, transform=axis.transAxes,
              ha="left", va="center", fontsize=10.5, fontweight="bold",
              color=colour)
    axis.text(0.95, 0.17, detail, transform=axis.transAxes,
              ha="right", va="center", fontsize=7.8, color=MUTED)


def normalized_status_bars(axis, groups, group_values, statuses, title):
    colours = {"T": HASH_TRUE, "P": HASH_POSSIBLE, "No match": HASH_UNMATCHED}
    left = np.zeros(len(groups), dtype=float)
    for status in statuses:
        raw = np.asarray([
            np.sum((group_values == group) & (match_class == status))
            for group in groups
        ], dtype=float)
        totals = np.asarray([np.sum(group_values == group) for group in groups], dtype=float)
        percentages = np.divide(raw, totals, out=np.zeros_like(raw), where=totals > 0) * 100.0
        bars = axis.barh(np.arange(len(groups)), percentages, left=left,
                         color=colours[status], edgecolor="white",
                         linewidth=0.8, label={"T": "HASH True", "P": "HASH Possible",
                                                "No match": "No 4.5\" match"}[status])
        for bar, count, percent in zip(bars, raw.astype(int), percentages):
            if count and percent >= 4.5:
                axis.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_y() + bar.get_height() / 2,
                          str(count), ha="center", va="center",
                          fontsize=8.5, fontweight="bold", color="white")
            elif count:
                axis.text(bar.get_x() + bar.get_width() + 0.8,
                          bar.get_y() + bar.get_height() / 2,
                          str(count), ha="left", va="center",
                          fontsize=8.0, fontweight="bold", color=colours[status])
        left += percentages

    totals = [int(np.sum(group_values == group)) for group in groups]
    axis.set_yticks(np.arange(len(groups)), [f"{group}  (n={total})" for group, total in zip(groups, totals)])
    axis.set_xlim(0, 104)
    axis.set_xlabel("Percentage within group")
    axis.set_title(title, loc="left", fontweight="bold", color=INK)
    axis.grid(axis="x", color="#DCE3E9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.invert_yaxis()


print("step04c: cross-matching the radio detections against HASH")

radio = Table.read(RADIO_FILE, format="votable")
hash_full = Table.read(HASH_FILE, format="votable")
lmc_mask = np.asarray([clean_text(value) == "LMC" for value in hash_full["Domain"]])
hash_lmc = hash_full[lmc_mask]

print("\n[1] Loaded catalogues")
print(f"    Radio-detected Reid union : {len(radio)}")
print(f"    HASH V/163 full catalogue : {len(hash_full)}")
print(f"    HASH rows in LMC domain   : {len(hash_lmc)}")

radio_coords = SkyCoord(numeric(radio, "RA") * u.deg, numeric(radio, "Dec") * u.deg)
hash_coords = SkyCoord(numeric(hash_lmc, "RAJ2000") * u.deg,
                       numeric(hash_lmc, "DEJ2000") * u.deg)

idx_radio, idx_hash, separations, _ = search_around_sky(
    radio_coords, hash_coords, MATCH_RADIUS_ARCSEC * u.arcsec
)
order = np.argsort(separations.to_value(u.arcsec))

candidate_count = np.bincount(idx_radio, minlength=len(radio)).astype(np.int16)
hash_candidate_count = np.bincount(idx_hash, minlength=len(hash_lmc)).astype(np.int16)

# Explicit one-to-one assignment, closest pair first.
match_hash_index = np.full(len(radio), -1, dtype=int)
match_separation = np.full(len(radio), np.nan, dtype=float)
used_hash: set[int] = set()
for candidate in order:
    i = int(idx_radio[candidate])
    j = int(idx_hash[candidate])
    if match_hash_index[i] >= 0 or j in used_hash:
        continue
    match_hash_index[i] = j
    match_separation[i] = separations[candidate].to_value(u.arcsec)
    used_hash.add(j)

matched = match_hash_index >= 0
ambiguous = candidate_count > 1

hash_id = np.full(len(radio), -1, dtype=np.int32)
hash_ra = np.full(len(radio), np.nan)
hash_dec = np.full(len(radio), np.nan)
hash_major = np.full(len(radio), np.nan)
hash_minor = np.full(len(radio), np.nan)
hash_png = np.full(len(radio), "", dtype="U16")
hash_name = np.full(len(radio), "", dtype="U64")
hash_catalogue = np.full(len(radio), "", dtype="U48")
hash_simbad = np.full(len(radio), "", dtype="U64")
hash_status = np.full(len(radio), "", dtype="U16")
hash_domain = np.full(len(radio), "", dtype="U8")
hash_coord_ref = np.full(len(radio), "", dtype="U32")
hash_morphology = np.full(len(radio), "", dtype="U16")
hash_status_result = np.full(len(radio), "not_listed_within_4p5arcsec", dtype="U32")
offset_ra = np.full(len(radio), np.nan)
offset_dec = np.full(len(radio), np.nan)

for i in np.where(matched)[0]:
    j = match_hash_index[i]
    row = hash_lmc[j]
    hash_id[i] = int(row["HASH"])
    hash_ra[i] = float(row["RAJ2000"])
    hash_dec[i] = float(row["DEJ2000"])
    hash_major[i] = float(row["MajDiam"]) if not np.ma.is_masked(row["MajDiam"]) else np.nan
    hash_minor[i] = float(row["MinDiam"]) if not np.ma.is_masked(row["MinDiam"]) else np.nan
    hash_png[i] = clean_text(row["PNG"])
    hash_name[i] = clean_text(row["Name"])
    hash_catalogue[i] = clean_text(row["Catalogue"])
    hash_simbad[i] = clean_text(row["SimbadID"])
    hash_status[i] = clean_text(row["PNstat"])
    hash_domain[i] = clean_text(row["Domain"])
    hash_coord_ref[i] = clean_text(row["rpos"])
    hash_morphology[i] = clean_text(row["MorphMainCl"]) + clean_text(row["MorphSubCl"])
    if hash_status[i] == "T":
        hash_status_result[i] = "listed_true"
    elif hash_status[i] == "L":
        hash_status_result[i] = "listed_likely"
    elif hash_status[i] == "P":
        hash_status_result[i] = "listed_possible"
    else:
        hash_status_result[i] = "listed_other"
    dra, ddec = radio_coords[i].spherical_offsets_to(hash_coords[j])
    offset_ra[i] = dra.to_value(u.arcsec)
    offset_dec[i] = ddec.to_value(u.arcsec)

hash_true = matched & (hash_status == "T")
hash_possible = matched & (hash_status == "P")
match_class = np.full(len(radio), "No match", dtype="U16")
match_class[hash_true] = "T"
match_class[hash_possible] = "P"
match_class[matched & ~hash_true & ~hash_possible] = "Other"

annotated = radio.copy()
add_column(annotated, matched.astype(np.int16), "hash_matched",
           description="1 when a one-to-one HASH LMC association lies within 4.5 arcsec")
add_column(annotated, candidate_count, "hash_candidate_count",
           description="Number of HASH LMC entries within the 4.5 arcsec search radius")
add_column(annotated, ambiguous.astype(np.int16), "hash_match_ambiguous",
           description="1 when more than one HASH candidate lies within 4.5 arcsec")
add_column(annotated, match_separation.astype("f4"), "hash_separation_arcsec",
           unit="arcsec", description="Reid-to-HASH angular separation")
add_column(annotated, offset_ra.astype("f4"), "hash_offset_ra_cosdec_arcsec", unit="arcsec")
add_column(annotated, offset_dec.astype("f4"), "hash_offset_dec_arcsec", unit="arcsec")
add_column(annotated, hash_id, "hash_id", description="Unique HASH idPNMain identifier; -1 if unmatched")
add_column(annotated, hash_png, "hash_png")
add_column(annotated, hash_name, "hash_name")
add_column(annotated, hash_catalogue, "hash_origin_catalogue")
add_column(annotated, hash_simbad, "hash_simbad_id")
add_column(annotated, hash_status, "hash_pn_status",
           description="HASH PNstat: T=True, L=Likely, P=Possible; blank if unmatched")
add_column(annotated, np.asarray([status_label(value) for value in hash_status], dtype="U32"),
           "hash_status_label")
add_column(annotated, hash_status_result, "hash_catalogue_status_result",
           description="HASH catalogue-status result; not independent evidence and not used in Step 06b")
add_column(annotated, hash_true.astype(np.int16), "hash_status_true",
           description="1 when HASH currently lists PNstat=T; not an independent confirmation flag")
add_column(annotated, hash_domain, "hash_domain")
add_column(annotated, hash_ra.astype("f8"), "hash_ra_deg", unit="deg")
add_column(annotated, hash_dec.astype("f8"), "hash_dec_deg", unit="deg")
add_column(annotated, hash_coord_ref, "hash_coordinate_reference")
add_column(annotated, hash_morphology, "hash_morphology")
add_column(annotated, hash_major.astype("f4"), "hash_major_diameter_arcsec", unit="arcsec")
add_column(annotated, hash_minor.astype("f4"), "hash_minor_diameter_arcsec", unit="arcsec")

annotated.meta["HASH_release"] = HASH_RELEASE
annotated.meta["HASH_retrieved"] = HASH_RETRIEVED
annotated.meta["match_radius_arcsec"] = MATCH_RADIUS_ARCSEC
annotated.meta["match_method"] = "one-to-one positional match; closest candidate first"
annotated.meta["classification_note"] = (
    "HASH is a catalogue-status check; provenance is usually Reid and Parker, "
    "so it is not independent evidence and does not alter our confidence score"
)

annotated.write(OUT_ALL, format="votable", overwrite=True)
annotated[matched].write(OUT_MATCHED, format="votable", overwrite=True)
annotated[~matched].write(OUT_UNMATCHED, format="votable", overwrite=True)

print("\n[2] Strict 4.5-arcsec result")
print(f"    HASH-listed associations : {int(matched.sum())}")
print(f"    Listed as True in HASH   : {int(hash_true.sum())}")
print(f"    Listed as Possible       : {int(hash_possible.sum())}")
print(f"    No HASH match            : {int((~matched).sum())}")
print(f"    Multiple candidates      : {int(ambiguous.sum())}")
if matched.any():
    print(f"    Median separation        : {np.nanmedian(match_separation):.4f} arcsec")
    print(f"    Maximum separation       : {np.nanmax(match_separation):.4f} arcsec")

# A reproducible local-shift test estimates the expected number of accidental
# associations without changing the catalogue or the acceptance rule.
rng = np.random.default_rng(SHIFT_SEED)
random_counts = np.zeros(N_SHIFT_TRIALS, dtype=int)
for trial in range(N_SHIFT_TRIALS):
    angle = rng.uniform(0.0, 360.0, len(radio_coords)) * u.deg
    distance = rng.uniform(SHIFT_MIN_ARCSEC, SHIFT_MAX_ARCSEC, len(radio_coords)) * u.arcsec
    shifted = radio_coords.directional_offset_by(angle, distance)
    _, shifted_sep, _ = shifted.match_to_catalog_sky(hash_coords)
    random_counts[trial] = int(np.sum(shifted_sep <= MATCH_RADIUS_ARCSEC * u.arcsec))
expected_random = float(np.mean(random_counts))
random_upper95 = float(np.percentile(random_counts, 95.0))

print("\n[3] Local-shift chance-alignment diagnostic")
print(f"    Trials                   : {N_SHIFT_TRIALS}")
print(f"    Mean accidental matches  : {expected_random:.3f}")
print(f"    95th percentile          : {random_upper95:.1f}")

# Summary figure
fig = plt.figure(figsize=(15.6, 8.8), facecolor="white")
outer = GridSpec(2, 1, height_ratios=[0.92, 3.50], hspace=0.30, figure=fig)
cards = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.18)
charts = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25)

add_card(fig.add_subplot(cards[0]), len(radio), "Radio-detected Reid sources", INK,
         "MeerKAT or ASKAP union")
add_card(fig.add_subplot(cards[1]), int(hash_true.sum()), "Listed as True in HASH", HASH_TRUE,
         "PNstat = T; not independent evidence")
add_card(fig.add_subplot(cards[2]), int(hash_possible.sum()), "Listed as Possible in HASH", HASH_POSSIBLE,
         "PNstat = P")
add_card(fig.add_subplot(cards[3]), int((~matched).sum()), "No strict HASH listing", HASH_UNMATCHED,
         "No LMC entry within 4.5 arcsec")

ax_reid = fig.add_subplot(charts[0])
reid_values = np.asarray([clean_text(value) for value in radio["reid_class"]])
reid_order = [value for value in ["Known", "True", "Possible", "Likely", "Unknown"]
              if np.any(reid_values == value)]
normalized_status_bars(ax_reid, reid_order, reid_values,
                       ["T", "P", "No match"], "HASH result by Reid classification")

ax_survey = fig.add_subplot(charts[1])
survey_values = np.asarray([clean_text(value) for value in radio["detection_surveys"]])
survey_labels = {"MeerKAT": "MeerKAT only", "both": "MeerKAT + ASKAP", "ASKAP": "ASKAP only"}
survey_groups_raw = [value for value in ["MeerKAT", "both", "ASKAP"] if np.any(survey_values == value)]
normalized_status_bars(ax_survey, survey_groups_raw, survey_values,
                       ["T", "P", "No match"], "HASH result by radio-detection subset")
ax_survey.set_yticklabels([
    f"{survey_labels.get(value, value)}  (n={int(np.sum(survey_values == value))})"
    for value in survey_groups_raw
])

handles, labels = ax_reid.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.038))
fig.suptitle("HASH catalogue-status check for the multisurvey radio PN sample",
             x=0.055, y=0.982, ha="left", fontsize=17.2,
             fontweight="bold", color=INK)
fig.text(0.055, 0.945,
         "Official HASH V/163 LMC domain | positional radius 4.5 arcsec | one-to-one associations",
         ha="left", va="center", fontsize=9.7, color=MUTED)
fig.text(0.055, 0.012,
         f"{HASH_RELEASE}; retrieved {HASH_RETRIEVED}.  "
         f"Local-shift test: {expected_random:.3f} accidental matches expected per trial "
         f"({N_SHIFT_TRIALS} trials; 60-300 arcsec shifts). HASH status is not an extra confidence unit.",
         ha="left", va="bottom", fontsize=8.1, color=MUTED)
fig.subplots_adjust(top=0.90, bottom=0.12, left=0.07, right=0.98)
fig.savefig(FIG_SUMMARY, format="pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

# Positional diagnostics figure
fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), facecolor="white")

ax = axes[0]
sep_values = np.sort(match_separation[matched])
ecdf = np.arange(1, len(sep_values) + 1) / len(sep_values)
ax.step(np.maximum(sep_values, 1.0e-3), ecdf, where="post", color=HASH_TRUE, linewidth=2.2)
ax.axvline(MATCH_RADIUS_ARCSEC, color=HASH_UNMATCHED, linestyle="--", linewidth=1.5,
           label="Acceptance radius = 4.5\"")
ax.axvline(np.nanmedian(sep_values), color=INK, linestyle=":", linewidth=1.4,
           label=f"Median = {np.nanmedian(sep_values):.3f}\"")
ax.set_xscale("log")
ax.set_xlim(8.0e-4, 6.0)
ax.set_ylim(0, 1.03)
ax.set_xlabel("Reid-HASH separation (arcsec; log scale)")
ax.set_ylabel("Cumulative matched fraction")
ax.set_title("A  Separation distribution", loc="left", fontweight="bold", color=INK)
ax.legend(frameon=False, loc="lower right", fontsize=8.3)
ax.grid(color="#DCE3E9", linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for status, marker, colour, label, size in [
    ("T", "o", HASH_TRUE, "HASH True", 24),
    ("P", "D", HASH_POSSIBLE, "HASH Possible", 44),
]:
    use = matched & (hash_status == status)
    ax.scatter(offset_ra[use], offset_dec[use], s=size, marker=marker,
               color=colour, edgecolor="white", linewidth=0.5, alpha=0.88,
               label=f"{label} (n={int(use.sum())})", zorder=3)
ax.add_patch(Circle((0, 0), MATCH_RADIUS_ARCSEC, fill=False,
                    edgecolor=HASH_UNMATCHED, linewidth=1.5, linestyle="--"))
ax.axhline(0, color="#B9C4CE", linewidth=0.7)
ax.axvline(0, color="#B9C4CE", linewidth=0.7)
ax.set_xlim(-4.8, 4.8)
ax.set_ylim(-4.8, 4.8)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$\Delta$RA cos(Dec) (arcsec)")
ax.set_ylabel(r"$\Delta$Dec (arcsec)")
ax.set_title("B  Positional residuals", loc="left", fontweight="bold", color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=8.2)
ax.grid(color="#E1E7EC", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[2]
ra_values = numeric(radio, "RA")
dec_values = numeric(radio, "Dec")
ax.scatter(ra_values, dec_values, s=13, facecolor="#CBD4DB", edgecolor="none",
           alpha=0.65, label="All radio detections")
ax.scatter(ra_values[hash_true], dec_values[hash_true], s=18, marker="o",
           facecolor=HASH_TRUE, edgecolor="white", linewidth=0.35, alpha=0.82,
           label=f"HASH True (n={int(hash_true.sum())})")
ax.scatter(ra_values[hash_possible], dec_values[hash_possible], s=38, marker="D",
           facecolor=HASH_POSSIBLE, edgecolor=INK, linewidth=0.35,
           label=f"HASH Possible (n={int(hash_possible.sum())})")
ax.scatter(ra_values[~matched], dec_values[~matched], s=45, marker="x",
           color=HASH_UNMATCHED, linewidth=1.35,
           label=f"No match (n={int((~matched).sum())})")
ax.invert_xaxis()
ax.set_xlabel("Right ascension (deg, J2000)")
ax.set_ylabel("Declination (deg, J2000)")
ax.set_title("C  Distribution across the LMC", loc="left", fontweight="bold", color=INK)
ax.legend(frameon=False, loc="lower left", fontsize=7.8)
ax.grid(color="#E1E7EC", linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("HASH positional cross-match diagnostics",
             x=0.06, y=1.01, ha="left", fontsize=16.2,
             fontweight="bold", color=INK)
fig.text(0.06, 0.005,
         f"All {int(matched.sum())} accepted pairs are one-to-one; "
         f"no radio source has multiple HASH candidates within {MATCH_RADIUS_ARCSEC:.1f} arcsec.  "
         "A zero/near-zero separation is expected where HASH ingested the Reid position.",
         ha="left", va="bottom", fontsize=8.1, color=MUTED)
fig.subplots_adjust(top=0.89, bottom=0.16, left=0.065, right=0.985, wspace=0.30)
fig.savefig(FIG_POSITION, format="pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\n[4] Outputs")
for path in [OUT_ALL, OUT_MATCHED, OUT_UNMATCHED, FIG_SUMMARY, FIG_POSITION]:
    print(f"    {path}")
print("\nstep04c complete")
