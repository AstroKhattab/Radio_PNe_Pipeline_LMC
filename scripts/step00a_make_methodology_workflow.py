#!/usr/bin/env python3
"""Draw the PN LMC methodology workflow schematic with orthogonal connectors.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from _support.plot_style import apply_paper_style


apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUT = os.path.join(BASE, "04_Figures", "step00a_methodology_workflow.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

INK = "#243447"
MUTED = "#607080"
STRUCTURE = "#9AA8B6"
MEERKAT = "#0072B2"
ASKAP = "#8E5AA9"
JOINT = "#6F2DBD"
GOLD = "#D28A00"
TEAL = "#237A66"
RED = "#B84A4A"


fig, ax = plt.subplots(figsize=(17.4, 11.8), facecolor="white")
ax.set_xlim(0, 15.6)
ax.set_ylim(0, 10.5)
ax.axis("off")


def phase_label(y, text):
    """Add a restrained row heading and horizontal guide."""
    ax.text(0.28, y, text.upper(), ha="left", va="center", fontsize=8.0,
            fontweight="bold", color=MUTED, zorder=5)
    ax.plot([2.75, 15.30], [y, y], color="#E6EBF0", lw=0.65, zorder=0)


def workflow_box(
    x,
    y,
    w,
    h,
    title,
    lines,
    edge,
    *,
    fill="white",
    badge=None,
    title_size=9.8,
    body_size=7.8,
):
    """Draw one compact workflow step."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.035,rounding_size=0.11",
        facecolor=fill, edgecolor=edge, linewidth=1.35, zorder=4,
    ))
    title_x = x + 0.22
    if badge:
        badge_w = max(0.58, 0.15 * len(badge) + 0.20)
        ax.add_patch(FancyBboxPatch(
            (x + 0.16, y + h - 0.43), badge_w, 0.28,
            boxstyle="round,pad=0.012,rounding_size=0.055",
            facecolor=edge, edgecolor=edge, linewidth=0, zorder=5,
        ))
        ax.text(x + 0.16 + badge_w / 2, y + h - 0.29, badge,
                ha="center", va="center", fontsize=6.8,
                fontweight="bold", color="white", zorder=6)
        title_x = x + 0.24 + badge_w
    ax.text(title_x, y + h - 0.29, title, ha="left", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=6)
    for index, line in enumerate(lines):
        ax.text(x + 0.22, y + h - 0.69 - index * 0.29, line,
                ha="left", va="center", fontsize=body_size,
                color=MUTED, zorder=6)


def segment(points, colour=STRUCTURE, width=1.25, arrow=False):
    """Draw an orthogonal connector; every segment is horizontal or vertical."""
    for start, end in zip(points[:-1], points[1:]):
        if not (abs(start[0] - end[0]) < 1e-9 or abs(start[1] - end[1]) < 1e-9):
            raise ValueError(f"Non-orthogonal workflow segment: {start} -> {end}")
    if len(points) > 2:
        xs = [point[0] for point in points[:-1]]
        ys = [point[1] for point in points[:-1]]
        ax.plot(xs, ys, color=colour, lw=width, solid_capstyle="round", zorder=2)
    start, end = points[-2], points[-1]
    if arrow:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            linewidth=width, color=colour,
            connectionstyle="arc3,rad=0", zorder=3,
        ))
    else:
        ax.plot([start[0], end[0]], [start[1], end[1]], color=colour,
                lw=width, solid_capstyle="round", zorder=2)


def chip(x, y, w, text, colour):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, 0.29,
        boxstyle="round,pad=0.012,rounding_size=0.06",
        facecolor=colour, edgecolor="none", alpha=0.95, zorder=7,
    ))
    ax.text(x + w / 2, y + 0.145, text, ha="center", va="center",
            fontsize=6.6, fontweight="bold", color="white", zorder=8)


# Title and reading direction
ax.text(0.25, 10.19,
        "Methodology: from Reid candidates to a confidence-classified PN list",
        ha="left", va="center", fontsize=18.5, fontweight="bold", color=INK)
ax.text(15.32, 10.19, "Read from top to bottom",
        ha="right", va="center", fontsize=8.5, color=MUTED)

phase_label(9.82, "01  Candidate base")
phase_label(8.26, "02-03  Parallel survey association and quality control")
phase_label(6.16, "04  Multisurvey merge and HASH catalogue-status check")
phase_label(4.28, "05-07  Physical diagnostics")
phase_label(2.27, "06-07  Evidence synthesis and final scientific products")


# Step 01: one optical master
workflow_box(4.30, 8.72, 7.00, 0.92, "Unified optical PN candidate master", [
    "Reid and Parker classifications + Warren positions and photometry | N = 679",
], INK, fill="#F7F9FB", badge="01a", title_size=9.5, body_size=7.6)


# Steps 02-03: parallel radio-association branches
workflow_box(0.70, 6.58, 6.45, 1.35, "MeerKAT association", [
    "1.295 GHz catalogue match: 188 | visual radio/optical recovery: 32",
    "Footprint, offset, S/N and multiwavelength inspection | detections: 220",
], MEERKAT, fill="#F5FAFE", badge="02a/03a-b", title_size=9.4)

workflow_box(8.45, 6.58, 6.45, 1.35, "ASKAP association", [
    "888 MHz full-catalogue match: 61 | expected random matches: 1.52",
    "Accepted and review samples inspected separately | wider review cases: 3",
], ASKAP, fill="#FBF7FC", badge="02b/03c", title_size=9.4)

# Optical master splits cleanly into the two survey branches.
segment([(7.80, 8.72), (7.80, 8.34)], INK)
segment([(3.93, 8.34), (11.68, 8.34)], INK)
segment([(3.93, 8.34), (3.93, 7.93)], MEERKAT, arrow=True)
segment([(11.68, 8.34), (11.68, 7.93)], ASKAP, arrow=True)


# Step 04: branches join once; no source is discarded for lacking one survey.
workflow_box(1.20, 4.76, 6.40, 1.20, "Multisurvey master and radio-detected union", [
    "Separate survey measurements retained for all 679 Reid sources",
], JOINT, fill="#FBF9FD", badge="04b", title_size=9.2, body_size=7.3)

workflow_box(8.10, 4.76, 6.60, 1.20, "HASH catalogue-status check", [
    "V/163 LMC domain: 845 entries | one-to-one radius: 4.5 arcsec",
    "200 listed T | 8 listed P | 17 unmatched | Reid-derived status not scored",
], TEAL, fill="#F5FBF8", badge="04c", title_size=9.0, body_size=7.0)

segment([(3.93, 6.58), (3.93, 6.25)], MEERKAT)
segment([(11.68, 6.58), (11.68, 6.25)], ASKAP)
segment([(3.93, 6.25), (11.68, 6.25)], JOINT)
segment([(4.40, 6.25), (4.40, 5.96)], JOINT, arrow=True)
segment([(7.60, 5.36), (8.10, 5.36)], TEAL, arrow=True)

chip(1.43, 4.88, 1.35, "MKT only 164", MEERKAT)
chip(2.92, 4.88, 1.10, "Both 56", JOINT)
chip(4.16, 4.88, 1.40, "ASKAP only 5", GOLD)
chip(5.70, 4.88, 1.38, "Union 225", TEAL)


# Steps 05-07: three distinct physical analyses.
workflow_box(0.45, 2.67, 4.70, 1.35, "Combined radio SED fitting", [
    "All valid MeerKAT subbands + independent ASKAP 888 MHz measurement",
    "Reliable fits: 98 | thermal 31 | uncertain 31 | steep 36",
], MEERKAT, fill="#F5FAFE", badge="05a", title_size=8.9, body_size=6.9)

workflow_box(5.45, 2.67, 4.70, 1.35, "MIR/radio screening", [
    "SAGE 8 um ratios at 888, 996.646 and 1295 MHz",
    "Cohen near-1-GHz diagnostic used as supporting evidence",
], GOLD, fill="#FFFDF7", badge="06a", title_size=9.0, body_size=7.0)

workflow_box(10.45, 2.67, 4.70, 1.35, "MeerKAT radio PNLF", [
    "1.295-GHz MeerKAT sample: 220 | ASKAP-only sources excluded: 5",
    "Completeness checks + empirical and Ciardullo model comparison",
], RED, fill="#FFF9F8", badge="07a-d", title_size=9.0, body_size=6.9)

# The HASH-annotated union splits on one horizontal bus, then drops vertically
# into the physical analyses. HASH status is carried as an independent column.
segment([(11.40, 4.76), (11.40, 4.42)], TEAL)
segment([(2.80, 4.42), (12.80, 4.42)], JOINT)
segment([(2.80, 4.42), (2.80, 4.02)], MEERKAT, arrow=True)
segment([(7.80, 4.42), (7.80, 4.02)], GOLD, arrow=True)
segment([(12.80, 4.42), (12.80, 4.02)], RED, arrow=True)


# Evidence synthesis and principal scientific products.  The left panel makes
# the confidence rules explicit so the final PN list can be reproduced from
# catalogue columns rather than inferred from a visual impression.
workflow_box(0.45, 0.18, 10.00, 2.00,
             "Final confidence-classified radio PN candidate list (N = 225)", [
    "Inputs: Reid class | radio association | spectrum | MIR/radio; HASH status retained as a check only",
    "Positive units: secure MeerKAT +1 | both surveys +1 | thermal spectrum +1 | PN-like MIR/radio +1",
    "Base rule: High = Known/True + >=3; Probable = Known/True + >=1, Likely + >=2, Possible + >=3",
    "Caution: one conflict caps Possible; ASKAP-only/Bronze/component or >=2 conflicts -> review",
], TEAL, fill="#F5FBF8", badge="06b", title_size=9.5, body_size=7.1)

chip(0.78, 0.24, 2.08, "High-confidence 28", "#2A9D8F")
chip(3.00, 0.24, 1.75, "Probable 90", "#457B9D")
chip(4.89, 0.24, 1.75, "Possible 97", "#D7A93B")
chip(6.78, 0.24, 2.56, "Questionable/review 10", RED)

workflow_box(10.72, 0.18, 4.43, 2.00, "MeerKAT radio PNLF result", [
    "Primary sample: 220 at 1.295 GHz",
    "ASKAP-only sources excluded: 5",
    "Completeness and model diagnostics",
    "Best AIC: canonical Ciardullo",
], RED, fill="#FFF9F8", badge="07", title_size=8.9, body_size=7.2)

# SED and MIR evidence join before classification; the MeerKAT PNLF branch
# remains separate because it does not use the five ASKAP-only detections.
segment([(2.80, 2.67), (2.80, 2.38)], MEERKAT)
segment([(7.80, 2.67), (7.80, 2.38)], GOLD)
segment([(2.80, 2.38), (7.80, 2.38)], TEAL)
segment([(5.45, 2.38), (5.45, 2.18)], TEAL, arrow=True)
segment([(12.80, 2.67), (12.80, 2.18)], RED, arrow=True)


plt.savefig(OUT, format="pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUT)
