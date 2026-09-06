"""Figure 1: the pipeline, two rows of stages.

Every count on this figure is read out of the tables the pipeline wrote, not
typed in.  A diagram with hand-entered numbers goes stale the first time a cut
changes and there is nothing to notice it, which is what happened to the
previous version of this file.

Input   the step tables in the outputs directory
Output  figures/step00b_methodology_workflow_2row.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

warnings.simplefilter("ignore", AstropyWarning)
plt.rcParams.update({'font.family': 'DejaVu Sans'})

cfg = config.load()
REPO_URL = "github.com/AstroKhattab/Radio_PNe_Pipeline_LMC"


def read(name):
    return Table.read(cfg.out(name), format="votable")


def text(table, key):
    return np.array([str(x).strip() for x in table[key]])


def csv_lookup(name, key_column, value_column):
    t = Table.read(cfg.out(name), format="ascii.csv")
    return {str(k).strip(): v for k, v in zip(t[key_column], t[value_column])}


parent = read("step1_parent_catalogue.vot")
outside = read("step3a_meerkat_outside.vot")
to_check = read("step3a_meerkat_to_check.vot")
visual = read("step3b_meerkat_visual.vot")
master = read("step3d_multisurvey_master.vot")
tally = read("step5d_criteria_tally.vot")
pnlf = read("step6a_pnlf.vot")

n_parent = len(parent)
pnstat = text(parent, "hash_pnstat")
n_true, n_prob = int((pnstat == "T").sum()), int((pnstat == "P").sum())
n_rp = int(np.sum([not r.startswith("HASH") for r in text(parent, "RP_ID")]))
n_phot = int(np.asarray(parent["has_phot"], dtype=int).sum())
n_setaside = len(read("step1_lmc_nonpn.vot"))

mkt_stats = csv_lookup("step2a_meerkat_match_stats.csv", "quantity", "value")
askap_stats = csv_lookup("step2b_askap_match_stats.csv", "quantity", "value")
n_mkt = int(mkt_stats["n_matched"])
n_askap = int(askap_stats["n_accepted"])
median_offset = mkt_stats["median_offset_arcsec"]
chance = askap_stats["expected_chance_matches"]

union = csv_lookup("step3d_detection_summary.csv", "category", "N")
n_union = int(union["radio_union"])

sp = text(tally, "sp_class")
n_reliable = int(np.isin(sp, ["thermal", "uncertain", "steep"]).sum())
n_thermal = int((sp == "thermal").sum())
n_steep = int((sp == "steep").sum())

criteria = Table.read(cfg.out("step5d_criteria_outcomes.csv"), format="ascii.csv")
n_above = int(criteria["fail"][list(criteria["criterion"]).index("4c_flux_ceiling")])

verdict = text(tally, "radio_only_class")
counts = {v: int((verdict == v).sum()) for v in
          ("Radio_High", "Radio_Possible", "Radio_Weak", "Radio_Rejected")}
hs = text(tally, "hash_status_label")
rate_true = 100 * np.sum((hs == "Listed as True in HASH") & (verdict == "Radio_High")) \
    / max(np.sum(hs == "Listed as True in HASH"), 1)
rate_prob = 100 * np.sum((hs == "Listed as Possible in HASH") & (verdict == "Radio_High")) \
    / max(np.sum(hs == "Listed as Possible in HASH"), 1)

n_fitted = int(np.asarray(pnlf["pnlf_included"], dtype=int).sum())
models = Table.read(cfg.out("step6a_model_parameters.csv"), format="ascii.csv")
canonical = models[np.array([str(m) for m in models["model"]]) == "Ciardullo (canonical)"][0]
m_star = float(str(canonical["parameters"]).split(",")[-1])
redchi = float(canonical["redchi"])
n_bins = int(canonical["dof"]) + int(canonical["k"])

NAVY = '#12305a'; BLUE = '#2f6fb3'; TEAL = '#1b8a8f'; AMBER = '#cf7f28'; RED = '#b4402f'
GREY = '#59626e'; EDGE = '#c3cdd9'

fig, ax = plt.subplots(figsize=(13.2, 8.0))
ax.set_xlim(-1.5, 103); ax.set_ylim(0, 100); ax.axis('off')

BW = 29.2      # box width
GAPX = 3.6     # horizontal gap
BH = 36.0      # box height


def stage(x, y, title, colour, items):
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.4,rounding_size=1.1",
                 fc='white', ec=EDGE, lw=1.2, zorder=2))
    ax.add_patch(FancyBboxPatch((x, y + BH - 5.4), BW, 5.4, boxstyle="round,pad=0.4,rounding_size=1.1",
                 fc=colour, ec=colour, lw=1.2, zorder=3))
    ax.text(x + BW / 2, y + BH - 2.7, title, ha='center', va='center', color='white',
            fontsize=13.5, fontweight='bold', zorder=4)
    yy = y + BH - 8.4
    for code, txt, note in items:
        ax.text(x + 1.4, yy, code, ha='left', va='top', fontsize=9.4, color=colour, fontweight='bold', zorder=4)
        ax.text(x + 6.4, yy, txt, ha='left', va='top', fontsize=10.6, color='#161b22', zorder=4)
        ax.text(x + 6.4, yy - 2.9, note, ha='left', va='top', fontsize=9.2, color=GREY, style='italic', zorder=4)
        yy -= 7.4


def harrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=19,
                                 lw=2.0, color=GREY, zorder=5))


ROW1 = 48.0; ROW2 = 7.0
X = [3.0, 3.0 + BW + GAPX, 3.0 + 2 * (BW + GAPX)]

stage(X[0], ROW1, '1.  PARENT CATALOGUE', NAVY, [
    ('1', 'HASH V/163, LMC domain', f'{n_true} true + {n_prob} probable  →  {n_parent} PNe'),
    ('', 'Reid & Parker cross-match', f'RP number and optical class  →  {n_rp}'),
    ('', 'Spitzer SAGE 8 μm', f'{n_phot} PNe with photometry'),
    ('', f'{n_setaside} LMC entries set aside', 'clusters, SNRs, H II regions')])
harrow(X[0] + BW + 0.6, X[1] - 0.6, ROW1 + BH / 2)

stage(X[1], ROW1, '2.  CROSS-MATCH', BLUE, [
    ('2a', 'Parent × MeerKAT 1.295 GHz', f'4.5″ nearest neighbour  →  {n_mkt}'),
    ('2b', 'Parent × ASKAP 888 MHz', f'4.5″ accept, wider pairs flagged  →  {n_askap}'),
    ('', 'One parent, both surveys', f'the bookkeeping closes on {n_parent}'),
    ('', f'Median offset {median_offset:.2f}″', f'{chance:.1f} chance matches expected')])
harrow(X[1] + BW + 0.6, X[2] - 0.6, ROW1 + BH / 2)

stage(X[2], ROW1, '3.  INSPECT & QA', TEAL, [
    ('3a', 'Split the unmatched', f'{len(to_check)} inspected, {len(outside)} outside the mosaic'),
    ('3b', 'Inspect and extract', f'aperture + local RMS  →  {len(visual)} recovered'),
    ('3b2', 'Aperture correction', f'enclosed-flux factor {cfg.enclosed_fraction():.2f} applied'),
    ('3d', 'Multi-survey union', f'{n_union} detected by either survey')])

# wrap-around connector from the end of row 1 to the start of row 2
yA = ROW1 + BH / 2; yB = ROW2 + BH / 2
for a, b in (((X[2] + BW + 0.6, yA), (X[2] + BW + 3.0, yA)),
             ((X[2] + BW + 3.0, yA), (X[2] + BW + 3.0, (yA + yB) / 2)),
             ((X[2] + BW + 3.0, (yA + yB) / 2), (1.0, (yA + yB) / 2)),
             ((1.0, (yA + yB) / 2), (1.0, yB))):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-', lw=2.0, color=GREY, zorder=5))
ax.add_patch(FancyArrowPatch((1.0, yB), (X[0] - 0.6, yB), arrowstyle='-|>',
                             mutation_scale=19, lw=2.0, color=GREY, zorder=5))

stage(X[0], ROW2, '4.  THREE CRITERIA', AMBER, [
    ('4a', 'Spectral index', f'{n_reliable} reliable  →  {n_thermal} thermal, {n_steep} steep'),
    ('4b', 'MIR / radio ratio', '888 or 1295 MHz, never rescaled'),
    ('4c', 'Radio flux ceiling',
     f'{cfg["flux_ceiling"]["ceiling_mjy"]} mJy at {cfg["distance"]["lmc_kpc"]} kpc  →  {n_above} above'),
    ('', 'No optical input', 'the criteria stay independent')])
harrow(X[0] + BW + 0.6, X[1] - 0.6, ROW2 + BH / 2)

stage(X[1], ROW2, '5.  CLASSIFICATION', RED, [
    ('5d', 'Scored out of what is measurable',
     f'{counts["Radio_High"]} High / {counts["Radio_Possible"]} Possible / {counts["Radio_Weak"]} Weak'),
    ('5d', 'Hard rejections',
     f'above the ceiling, or offset > 4.5″  →  {counts["Radio_Rejected"]}'),
    ('5d', 'Tested against HASH and Reid',
     f'{rate_true:.0f}% vs {rate_prob:.0f}% reach the highest class'),
    ('', 'Optical class never used', 'which is what makes the test valid')])
harrow(X[1] + BW + 0.6, X[2] - 0.6, ROW2 + BH / 2)

stage(X[2], ROW2, '6.  RADIO PNLF', NAVY, [
    ('6a', 'Luminosities and binning', f'{n_fitted} enter the fit, {n_bins} bins'),
    ('6a', 'Ten models ranked by AIC', 'Ciardullo + empirical forms'),
    ('6b', 'Canonical Ciardullo', f'M* = −{abs(m_star):.2f} mag, red-χ² = {redchi:.2f}'),
    ('', 'Two masks only', 'the 5σ limit and the flux ceiling')])

ax.text(3.0, 97.0, 'Radio planetary-nebula pipeline for the Large Magellanic Cloud',
        fontsize=17.5, fontweight='bold', color=NAVY, ha='left')
ax.text(3.0, 92.6, 'Each block is one numbered script in the public repository; every count is read from the tables this run wrote.',
        fontsize=11.2, color=GREY, ha='left')
ax.plot([3.0, 97.0], [89.6, 89.6], lw=1.0, color=EDGE)
ax.text(97.0, 1.5, REPO_URL, fontsize=9.6, color=GREY, ha='right', style='italic')

fig.tight_layout(pad=0.3)
out = cfg.fig("step00b_methodology_workflow_2row.pdf")
fig.savefig(out)
print("wrote", out)
