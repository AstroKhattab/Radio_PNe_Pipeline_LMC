"""Compare the radio PNLF under three masking choices.

Why this figure exists.  The bright end of the luminosity function is where
the contamination lives, and how it is handled decides M*.  Three treatments
are shown side by side on the same sample:

    A   no mask at all
    B   a hand-chosen bright-bin mask, the approach used before
    C   the physical flux ceiling of Step 4c, the approach adopted here

Panel B is the point of the figure.  Choosing which bright bins to drop by eye
lets the fit be anchored by whichever contaminant happens to survive the cut,
and the resulting M* is an artefact of that choice.  The ceiling is derived
from NGC 7027 at the LMC distance and does not depend on looking at the data.

Bars are coloured by spectral class so the steep-spectrum sources -- the
likely background galaxies -- are visible where they sit.

Input   03_Outputs/step6a_pnlf.vot
        03_Outputs/step5d_criteria_tally.vot
Output  05_Figures/step6e_pnlf_mask_vs_clean.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from scipy.optimize import differential_evolution, curve_fit

warnings.simplefilter("ignore", AstropyWarning)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()
t = Table.read(cfg.out("step6a_pnlf.vot"), format="votable")
c = Table.read(cfg.out("step5d_criteria_tally.vot"), format="votable")

M = np.array(t["M_radio"], dtype=float)
idt = np.array([str(x) for x in t["RP_ID"]])
idc = {str(k): i for i, k in enumerate(c["RP_ID"])}
J = np.array([idc.get(k, -1) for k in idt])
sp = np.array([str(c["sp_class"][j]) if j >= 0 else "no_fit" for j in J])
sp[~np.isin(sp, ["thermal", "uncertain", "steep"])] = "no_fit"

fin = np.isfinite(M)
N_TOT = int(fin.sum())

BIN = cfg["pnlf"]["bin_width_mag"]
MLIM = cfg["pnlf"]["completeness_limit_mag"]   # 5 sigma completeness limit
MCEIL = cfg.ceiling_magnitude()                # physical flux ceiling, Step 4c
ALO, AHI = -4.4, -3.7   # the old hand-chosen bright-bin mask, panel B only

COL = {"thermal": "#1f6fb4", "uncertain": "#e59a1e",
       "steep": "#c0392b", "no_fit": "0.72"}
ORDER = ["thermal", "uncertain", "no_fit", "steep"]


def ciardullo(m, N, Ms):
    v = N * np.exp(cfg["pnlf"]["ciardullo_alpha"] * (m - Ms)) * (1 - np.exp(cfg["pnlf"]["ciardullo_beta"] * (Ms - m)))
    return np.where(m > Ms, np.maximum(v, 1e-4), 1e-4)


# One binning for every panel, so the three are directly comparable.
edges = np.arange(np.floor(M[fin].min() / BIN) * BIN,
                  np.ceil(M[fin].max() / BIN) * BIN + BIN, BIN)
counts, _ = np.histogram(M[fin], bins=edges)
centres = 0.5 * (edges[:-1] + edges[1:])
nz = counts > 0
mx, my = centres[nz], counts[nz].astype(float)

stack = {}
for s in ORDER:
    h, _ = np.histogram(M[fin & (sp == s)], bins=edges)
    stack[s] = h[nz].astype(float)


def fit(mask):
    x, y = mx[mask], my[mask]
    ye = np.sqrt(y)
    ye[ye == 0] = 1.0
    lb = [0.0, x.min() - 5.0]
    ub = [1e6, x.min() - 0.05]
    r = lambda p: np.sum(((y - ciardullo(x, *p)) / ye) ** 2)
    de = differential_evolution(r, bounds=list(zip(lb, ub)), seed=cfg["pnlf"]["fit_seed"],
                                maxiter=9000, tol=1e-12, popsize=35)
    po, _ = curve_fit(ciardullo, x, y, p0=de.x, bounds=(lb, ub),
                      sigma=ye, absolute_sigma=True, maxfev=200000)
    chi = np.sum(((y - ciardullo(x, *po)) / ye) ** 2)
    return po, chi + 2 * len(po), chi / max(len(y) - len(po), 1)


mask_A = mx <= MLIM
mask_B = (mx <= MLIM) & ~((mx > ALO) & (mx < AHI))
mask_C = (mx <= MLIM) & (mx >= MCEIL)

poA, aicA, rcA = fit(mask_A)
poB, aicB, rcB = fit(mask_B)
poC, aicC, rcC = fit(mask_C)

# Panel A is also the answer to "what happens with the over-ceiling objects
# left in the fit", so it is written out as a number and not only drawn.
from astropy.table import Table as _Table
_Table(rows=[("no_ceiling_mask", float(poA[1]), float(rcA), int(mask_A.sum())),
             ("hand_chosen_bright_mask", float(poB[1]), float(rcB), int(mask_B.sum())),
             ("physical_flux_ceiling", float(poC[1]), float(rcC), int(mask_C.sum()))],
        names=("treatment", "M_star", "chi2_nu", "n_bins")).write(
    cfg.out("step6e_mask_comparison.csv"), format="ascii.csv", overwrite=True)

# ------------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharey=True)

# Axis range from the data, with a margin.  A hard-coded range silently cut
# the brightest bins out of an earlier version of this figure.
XLO = min(M[fin].min(), MCEIL) - 0.45
XHI = M[fin].max() + 0.45
YTOP = max(my.max() * 2.2, 100)

panels = [
    (axes[0], mask_A, poA,
     f"A.  All {N_TOT}, no mask",
     f"$M^*$ = {poA[1]:.2f}   $\\chi^2_\\nu$ = {rcA:.2f}"),
    (axes[1], mask_B, poB,
     f"B.  All {N_TOT}, hand-chosen bright-bin mask",
     f"$M^*$ = {poB[1]:.2f}   $\\chi^2_\\nu$ = {rcB:.2f}   $\\Delta$AIC = {aicB - aicC:+.1f}"),
    (axes[2], mask_C, poC,
     f"C.  All {N_TOT}, physical flux ceiling (adopted)",
     f"$M^*$ = {poC[1]:.2f}   $\\chi^2_\\nu$ = {rcC:.2f}"),
]

for ax, mask, po, ttl, sub in panels:
    # ghosted bars everywhere, solid bars only where the fit uses them
    bot = np.zeros(len(mx))
    for s in ORDER:
        ax.bar(mx, stack[s], bottom=bot, width=BIN * 0.92, color=COL[s],
               lw=0, alpha=0.30)
        bot += stack[s]
    bot = np.zeros(int(mask.sum()))
    for s in ORDER:
        ax.bar(mx[mask], stack[s][mask], bottom=bot, width=BIN * 0.92,
               color=COL[s], lw=0)
        bot += stack[s][mask]

    g = np.linspace(po[1] + 0.01, MLIM, 500)
    ax.plot(g, ciardullo(g, *po), "k-", lw=2, zorder=5)

    ax.axvline(MLIM, ls=":", color="0.4")
    ax.set_yscale("log")
    ax.set_ylim(0.5, YTOP)
    ax.set_xlim(XLO, XHI)
    ax.set_title(ttl + "\n" + sub, fontsize=9.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# panel-specific annotation
axes[1].axvspan(ALO, AHI, color="k", alpha=0.07, zorder=0)
axes[2].axvline(MCEIL, ls="--", color="#c0392b", lw=1.3)
axes[2].text(MCEIL - 0.08, YTOP * 0.45, "flux ceiling", rotation=90,
             ha="right", va="top", fontsize=7.8, color="#c0392b")

axes[0].set_ylabel("$N$ per 0.3 mag bin")
axes[1].set_xlabel("$M_{\\rm radio}$ / mag")

axes[0].legend(handles=[Patch(facecolor=COL[s], label=s.replace("_", " "))
                        for s in ORDER],
               fontsize=7.5, frameon=False, loc="upper left")
axes[1].legend(handles=[Patch(facecolor="0.55", alpha=0.35,
                              label="Masked bins (plotted, not fitted)")],
               fontsize=7.5, frameon=False, loc="upper left")
axes[2].legend(handles=[plt.Line2D([], [], color="k", lw=2,
                                   label="Ciardullo canonical")],
               fontsize=7.5, frameon=False, loc="upper left")

fig.suptitle("Radio PNLF — how the bright-end treatment changes $M^*$",
             fontsize=12.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(cfg.fig("step6e_pnlf_mask_vs_clean.pdf"), bbox_inches="tight")

print(f"sample: {N_TOT} sources, {len(mx)} populated bins")
print(f"  A  no mask               M*={poA[1]:7.3f}  chi2red={rcA:5.2f}  bins={int(mask_A.sum())}")
print(f"  B  hand-chosen mask      M*={poB[1]:7.3f}  chi2red={rcB:5.2f}  bins={int(mask_B.sum())}")
print(f"  C  physical ceiling      M*={poC[1]:7.3f}  chi2red={rcC:5.2f}  bins={int(mask_C.sum())}")
print(f"  x-range {XLO:.2f} to {XHI:.2f}  (data {M[fin].min():.2f} to {M[fin].max():.2f})")
print("saved step6e_pnlf_mask_vs_clean.pdf")
