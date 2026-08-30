"""Step 07d: inspect the MeerKAT PNLF sample split and completeness logic.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings

import matplotlib
import numpy as np
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SCRIPT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_ROOT not in sys.path:
    sys.path.insert(0, SCRIPT_ROOT)
from _support.plot_style import apply_paper_style

warnings.filterwarnings("ignore")
apply_paper_style()

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
INPUT = os.path.join(BASE, "03_Outputs", "step07a_pnlf.vot")
OUT = os.path.join(BASE, "04_Figures", "Inspect", "step07d_pnlf_consistency.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

t = Table.read(INPUT, format="votable")
included = np.asarray(t["pnlf_included"], dtype=int) == 1
magnitude = np.ma.asarray(t["M_radio"], dtype=float).filled(np.nan)
method = np.asarray(t["mkt_detection_method"], dtype=str)
valid = included & np.isfinite(magnitude)
catalogue = valid & (method == "catalogue")
visual = valid & (method == "visual")
askap_only = np.asarray(t["det_askap"], dtype=int).astype(bool) & ~np.asarray(t["det_meerkat"], dtype=int).astype(bool)

with PdfPages(OUT) as pdf:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bins = np.arange(np.floor(np.nanmin(magnitude[valid]) / 0.3) * 0.3,
                     np.ceil(np.nanmax(magnitude[valid]) / 0.3) * 0.3 + 0.3, 0.3)
    ax.hist([magnitude[catalogue], magnitude[visual]], bins=bins, stacked=True,
            color=["#2166AC", "#E67E22"], edgecolor="white",
            label=[f"Catalogue (N={catalogue.sum()})", f"Visual (N={visual.sum()})"])
    ax.axvline(-0.5, color="#333333", linestyle="--", label="Completeness limit")
    ax.set_xlabel("MeerKAT radio absolute magnitude")
    ax.set_ylabel("Number of objects")
    ax.legend(loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.95, f"PNLF included: {valid.sum()}\nASKAP-only excluded: {askap_only.sum()}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb"))
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mask, color, label in [(catalogue, "#2166AC", "Catalogue"),
                               (visual, "#E67E22", "Visual")]:
        values = np.sort(magnitude[mask])
        if len(values):
            ax.step(values, np.arange(1, len(values) + 1), where="post",
                    color=color, linewidth=2, label=f"{label} (N={len(values)})")
    ax.axvline(-0.5, color="#333333", linestyle="--")
    ax.set_xlabel("MeerKAT radio absolute magnitude")
    ax.set_ylabel("Cumulative number")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"Step 07d: PNLF={valid.sum()}, catalogue={catalogue.sum()}, visual={visual.sum()}, ASKAP-only excluded={askap_only.sum()}")
print(OUT)
