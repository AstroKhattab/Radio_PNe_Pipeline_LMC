"""Step 05b: verify combined spectral indices against the MeerKAT catalogue fit.

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
INPUT = os.path.join(BASE, "03_Outputs", "step05a_spectral_index.vot")
OUT = os.path.join(BASE, "04_Figures", "Inspect", "step05b_spectral_consistency.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

t = Table.read(INPUT, format="votable")
combined = np.ma.asarray(t["alpha_combined"], dtype=float).filled(np.nan)
catalogue = np.ma.asarray(t["alpha_meerkat_catalogue"], dtype=float).filled(np.nan)
has_askap = np.asarray(t["alpha_has_askap"], dtype=int) == 1
valid = np.isfinite(combined) & np.isfinite(catalogue)
spectral_class = np.asarray(t["sp_class"], dtype=str)
reliable = valid & (spectral_class != "no_reliable_alpha")
delta = combined[valid] - catalogue[valid]
median = float(np.nanmedian(delta))
mad = float(1.4826 * np.nanmedian(np.abs(delta - median)))
rmse = float(np.sqrt(np.nanmean(delta ** 2)))
reliable_delta = combined[reliable] - catalogue[reliable]
reliable_median = float(np.nanmedian(reliable_delta))
reliable_mad = float(1.4826 * np.nanmedian(np.abs(reliable_delta - reliable_median)))
reliable_rmse = float(np.sqrt(np.nanmean(reliable_delta ** 2)))

with PdfPages(OUT) as pdf:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    vmin = min(np.nanmin(combined[valid]), np.nanmin(catalogue[valid])) - 0.15
    vmax = max(np.nanmax(combined[valid]), np.nanmax(catalogue[valid])) + 0.15
    ax.scatter(catalogue[valid & ~reliable], combined[valid & ~reliable],
               s=28, alpha=0.45, color="#999999", marker="x",
               label=f"Rejected by fit-quality rules (N={int((valid & ~reliable).sum())})")
    for mask, color, label, marker in [
        (reliable & ~has_askap, "#2166AC", "Reliable: MeerKAT channels only", "o"),
        (reliable & has_askap, "#E67E22", "Reliable: MeerKAT + ASKAP 888 MHz", "s"),
    ]:
        ax.scatter(catalogue[mask], combined[mask], s=34, alpha=0.78,
                   color=color, marker=marker, edgecolor="white", linewidth=0.4,
                   label=f"{label} (N={int(mask.sum())})")
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="#333333", label="1:1")
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_xlabel("MeerKAT catalogue spectral index")
    ax.set_ylabel("Combined refit spectral index")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.text(0.03, 0.97,
            f"Reliable N={reliable.sum()}\nmedian Δα={reliable_median:+.3f}\n"
            f"MAD={reliable_mad:.3f}\nRMSE={reliable_rmse:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb"))
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    bins_delta = np.arange(-1.55, 1.60, 0.10)
    rejected_delta = combined[valid & ~reliable] - catalogue[valid & ~reliable]
    ax1.hist(rejected_delta, bins=bins_delta, color="#BDBDBD", alpha=0.8,
             edgecolor="white", linewidth=0.5, label="Rejected by quality rules")
    ax1.hist(reliable_delta, bins=bins_delta, color="#457B9D", alpha=0.85,
             edgecolor="white", linewidth=0.5, label="Reliable spectra")
    ax1.axvline(0, color="#333333", linestyle="--")
    ax1.axvline(reliable_median, color="#C23B33", linewidth=1.8,
                label=f"reliable median={reliable_median:+.3f}")
    ax1.set_xlabel("Δα = combined - MeerKAT catalogue")
    ax1.set_ylabel("Number of sources")
    ax1.legend()

    order = np.argsort(np.abs(delta))[::-1][:12]
    valid_rows = np.where(valid)[0]
    labels = [str(t["RP_ID"][valid_rows[i]]).strip() or f"row {valid_rows[i]}" for i in order]
    values = delta[order]
    y = np.arange(len(order))
    ax2.barh(y, values, color=np.where(values >= 0, "#E67E22", "#2166AC"))
    ax2.set_yticks(y, labels); ax2.invert_yaxis(); ax2.axvline(0, color="#333333", linewidth=0.8)
    ax2.set_xlabel("Δα")
    ax2.set_title("Largest absolute differences for manual review")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

print(f"Step 05b: all comparisons N={valid.sum()}, reliable N={reliable.sum()}, "
      f"reliable median Δalpha={reliable_median:+.3f}, MAD={reliable_mad:.3f}, "
      f"RMSE={reliable_rmse:.3f}")
print(OUT)
