"""
Flux-ceiling criterion for the radio PN candidates.

A planetary nebula at the LMC distance cannot exceed a maximum observable flux
density, because its intrinsic radio luminosity is bounded. Filipovic et al.
(2009) placed NGC 7027, the most luminous known Galactic PN, at the LMC distance
and obtained 2.2 mJy at 4.8 GHz. Free-free emission from an optically thin shell
is close to flat (alpha = -0.1), so that value carries across to the MeerKAT band
essentially unchanged; adopting it at 1.295 GHz is marginally conservative, since
a flat spectrum would put the same source nearer 2.5 mJy here.

Input:  03_Outputs/step4b_mir_radio_ratio.vot
Output: 03_Outputs/step4c_flux_ceiling.vot
        05_Figures/step4c_flux_ceiling.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

BASE = os.environ.get("PNBASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "05_Figures")

IN_FILE = os.path.join(OUTS, "step4b_mir_radio_ratio.vot")
OUT_VOT = os.path.join(OUTS, "step4c_flux_ceiling.vot")
OUT_PDF = os.path.join(FIGS, "step4c_flux_ceiling.pdf")

CEILING_MJY = 2.2      # Filipovic et al. 2009, NGC 7027 placed at the LMC distance
DIST_KPC = 49.59       # Pietrzynski et al. 2019
NGC7027_JY = 1.543     # Zijlstra et al. 2008, 1.465 GHz integrated
NGC7027_KPC = 0.98     # Zijlstra et al. 2008
SMP83_MJY = 1.36       # Pennock et al. 2021, brightest LMC PN detected to date

COLOURS = {"thermal": "#1f6fb4", "uncertain": "#e0a020",
           "steep": "#c0392b", "no_fit": "#b8bec8"}


def text_col(table, name):
    return np.array([str(v).strip() for v in table[name]])


def numeric(table, name):
    col = table[name]
    values = np.ma.getdata(col).astype(float).copy()
    values[np.ma.getmaskarray(col)] = np.nan
    return values


print("step4c: applying the flux ceiling")
cat = Table.read(IN_FILE, format="votable")
flux = numeric(cat, "mkt_int_flux_Jy") * 1e3
measured = np.isfinite(flux) & (flux > 0)

# Sources with no 1.295 GHz measurement are not scored either way; the criterion
# is simply unavailable for them, which is not the same as failing it.
verdict = np.full(len(cat), "not_measured", dtype="U13")
verdict[measured & (flux <= CEILING_MJY)] = "below_ceiling"
verdict[measured & (flux > CEILING_MJY)] = "above_ceiling"

cat.add_column(Column(np.round(flux, 5), name="flux_1295_mJy", unit="mJy"))
cat.add_column(Column(np.full(len(cat), CEILING_MJY), name="flux_ceiling_mJy", unit="mJy"))
cat.add_column(Column(verdict, name="flux_ceiling_result"))
cat.add_column(Column((verdict == "below_ceiling").astype("i2"), name="flux_ceiling_pass"))

cat.meta["flux_ceiling_mJy"] = CEILING_MJY
cat.meta["flux_ceiling_basis"] = ("NGC 7027 scaled to the LMC by Filipovic et al. 2009; "
                                  "free-free is near-flat so the 4.8 GHz value is adopted at 1.295 GHz")

n_above = int((verdict == "above_ceiling").sum())
n_below = int((verdict == "below_ceiling").sum())
n_none = int((verdict == "not_measured").sum())
print(f"  ceiling {CEILING_MJY} mJy: {n_below} below, {n_above} above, {n_none} without a measurement")

names = text_col(cat, "Name")
ids = text_col(cat, "RP_ID")
sp = text_col(cat, "sp_class") if "sp_class" in cat.colnames else np.full(len(cat), "")
for i in np.where(verdict == "above_ceiling")[0]:
    label = names[i] if names[i] else ids[i]
    print(f"    rejected: {label:12s} {flux[i]:6.2f} mJy  {sp[i]}")

cat.write(OUT_VOT, format="votable", overwrite=True)
print(f"  wrote {os.path.basename(OUT_VOT)}")

# Spectral class for the colouring.  The recut table is produced later in the
# chain, so this step normally has to use the class from Step 4a; the earlier
# code looked only for the recut file and, not finding it, coloured every point
# as "no fit", which left the right-hand panel empty.
recut_path = os.path.join(OUTS, "step5b_spectral_index_recut.vot")
if os.path.exists(recut_path):
    sp_class = text_col(Table.read(recut_path, format="votable"), "sp_class_recut")
elif "sp_class" in cat.colnames:
    sp_class = text_col(cat, "sp_class")
    sp_class[sp_class == "no_reliable_alpha"] = "no_fit"
else:
    sp_class = np.full(len(cat), "no_fit")

ngc7027_at_lmc = NGC7027_JY * (NGC7027_KPC / DIST_KPC) ** 2 * 1e3

fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9.2, 3.6))

bins = np.logspace(np.log10(0.008), np.log10(5.0), 32)
bottom = np.zeros(len(bins) - 1)
for cls in ("no_fit", "uncertain", "thermal", "steep"):
    sel = measured & (sp_class == cls)
    counts, _ = np.histogram(flux[sel], bins=bins)
    ax_hist.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom, align="edge",
                color=COLOURS[cls], edgecolor="white", linewidth=0.3,
                label=f"{cls} ({int(sel.sum())})")
    bottom += counts

for value, style in ((ngc7027_at_lmc, ":"), (SMP83_MJY, "-."), (CEILING_MJY, "-")):
    ax_hist.axvline(value, color="0.15", linestyle=style, linewidth=1.1)
ax_hist.axvspan(CEILING_MJY, 5.0, color="#c0392b", alpha=0.09)
ax_hist.annotate(f"NGC 7027\nat LMC\n{ngc7027_at_lmc:.2f}", xy=(ngc7027_at_lmc, 0.97),
                 xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=6.5, color="0.3")
ax_hist.annotate(f"SMP LMC 83\n{SMP83_MJY:.2f}", xy=(SMP83_MJY, 0.63),
                 xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=6.5, color="0.3")
ax_hist.annotate(f"ceiling {CEILING_MJY} mJy", xy=(CEILING_MJY, 0.97),
                 xycoords=("data", "axes fraction"), ha="left", va="top", fontsize=7, color="#8e2820")
ax_hist.set_xscale("log")
ax_hist.set_xlim(0.008, 5.0)
ax_hist.set_xlabel(r"$S_{\rm 1.295\,GHz}$ (mJy)")
ax_hist.set_ylabel("Number of sources")
ax_hist.legend(fontsize=6.5, loc="upper left", frameon=False)
ax_hist.text(0.02, 1.02, "(a)", transform=ax_hist.transAxes, fontweight="bold", fontsize=8)

alpha = numeric(cat, "alpha_combined") if "alpha_combined" in cat.colnames else np.full(len(cat), np.nan)
has_alpha = measured & np.isfinite(alpha)
for cls in ("thermal", "uncertain", "steep"):
    sel = has_alpha & (sp_class == cls)
    ax_scatter.scatter(flux[sel], alpha[sel], s=13, color=COLOURS[cls],
                       edgecolor="white", linewidth=0.3, label=cls)
ax_scatter.axvline(CEILING_MJY, color="0.15", linewidth=1.1)
ax_scatter.axvspan(CEILING_MJY, 5.0, color="#c0392b", alpha=0.09)
ax_scatter.axhline(-0.5, color="0.55", linestyle="--", linewidth=0.9)
ax_scatter.axhline(-0.2, color="0.55", linestyle="--", linewidth=0.9)
ax_scatter.set_xscale("log")
ax_scatter.set_xlim(0.008, 5.0)
ax_scatter.set_xlabel(r"$S_{\rm 1.295\,GHz}$ (mJy)")
ax_scatter.set_ylabel(r"combined spectral index $\alpha$")
ax_scatter.legend(fontsize=6.5, loc="lower left", frameon=False)
ax_scatter.text(0.02, 1.02, "(b)", transform=ax_scatter.transAxes, fontweight="bold", fontsize=8)

for axis in (ax_hist, ax_scatter):
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PDF)
plt.close()
print(f"  wrote {os.path.basename(OUT_PDF)}")
