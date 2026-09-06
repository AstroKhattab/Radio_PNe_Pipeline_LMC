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
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()
IN_FILE = cfg.out("step4b_mir_radio_ratio.vot")
OUT_VOT = cfg.out("step4c_flux_ceiling.vot")
OUT_CSV = cfg.out("step4c_above_ceiling.csv")
OUT_PDF = cfg.fig("step4c_flux_ceiling.pdf")

CEILING_MJY = cfg["flux_ceiling"]["ceiling_mjy"]
DIST_KPC = cfg["distance"]["lmc_kpc"]
NGC7027_JY = cfg["flux_ceiling"]["ngc7027_jy"]
NGC7027_KPC = cfg["flux_ceiling"]["ngc7027_kpc"]
SMP83_MJY = cfg["flux_ceiling"]["smp_lmc_83_mjy"]

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
ngc7027_at_lmc = NGC7027_JY * (NGC7027_KPC / DIST_KPC) ** 2 * 1e3
above = np.where(verdict == "above_ceiling")[0]
order = above[np.argsort(-flux[above])]
pnstat = text_col(cat, "hash_pnstat")
listing = Table()
listing["identifier"] = [names[i] if names[i] else ids[i] for i in order]
listing["RP_ID"] = ids[order]
listing["S_1295_mJy"] = np.round(flux[order], 2)
listing["hash_pnstat"] = pnstat[order]
listing["sp_class"] = sp[order]
listing["S_over_S_7027"] = np.round(flux[order] / ngc7027_at_lmc, 1)
listing.write(OUT_CSV, format="ascii.csv", overwrite=True)
for row in listing:
    print(f"    rejected: {row['identifier']:12s} {row['S_1295_mJy']:6.2f} mJy  "
          f"{row['hash_pnstat']}  {row['sp_class']}  "
          f"{row['S_over_S_7027']:.1f}x NGC 7027")

cat.write(OUT_VOT, format="votable", overwrite=True)
print(f"  wrote {os.path.basename(OUT_VOT)}")

# Spectral class for the colouring, from Step 4a.
sp_class = text_col(cat, "sp_class")
sp_class[sp_class == "no_reliable_alpha"] = "no_fit"

fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(9.2, 3.6))

x_top = max(5.0, float(np.nanmax(flux)) * 1.3)
bins = np.logspace(np.log10(0.008), np.log10(x_top), 32)
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
ax_hist.axvspan(CEILING_MJY, x_top, color="#c0392b", alpha=0.09)
ax_hist.annotate(f"NGC 7027\nat LMC\n{ngc7027_at_lmc:.2f}", xy=(ngc7027_at_lmc, 0.97),
                 xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=6.5, color="0.3")
ax_hist.annotate(f"SMP LMC 83\n{SMP83_MJY:.2f}", xy=(SMP83_MJY, 0.63),
                 xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=6.5, color="0.3")
ax_hist.annotate(f"ceiling {CEILING_MJY} mJy", xy=(CEILING_MJY, 0.97),
                 xycoords=("data", "axes fraction"), ha="left", va="top", fontsize=7, color="#8e2820")
ax_hist.set_xscale("log")
ax_hist.set_xlim(0.008, x_top)
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
ax_scatter.axvspan(CEILING_MJY, x_top, color="#c0392b", alpha=0.09)
ax_scatter.axhline(-0.5, color="0.55", linestyle="--", linewidth=0.9)
ax_scatter.axhline(-0.2, color="0.55", linestyle="--", linewidth=0.9)
ax_scatter.set_xscale("log")
ax_scatter.set_xlim(0.008, x_top)
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
