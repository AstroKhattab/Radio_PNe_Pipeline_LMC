"""
step04d_flux_ceiling.py
=======================
Step 4d - Third criterion: an upper limit on how bright an LMC PN can be.

Radio luminosity is fixed by the amount of ionised gas a PN can hold, and that
has a ceiling.  Put the brightest and best studied Galactic PN, NGC 7027, at
the distance of the LMC and it would be faint:

    S(LMC) = S(NGC 7027) x (d_7027 / d_LMC)^2
           = 1.543 Jy x (0.98 / 49.59)^2
           = 0.60 mJy

Filipovic et al. (2009) did the same calculation for the Magellanic Clouds at
4.8 GHz and adopted 2.2 mJy as the practical ceiling, allowing for objects
somewhat more luminous than NGC 7027 and for the uncertainty in the scaling.
We keep their number.  Because PNe are optically thin at these frequencies
their spectra are close to flat, so a limit set at 4.8 GHz carries across to
1.295 GHz essentially unchanged - which is the only reason a single number can
be used at both frequencies.

The observational check agrees: the brightest LMC PN detected to date is
SMP LMC 83 at 1.36 mJy (Pennock et al. 2021), comfortably under the ceiling.
Cohen et al. (2011) argue independently that anything above about 5 mJy at
Magellanic distance is very unlikely to be a PN.

A source above the ceiling is not automatically rejected, but it is flagged,
and it is masked out of the luminosity function in Step 5 because a single
over-luminous object distorts the bright end of the fit.

Outputs
-------
    03_Outputs/step04d_flux_ceiling.vot
    03_Outputs/step04d_ceiling_summary.csv
    04_Figures/step04d_flux_ceiling.pdf

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

IN_FILE = os.path.join(OUTS, "step04c_mir_radio_ratio.vot")
OUT_TABLE = os.path.join(OUTS, "step04d_flux_ceiling.vot")
OUT_SUMMARY = os.path.join(OUTS, "step04d_ceiling_summary.csv")
OUT_FIG = os.path.join(FIGS, "step04d_flux_ceiling.pdf")

CEILING_MJY = 2.2        # Filipovic et al. 2009
DIST_KPC = 49.59         # Pietrzynski et al. 2019
NGC7027_JY = 1.543       # Zijlstra et al. 2008, 1.465 GHz integrated
NGC7027_KPC = 0.98
SMP83_MJY = 1.36         # Pennock et al. 2021, brightest LMC PN detected
COHEN_LIMIT_MJY = 5.0    # Cohen et al. 2011

print("=" * 60)
print("  Step 4d  -  radio flux ceiling")
print("=" * 60)

det = Table.read(IN_FILE, format="votable")
n = len(det)
flux = np.asarray(det["flux_1295_mJy"], dtype=float)
measured = np.isfinite(flux) & (flux > 0)

ngc7027_at_lmc = NGC7027_JY * (NGC7027_KPC / DIST_KPC) ** 2 * 1e3
print(f"\nNGC 7027 placed at {DIST_KPC} kpc : {ngc7027_at_lmc:.2f} mJy")
print(f"adopted ceiling                : {CEILING_MJY} mJy")
print(f"brightest LMC PN known         : {SMP83_MJY} mJy (SMP LMC 83)")

verdict = np.full(n, "not_measured", dtype="U14")
verdict[measured & (flux <= CEILING_MJY)] = "below_ceiling"
verdict[measured & (flux > CEILING_MJY)] = "above_ceiling"
above = verdict == "above_ceiling"

det["flux_ceiling_mJy"] = Column(np.full(n, CEILING_MJY, dtype="f4"), unit="mJy")
det["ceiling_class"] = Column(verdict)
det.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("radio detections", n),
        ("with a measured flux", int(measured.sum())),
        (f"below the ceiling ({CEILING_MJY} mJy)", int((verdict == "below_ceiling").sum())),
        ("above the ceiling", int(above.sum())),
        ("no measured flux", int((verdict == "not_measured").sum()))]
Table(rows=rows, names=("quantity", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                                overwrite=True)
print("\n" + "-" * 50)
for label, v in rows:
    print(f"  {label:<40}{v:>6}")
print("-" * 50)

if above.sum():
    print("\nsources above the ceiling:")
    for k in np.where(above)[0]:
        print(f"    {str(det['PN_ID'][k]):<10} {flux[k]:6.2f} mJy  "
              f"alpha = {det['alpha'][k]:6.2f}  "
              f"{str(det['sp_class'][k]):<10} "
              f"Reid: {str(det['reid_class'][k])}  "
              f"HASH: {str(det['PNstat'][k])}  "
              f"{flux[k] / ngc7027_at_lmc:.1f}x NGC 7027")

# ====================================================================== figure
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
cols = {"thermal": "#22539b", "uncertain": "#e8a33d", "steep": "#c44e52",
        "no_alpha": "0.75"}

ax = axes[0]
lo = max(np.nanmin(flux[measured]), 1e-2)
bins = np.logspace(np.log10(lo), np.log10(np.nanmax(flux[measured]) * 1.2), 34)
bottom = np.zeros(len(bins) - 1)
spc = np.asarray(det["sp_class"])
for cls in ("no_alpha", "steep", "uncertain", "thermal"):
    h, _ = np.histogram(flux[measured & (spc == cls)], bins=bins)
    ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
           color=cols[cls], label=f"{cls} ({int((measured & (spc == cls)).sum())})")
    bottom += h
ax.axvline(CEILING_MJY, color="#c44e52", lw=1.6)
ax.axvline(ngc7027_at_lmc, color="#2c7d3f", lw=1.3, ls="--")
ax.axvline(SMP83_MJY, color="k", lw=1.1, ls=":")
ax.set_xscale("log")
top = ax.get_ylim()[1]
ax.text(CEILING_MJY * 1.06, top * 0.95, f"ceiling\n{CEILING_MJY} mJy",
        fontsize=7.5, color="#c44e52", va="top")
ax.text(ngc7027_at_lmc * 0.94, top * 0.95, "NGC 7027\nat LMC", fontsize=7.5,
        color="#2c7d3f", va="top", ha="right")
ax.text(SMP83_MJY * 0.94, top * 0.55, "SMP LMC 83", fontsize=7.5, va="top",
        ha="right", rotation=90)
ax.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax.set_ylabel("number of PNe")
ax.set_title("Flux distribution against the physical ceiling", fontsize=10)
ax.legend(fontsize=7.5, framealpha=0.9)
ax.tick_params(labelsize=8)

ax2 = axes[1]
alpha = np.asarray(det["alpha"], dtype=float)
for cls in ("no_alpha", "steep", "uncertain", "thermal"):
    m = measured & (spc == cls) & np.isfinite(alpha)
    if not m.sum():
        continue
    ax2.scatter(flux[m], alpha[m], s=24, c=cols[cls], edgecolors="none",
                alpha=0.85, label=cls)
ax2.axvline(CEILING_MJY, color="#c44e52", lw=1.6)
ax2.axvline(COHEN_LIMIT_MJY, color="#c44e52", lw=1.0, ls="--")
ax2.axhline(-0.5, color="0.4", ls="--", lw=0.9)
ax2.axhline(-0.2, color="0.4", ls="--", lw=0.9)
for k in np.where(above)[0]:
    if np.isfinite(alpha[k]):
        ax2.annotate(str(det["PN_ID"][k]), (flux[k], alpha[k]),
                     textcoords="offset points", xytext=(-6, 8),
                     ha="right", fontsize=8)
ax2.set_xscale("log")
ax2.set_ylim(-3.0, 2.5)
ax2.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax2.set_ylabel(r"spectral index $\alpha$")
ax2.set_title("Sources above the ceiling", fontsize=10)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
