"""
step04c_mir_radio_ratio.py
==========================
Step 4c - Second criterion: the mid-infrared to radio flux ratio.

An ionised nebula emits free-free radiation in the radio and reprocessed dust
emission in the mid-infrared, and the two scale together.  Cohen et al. (2011)
turned that into a discriminant: for Galactic PNe the ratio of the 8 micron
flux to the radio flux near 1 GHz clusters around 4.7 +/- 1.1, while H II
regions sit an order of magnitude higher and background radio galaxies, which
have radio emission but almost no 8 micron counterpart, sit far below.

Frequencies
-----------
Cohen et al. never defined the ratio at 1.295 GHz.  They combined MGPS-2 at
843 MHz with NVSS at 1.4 GHz and treated the pair as a single "near 1 GHz"
measurement.  We therefore compute the ratio at each frequency we actually
measured rather than scaling anything:

    ASKAP-EMU            888 MHz
    MeerKAT channel 4    996.6 MHz   closest to the Cohen convention
    MeerKAT broadband   1295 MHz

The adopted value uses channel 4 where it exists, then ASKAP, then the
broadband flux, and each source records which frequency its ratio came from.

The 8 micron flux comes from the IRAC magnitude through
F(8 um) = 64.13 x 10^(-m/2.5) Jy, the zero point of Reach et al. (2005).

A caveat worth stating plainly
------------------------------
The 4.7 calibration is Galactic.  Measurements in the Magellanic Clouds put
the same ratio higher - Filipovic et al. (2009) find 9 +/- 2 and Leverenz
et al. (2017) find 11.9 - which is what you would expect at lower metallicity
with less dust per unit ionised gas.  We keep the Cohen screen as published so
that our numbers stay comparable with the Galactic literature, but the
Magellanic band is drawn on the figure alongside it, and how the sample moves
between the two is discussed in the paper.  Nothing in the classification
depends on which band is adopted; the two are reported side by side.

Screen
------
    ratio < 0.5          radio excess; argues against a PN
    0.5 <= ratio <= 10   PN-like
    10 < ratio < 20      intermediate
    ratio >= 20          H II region-like

Outputs
-------
    03_Outputs/step04c_mir_radio_ratio.vot
    03_Outputs/step04c_mir_summary.csv
    04_Figures/step04c_mir_radio_ratio.pdf

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
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

IN_FILE = os.path.join(OUTS, "step04b_spectral_index.vot")
MKT5_FILE = os.path.join(DATA, "MeerKAT_Final_5sigma.vot")
OUT_TABLE = os.path.join(OUTS, "step04c_mir_radio_ratio.vot")
OUT_SUMMARY = os.path.join(OUTS, "step04c_mir_summary.csv")
OUT_FIG = os.path.join(FIGS, "step04c_mir_radio_ratio.pdf")

ZP_8UM_JY = 64.13        # Reach et al. 2005, IRAC channel 4 zero point
COHEN_MEDIAN = 4.7       # Cohen et al. 2011, Galactic PNe
COHEN_SIGMA = 1.1
MC_LOW, MC_HIGH = 7.0, 13.9   # Filipovic 2009 (9 +/- 2), Leverenz 2017 (11.9)
SCREEN_LOW = 0.5
SCREEN_HIGH = 10.0
HII_LOW = 20.0


def ratio_and_error(f_ir, e_ir, f_radio, e_radio):
    """MIR/radio ratio with errors added in quadrature, fractionally."""
    r = np.full(len(f_ir), np.nan)
    er = np.full(len(f_ir), np.nan)
    ok = np.isfinite(f_ir) & (f_ir > 0) & np.isfinite(f_radio) & (f_radio > 0)
    r[ok] = f_ir[ok] / f_radio[ok]
    ok2 = ok & np.isfinite(e_ir) & np.isfinite(e_radio) & (e_radio > 0)
    er[ok2] = r[ok2] * np.sqrt((e_ir[ok2] / f_ir[ok2]) ** 2
                               + (e_radio[ok2] / f_radio[ok2]) ** 2)
    return r, er


print("=" * 60)
print("  Step 4c  -  mid-infrared to radio ratio")
print("=" * 60)

det = Table.read(IN_FILE, format="votable")
n = len(det)
print(f"\ndetections : {n}")

mag8 = np.asarray(det["irac8_mag"], dtype=float)
emag8 = np.asarray(det["e_irac8_mag"], dtype=float)
f_ir = ZP_8UM_JY * 10.0 ** (-mag8 / 2.5) * 1e3          # mJy
e_ir = f_ir * np.log(10.0) / 2.5 * emag8
print(f"with IRAC 8 um photometry : {int(np.isfinite(f_ir).sum())}")

# ---------------------------------------------------- radio at each frequency
mkt5 = Table.read(MKT5_FILE, format="votable")
row5 = np.asarray(det["mkt5_row"], dtype=int)
ch4 = np.full(n, np.nan)
e_ch4 = np.full(n, np.nan)
if "ch4_int_flux_1" in mkt5.colnames:
    src = np.asarray(mkt5["ch4_int_flux_1"], dtype=float) * 1e3
    esrc = np.asarray(mkt5["ch4_err_int_flux"], dtype=float) * 1e3
    have = row5 >= 0
    ch4[have] = src[row5[have]]
    e_ch4[have] = esrc[row5[have]]
ch4[~(ch4 > 0)] = np.nan

ask = np.asarray(det["askap_int_mJy"], dtype=float)
e_ask = np.asarray(det["askap_e_int_mJy"], dtype=float)
bb = np.asarray(det["flux_1295_mJy"], dtype=float)
e_bb = np.asarray(det["e_flux_1295_mJy"], dtype=float)

r_ch4, er_ch4 = ratio_and_error(f_ir, e_ir, ch4, e_ch4)
r_ask, er_ask = ratio_and_error(f_ir, e_ir, ask, e_ask)
r_bb, er_bb = ratio_and_error(f_ir, e_ir, bb, e_bb)

ratio = np.full(n, np.nan)
e_ratio = np.full(n, np.nan)
freq_used = np.full(n, "", dtype="U20")
for r, er, lab in ((r_ch4, er_ch4, "MeerKAT 996.6 MHz"),
                   (r_ask, er_ask, "ASKAP 888 MHz"),
                   (r_bb, er_bb, "MeerKAT 1295 MHz")):
    take = ~np.isfinite(ratio) & np.isfinite(r)
    ratio[take] = r[take]
    e_ratio[take] = er[take]
    freq_used[take] = lab

mir_class = np.full(n, "no_photometry", dtype="U20")
ok = np.isfinite(ratio)
mir_class[ok & (ratio < SCREEN_LOW)] = "radio_excess"
mir_class[ok & (ratio >= SCREEN_LOW) & (ratio <= SCREEN_HIGH)] = "PN_like"
mir_class[ok & (ratio > SCREEN_HIGH) & (ratio < HII_LOW)] = "intermediate"
mir_class[ok & (ratio >= HII_LOW)] = "HII_like"

det["mir_flux_mJy"] = Column(f_ir.astype("f4"), unit="mJy")
det["e_mir_flux_mJy"] = Column(e_ir.astype("f4"), unit="mJy")
det["mir_ratio_ch4"] = Column(r_ch4.astype("f4"))
det["mir_ratio_askap"] = Column(r_ask.astype("f4"))
det["mir_ratio_broadband"] = Column(r_bb.astype("f4"))
det["mir_ratio"] = Column(ratio.astype("f4"))
det["e_mir_ratio"] = Column(e_ratio.astype("f4"))
det["mir_ratio_frequency"] = Column(freq_used)
det["mir_class"] = Column(mir_class)
det.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("radio detections", n),
        ("with IRAC 8 um photometry", int(np.isfinite(f_ir).sum())),
        ("with a usable ratio", int(ok.sum())),
        ("    radio excess (< 0.5)", int((mir_class == "radio_excess").sum())),
        ("    PN-like (0.5 to 10)", int((mir_class == "PN_like").sum())),
        ("    intermediate (10 to 20)", int((mir_class == "intermediate").sum())),
        ("    H II-like (>= 20)", int((mir_class == "HII_like").sum())),
        ("no 8 um photometry", int((~ok).sum()))]
Table(rows=rows, names=("quantity", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                                overwrite=True)
print("\n" + "-" * 52)
for label, v in rows:
    print(f"  {label:<42}{v:>6}")
print("-" * 52)

med = np.nanmedian(ratio[ok])
print(f"\nmedian ratio, this sample : {med:.1f}")
print(f"  Cohen et al. 2011, Galactic PNe : {COHEN_MEDIAN} +/- {COHEN_SIGMA}")
print(f"  Magellanic measurements         : {MC_LOW:.0f} to {MC_HIGH:.0f}")
for cls in ("thermal", "steep"):
    m = ok & (np.asarray(det["sp_class"]) == cls)
    if m.sum():
        print(f"  median for {cls:<8} sources : {np.nanmedian(ratio[m]):.1f}"
              f"   (N = {int(m.sum())})")

# ====================================================================== figure
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.1))
cols = {"thermal": "#22539b", "uncertain": "#e8a33d", "steep": "#c44e52",
        "no_alpha": "0.72"}

# panel 1 - the ratio distribution against the two calibrations
ax = axes[0]
bins = np.logspace(-2, 3, 44)
ax.hist(ratio[ok], bins=bins, color="0.8", label=f"all with a ratio ({int(ok.sum())})")
ax.hist(ratio[ok & (mir_class == "PN_like")], bins=bins, color="#22539b",
        label=f"PN-like ({int((mir_class == 'PN_like').sum())})")
ax.axvspan(COHEN_MEDIAN - COHEN_SIGMA, COHEN_MEDIAN + COHEN_SIGMA,
           color="#2c7d3f", alpha=0.22, lw=0)
ax.axvline(COHEN_MEDIAN, color="#2c7d3f", lw=1.4, label="Cohen+2011 Galactic")
ax.axvspan(MC_LOW, MC_HIGH, color="#7d3fa0", alpha=0.16, lw=0)
ax.axvline(np.sqrt(MC_LOW * MC_HIGH), color="#7d3fa0", lw=1.4, ls="--",
           label="Magellanic (Filipovic+09, Leverenz+17)")
ax.axvline(med, color="k", ls=":", lw=1.3, label=f"this sample ({med:.1f})")
ax.set_xscale("log")
ax.set_xlabel(r"$F_{8\,\mu\mathrm{m}}\,/\,S_{\mathrm{radio}}$")
ax.set_ylabel("number of PNe")
ax.set_title("Mid-infrared to radio ratio", fontsize=10)
ax.legend(fontsize=7, framealpha=0.9)
ax.tick_params(labelsize=8)

# panel 2 - ratio against spectral index; the two criteria should agree
ax2 = axes[1]
alpha = np.asarray(det["alpha"], dtype=float)
spc = np.asarray(det["sp_class"])
for cls in ("no_alpha", "steep", "uncertain", "thermal"):
    m = ok & (spc == cls)
    if not m.sum():
        continue
    ax2.scatter(ratio[m], alpha[m], s=22, c=cols[cls], edgecolors="none",
                alpha=0.85, label=f"{cls} ({int(m.sum())})")
ax2.axvspan(SCREEN_LOW, SCREEN_HIGH, color="#2c7d3f", alpha=0.10, lw=0)
ax2.axvline(SCREEN_LOW, color="0.35", ls="--", lw=1.0)
ax2.axvline(SCREEN_HIGH, color="0.35", ls="--", lw=1.0)
ax2.axhline(-0.5, color="0.35", ls="--", lw=1.0)
ax2.axhline(-0.2, color="0.35", ls="--", lw=1.0)
ax2.set_xscale("log")
ax2.set_ylim(-3.0, 2.5)
ax2.set_xlabel(r"$F_{8\,\mu\mathrm{m}}\,/\,S_{\mathrm{radio}}$")
ax2.set_ylabel(r"spectral index $\alpha$")
ax2.set_title("Do the two criteria agree?", fontsize=10)
ax2.legend(fontsize=7.5, framealpha=0.9)
ax2.tick_params(labelsize=8)

# panel 3 - the ratio is a flux-flux relation; show it as one
ax3 = axes[2]
radio_used = np.where(np.isfinite(r_ch4), ch4,
                      np.where(np.isfinite(r_ask), ask, bb))
for cls in ("no_alpha", "steep", "uncertain", "thermal"):
    m = ok & (spc == cls)
    if not m.sum():
        continue
    ax3.scatter(radio_used[m], f_ir[m], s=22, c=cols[cls], edgecolors="none",
                alpha=0.85, label=cls)
xx = np.logspace(-2, 2, 50)
ax3.plot(xx, COHEN_MEDIAN * xx, color="#2c7d3f", lw=1.4,
         label=f"ratio = {COHEN_MEDIAN}")
ax3.fill_between(xx, MC_LOW * xx, MC_HIGH * xx, color="#7d3fa0", alpha=0.16,
                 lw=0, label="Magellanic band")
ax3.plot(xx, SCREEN_LOW * xx, color="0.4", ls="--", lw=0.9)
ax3.plot(xx, SCREEN_HIGH * xx, color="0.4", ls="--", lw=0.9)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlim(np.nanmin(radio_used[ok]) * 0.6, np.nanmax(radio_used[ok]) * 1.7)
ax3.set_ylim(np.nanmin(f_ir[ok]) * 0.6, np.nanmax(f_ir[ok]) * 1.7)
ax3.set_xlabel("radio flux density (mJy)")
ax3.set_ylabel(r"$F_{8\,\mu\mathrm{m}}$ (mJy)")
ax3.set_title("The ratio as a flux-flux relation", fontsize=10)
ax3.legend(fontsize=7.5, framealpha=0.9, loc="upper left")
ax3.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
