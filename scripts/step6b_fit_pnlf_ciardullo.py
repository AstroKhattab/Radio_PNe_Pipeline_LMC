"""
step6b_fit_pnlf_ciardullo.py
============================
Radio PNLF fits with the Ciardullo et al. (1989) analytic PNLF.

Reads step6a_pnlf.vot produced by Step 07a.

Three Ciardullo fits with increasing parameter freedom:
  Fit 1 (black) : alpha=0.307, beta=3.0 fixed  (canonical optical)
  Fit 2 (blue)  : alpha free,  beta=3.0 fixed
  Fit 3 (red)   : alpha free,  beta free (>= 1.0)


Excluded bins (open circles, not fitted), and nothing else:
  - bins brighter than the Step 4c flux ceiling
  - bins fainter than the 5 sigma completeness limit

Output
------
    05_Figures/step6b_pnlf_ciardullo_fits.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from scipy.optimize import curve_fit, differential_evolution

from _support.plot_style import apply_paper_style

warnings.simplefilter("ignore", AstropyWarning)
apply_paper_style()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()
L_REF = cfg["pnlf"]["reference_luminosity_cgs"]
BIN_W = cfg["pnlf"]["bin_width_mag"]
M_LIM = cfg["pnlf"]["completeness_limit_mag"]
M_CEIL = cfg.ceiling_magnitude()      # the Step 4c flux ceiling as a magnitude
S_MLIM_UJY = cfg.completeness_flux_ujy()
RMS_MEERKAT = cfg["meerkat"]["median_rms_ujy"]
CIAR_ALPHA = cfg["pnlf"]["ciardullo_alpha"]
CIAR_BETA = cfg["pnlf"]["ciardullo_beta"]
SEED = cfg["pnlf"]["fit_seed"]
np.random.seed(SEED)

INPUT = cfg.out("step6a_pnlf.vot")
OUT   = cfg.fig("step6b_pnlf_ciardullo_fits.pdf")

print("step6b: fitting the Ciardullo PNLF")

# Load
t = Table.read(INPUT, format="votable")
print(f"\n[1] {len(t)} catalogue rows from step6a_pnlf.vot")

if "log_radio_lum" in t.colnames:
    log_lum = np.array(t["log_radio_lum"], dtype=float)
elif "log_lum_wHz" in t.colnames:
    log_lum = np.array(t["log_lum_wHz"], dtype=float) + 7.0
else:
    raise ValueError("No luminosity column found in step6a_pnlf.vot")

pnlf_included = np.asarray(t["pnlf_included"], dtype=int) == 1
askap_only = (
    (np.asarray(t["det_askap"], dtype=int) == 1) &
    (np.asarray(t["det_meerkat"], dtype=int) == 0)
)
# every MeerKAT detection is plotted; the bin-level masks below decide
# which of them the fit actually uses
valid = np.isfinite(log_lum) & (log_lum > 10) & ~askap_only
if np.any(valid & askap_only):
    raise ValueError("ASKAP-only sources must not enter the MeerKAT PNLF")
M_all   = -2.5 * (log_lum[valid] - np.log10(L_REF))
N_src   = valid.sum()
print(f"    Valid: {N_src}   M_radio: {M_all.min():.2f} to {M_all.max():.2f}")
print(f"    ASKAP-only explicitly excluded: {int(np.sum(askap_only & ~valid))}")
print(f"\n    Completeness: M_lim={M_LIM}  ->  "
      f"S={S_MLIM_UJY:.1f} muJy  "
      f"~ {S_MLIM_UJY/RMS_MEERKAT:.1f}sigma  "
      f"(rms~{RMS_MEERKAT:.0f} muJy/beam)")

# Bin
edges = np.arange(np.floor(M_all.min()/BIN_W)*BIN_W,
                  np.ceil (M_all.max()/BIN_W)*BIN_W + BIN_W, BIN_W)
cnts, _ = np.histogram(M_all, bins=edges)
ctrs    = 0.5*(edges[:-1]+edges[1:])
nz = cnts > 0
mx  = ctrs[nz];  my  = cnts[nz].astype(float)
mye = np.sqrt(my); mye[mye==0] = 1.0

# Classify bins
# Two masks, and nothing else.  Bins brighter than the physical ceiling
# (the 2.2 mJy ceiling as a magnitude) and bins fainter than the
# 5-sigma completeness limit are excluded from the fit but still drawn, as
# open symbols, so the reader can see what was left out.  The hand-picked
# 'anomalous' window that used to sit between -4.4 and -3.7 is gone.
too_bright = mx < M_CEIL
incompl    = mx > M_LIM
fit_m      = ~too_bright & ~incompl

mx_f = mx[fit_m];  my_f = my[fit_m];  mye_f = mye[fit_m]
mx_o = mx[~fit_m]; my_o = my[~fit_m]; mye_o = mye[~fit_m]

# The bright cutoff M* must be anchored on the data the fit actually uses.
# Anchoring it on the brightest source in the sample would let the objects
# above the flux ceiling - which are shown but deliberately not fitted - drag
# M* out to their own magnitude and flatten the fit.
M_fit_bright = mx_f.min()
Ms_max = M_fit_bright - 0.05
Ms_min = M_fit_bright - 5.0

print(f"\n[2] Bins in fit: {fit_m.sum()}   "
      f"(all beyond the completeness limit: {incompl.sum()})")
print(f"    M* search range: [{Ms_min:.2f}, {Ms_max:.2f}]")

# Ciardullo function
def ciar(m, N0, Mstar, alpha, beta):
    """
    Ciardullo (1989) PNLF:
      N(M) = N0 * exp(alpha*(M-M*)) * (1 - exp(beta*(M*-M)))
    Rising from N=0 at M* toward fainter magnitudes.
    alpha : faint-end slope   (optical canonical = 0.307)
    beta  : bright-end cutoff (optical canonical = 3.0)
    """
    val = N0 * np.exp(alpha*(m-Mstar)) * (1.0 - np.exp(beta*(Mstar-m)))
    return np.where(m > Mstar, np.maximum(val, 1e-4), 1e-4)

def fit_ciar(alpha_fix, beta_fix, alpha_free=False, beta_free=False):
    if not alpha_free and not beta_free:
        def model(m, N0, Mstar): return ciar(m, N0, Mstar, alpha_fix, beta_fix)
        lb=[0,Ms_min]; ub=[1e6,Ms_max]; npar=2
        lbl=f"$\\alpha={alpha_fix}$, $\\beta={beta_fix:.0f}$ (fixed)"
    elif alpha_free and not beta_free:
        def model(m, N0, Mstar, alpha): return ciar(m, N0, Mstar, alpha, beta_fix)
        lb=[0,Ms_min,0.05]; ub=[1e6,Ms_max,8.0]; npar=3
        lbl=f"$\\alpha$ free, $\\beta={beta_fix:.0f}$ (fixed)"
    else:
        def model(m, N0, Mstar, alpha, beta): return ciar(m, N0, Mstar, alpha, beta)
        lb=[0,Ms_min,0.05,1.0]; ub=[1e6,Ms_max,8.0,30.0]; npar=4
        lbl="$\\alpha$ free, $\\beta$ free ($\\beta\\geq1$)"

    def resid(p): return np.sum(((my_f-model(mx_f,*p))/mye_f)**2)
    try:
        de = differential_evolution(resid, bounds=list(zip(lb,ub)), seed=42,
                                    maxiter=10000, tol=1e-12, popsize=40,
                                    mutation=(0.4,1.8))
        popt,_ = curve_fit(model, mx_f, my_f, p0=de.x, bounds=(lb,ub),
                            sigma=mye_f, absolute_sigma=True, maxfev=200000)
        yf  = model(mx_f, *popt)
        r2  = 1 - np.sum((my_f-yf)**2)/np.sum((my_f-my_f.mean())**2)
        c2r = np.sum(((my_f-yf)/mye_f)**2)/(len(my_f)-npar)
        return popt, model, r2, c2r, lbl, True
    except Exception as e:
        print(f"    FAILED ({lbl}): {e}")
        return None,None,None,None,lbl,False

# Run fits
print("\n[3] Fitting Ciardullo function ...")

p1,m1,r1,c1,l1,ok1 = fit_ciar(0.307,3.0,False,False)
if ok1: print(f"    Fit 1 canonical : M*={p1[1]:.3f}  "
              f"R2={r1:.3f}  red-chi2={c1:.3f}")

p2,m2,r2_,c2,l2,ok2 = fit_ciar(0.307,3.0,True,False)
if ok2: print(f"    Fit 2 a free    : M*={p2[1]:.3f}  a={p2[2]:.3f}  "
              f"R2={r2_:.3f}  red-chi2={c2:.3f}")

p3,m3,r3,c3,l3,ok3 = fit_ciar(0.307,3.0,True,True)
if ok3: print(f"    Fit 3 a,b free  : M*={p3[1]:.3f}  a={p3[2]:.3f}  "
              f"b={p3[3]:.3f}  R2={r3:.3f}  red-chi2={c3:.3f}")

# Figure
print("\n[4] Generating figure ...")

COLORS = ["black", "#2166AC", "#D6604D"]
fits   = [(p1,m1,r1,l1,ok1),(p2,m2,r2_,l2,ok2),(p3,m3,r3,l3,ok3)]

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor("white")

# Reduced chi-square rather than R-squared: the three fits differ in how many
# parameters are free, and R-squared cannot be compared across them.
for (p,m,r,l,ok), col, cc in zip(fits, COLORS, [c1,c2,c3]):
    if not ok: continue
    mf = np.linspace(p[1]+0.01, M_LIM, 600)
    yf = np.where(m(mf,*p) > 0.15, m(mf,*p), np.nan)
    ax.plot(mf, yf, "-", color=col, lw=2.3, zorder=7,
            label=f"{l}$\\quad \\chi^2_\\nu={cc:.2f}$")

# Filled circles — complete, used in fit
ax.errorbar(mx_f, my_f, yerr=mye_f, fmt="ko", capsize=4,
            markersize=8, elinewidth=1.3, capthick=1.3, zorder=20)
# Open circles — excluded (beyond the completeness limit)
ax.errorbar(mx_o, my_o, yerr=mye_o, fmt="o", capsize=4,
            markersize=8, mfc="white", mec="black", ecolor="black",
            elinewidth=1.0, capthick=1.0, linestyle="none", zorder=20)

# Completeness boundary with physical annotation
ax.axvline(M_LIM, color="#555", ls="--", lw=1.3, zorder=1)
ax.text(M_LIM+0.07, 0.95,
        f"$M_{{\\rm lim}} = {M_LIM}$\n"
        f"($5\\sigma \\approx {S_MLIM_UJY:.0f}\\,\\mu$Jy)",
        fontsize=9.5, color="#444", ha="left", va="top",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

# Axes — NO TITLE
ax.set_xlabel(r"$M_{\rm radio}\ /\ \rm mag$", fontsize=13)
ax.set_ylabel("Number of objects", fontsize=13)
ax.set_yscale("log")
ax.set_xlim(M_all.min()-0.4, M_all.max()+0.4)
ax.set_ylim(0.4, my.max()*6)
ax.tick_params(labelsize=11)
ax.spines[["top","right"]].set_visible(False)
ax.legend(fontsize=9.5, framealpha=0.95, loc="upper left",
          handlelength=2.5, labelspacing=0.5, borderpad=0.6)

# Compact parameter box — bottom right
pbox = [f"$N = {N_src}$ sources"]
if ok1: pbox.append(f"Black: $M^*={p1[1]:.2f}$")
if ok2: pbox.append(f"Blue: $M^*={p2[1]:.2f}$,  $\\alpha={p2[2]:.3f}$")
if ok3: pbox.append(f"Red: $M^*={p3[1]:.2f}$,  "
                    f"$\\alpha={p3[2]:.3f}$,  $\\beta={p3[3]:.2f}$")
ax.text(0.98, 0.03, "\n".join(pbox), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  alpha=0.93, ec="#cccccc", lw=0.7))

plt.tight_layout()
plt.savefig(OUT, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    -> {OUT}")

print("\nstep6b complete")
print(f"  M*_radio = {p1[1]:.3f} mag  (canonical fit)")
print(f"  L*_radio = 10^{np.log10(L_REF * 10**(-p1[1]/2.5)):.2f} erg/s/Hz")
