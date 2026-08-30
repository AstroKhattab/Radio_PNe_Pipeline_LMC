"""
step07c_make_pnlf_paper_figures.py
==================================
Redraw the PNLF figures for the paper without the statistics panels and
equation boxes carried by the Step 07a/07b diagnostic versions.  The fits
are repeated here from step07a_pnlf.vot; nothing upstream is modified.

Outputs (04_Figures/):
    step07c_pnlf_model_overlay.pdf — top empirical fits + data
    step07c_pnlf_ciardullo_comparison.pdf — three Ciardullo variants

Run after Step 07a has produced step07a_pnlf.vot:
    cd ~/Desktop/PN\ LMC\ Paper
    python 02_Scripts/step07c_make_pnlf_paper_figures.py

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import skewnorm, t as student_t

warnings.filterwarnings("ignore")
apply_paper_style()
np.random.seed(42)

# Paths
BASE  = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
INPUT = os.path.join(BASE, "03_Outputs", "step07a_pnlf.vot")
FIGS  = os.path.join(BASE, "04_Figures")

# Settings
L_REF       = 1e20
BIN_W       = 0.3
M_LIM       = -0.5
M_ANOM_LO   = -4.4
M_ANOM_HI   = -3.7
# D_KPC       = 49.97
# updated 2026-08-30: Pietrzynski et al. 2019 gives 49.59 kpc (mu = 18.477); 49.97 kpc was inconsistent with the quoted modulus
D_KPC       = 49.59
# RMS_MEERKAT = 10.0
# updated 2026-08-30: median rms 11 uJy/beam from Cotton et al. 2026 / Rajabpour et al. 2026; S(M_lim) recomputed at d = 49.59 kpc
RMS_MEERKAT = 11.0
# S_MLIM_UJY  = 53.05
S_MLIM_UJY  = 53.87

print("step07c: making the PNLF paper figures")

# Load data
t = Table.read(INPUT, format="votable")
print(f"\n[1] Loaded {len(t)} sources")

if "log_radio_lum" in t.colnames:
    log_lum = np.array(t["log_radio_lum"], dtype=float)
elif "log_lum_wHz" in t.colnames:
    log_lum = np.array(t["log_lum_wHz"], dtype=float) + 7.0
else:
    raise ValueError("No luminosity column found.")

pnlf_included = np.asarray(t["pnlf_included"], dtype=int) == 1
askap_only = (
    (np.asarray(t["det_askap"], dtype=int) == 1) &
    (np.asarray(t["det_meerkat"], dtype=int) == 0)
)
valid = pnlf_included & np.isfinite(log_lum) & (log_lum > 10)
if np.any(valid & askap_only):
    raise ValueError("ASKAP-only sources must not enter the MeerKAT PNLF")
M_all = -2.5 * (log_lum[valid] - np.log10(L_REF))
N_src = valid.sum()
print(f"    Valid: {N_src}   M: {M_all.min():.2f} to {M_all.max():.2f}")
print(f"    ASKAP-only explicitly excluded: {int(np.sum(askap_only & ~valid))}")

# Binning
edges = np.arange(np.floor(M_all.min()/BIN_W)*BIN_W,
                  np.ceil(M_all.max()/BIN_W)*BIN_W + BIN_W, BIN_W)
cnts, _ = np.histogram(M_all, bins=edges)
ctrs = 0.5*(edges[:-1] + edges[1:])
nz = cnts > 0
mx = ctrs[nz]; my = cnts[nz].astype(float)
mye = np.sqrt(my); mye[mye == 0] = 1.0

anom    = (mx > M_ANOM_LO) & (mx < M_ANOM_HI)
incompl = mx > M_LIM
fit_m   = ~anom & ~incompl

mx_f = mx[fit_m];  my_f = my[fit_m];  mye_f = mye[fit_m]
mx_o = mx[~fit_m]; my_o = my[~fit_m]; mye_o = mye[~fit_m]

Ms_max = M_all.min() - 0.05
Ms_min = M_all.min() - 5.0
m_fine = np.linspace(M_all.min() - 0.5, M_all.max() + 0.5, 800)

print(f"[2] Bins fit: {fit_m.sum()}  excluded: {(~fit_m).sum()}")

# Model definitions
def f_gaussian(m, A, mu, sig):
    return A * np.exp(-0.5*((m - mu)/sig)**2)

def f_studentt(m, A, mu, sig, nu):
    sig = max(abs(sig), 0.01); nu = max(nu, 0.5)
    return A * student_t.pdf(m, nu, loc=mu, scale=sig)

def f_cauchy(m, A, mu, gamma):
    return A / (1 + ((m - mu)/gamma)**2)

def f_skewnorm(m, A, loc, sc, ask):
    return A * skewnorm.pdf(m, ask, loc=loc, scale=sc)

def f_poly2(m, a, b, c):    return a*m**2 + b*m + c
def f_poly3(m, a, b, c, d): return a*m**3 + b*m**2 + c*m + d

def f_exponential(m, a, b, c):
    return a * np.exp(b*m) + c

def f_schechter(m, phi, ms, alpha):
    dm = m - ms
    return np.maximum(phi * 10**(0.4*(alpha+1)*dm) * np.exp(-10**(0.4*dm)), 0)

def f_ciar_free(m, N0, Ms, alpha, beta):
    v = N0 * np.exp(alpha*(m - Ms)) * (1 - np.exp(beta*(Ms - m)))
    return np.where(m > Ms, np.maximum(v, 1e-4), 1e-4)

def f_ciar_canonical(m, N0, Ms):
    v = N0 * np.exp(0.307*(m - Ms)) * (1 - np.exp(3.0*(Ms - m)))
    return np.where(m > Ms, np.maximum(v, 1e-4), 1e-4)

# Generic fitter
m_peak = mx_f[np.argmax(my_f)]

def safe_fit(func, bounds, name):
    """Fit model using DE + LM refinement."""
    def resid(p):
        yp = func(mx_f, *p)
        return np.sum(((my_f - yp)/mye_f)**2)
    try:
        de = differential_evolution(resid, bounds=bounds, seed=42,
                                     maxiter=10000, tol=1e-12, popsize=40,
                                     mutation=(0.4, 1.8))
        popt, pcov = curve_fit(func, mx_f, my_f, p0=de.x,
                                bounds=([b[0] for b in bounds],
                                        [b[1] for b in bounds]),
                                sigma=mye_f, absolute_sigma=True, maxfev=200000)
        yf  = func(mx_f, *popt)
        ss_res = np.sum((my_f - yf)**2)
        ss_tot = np.sum((my_f - my_f.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        k = len(popt)
        chi2 = np.sum(((my_f - yf)/mye_f)**2)
        dof = len(mx_f) - k
        aic = chi2 + 2*k
        bic = chi2 + k*np.log(len(mx_f))
        redchi = chi2/dof if dof > 0 else np.nan
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full(k, np.nan)
        return {"popt": popt, "perr": perr, "func": func, "name": name,
                "r2": r2, "aic": aic, "bic": bic, "redchi": redchi, "k": k}
    except Exception as e:
        print(f"    FAILED: {name}: {e}")
        return None

# Fit all models
print("\n[3] Fitting all models ...")

models = [
    ("Gaussian",     f_gaussian,  [(0, 1e5), (m_peak-2, m_peak+2), (0.1, 5)]),
    ("Student-$t$",  f_studentt,  [(0, 1e5), (m_peak-2, m_peak+2), (0.1, 5), (0.5, 50)]),
    ("Cauchy",       f_cauchy,    [(0, 1e5), (m_peak-2, m_peak+2), (0.1, 5)]),
    ("Skew-normal",  f_skewnorm,  [(0, 1e6), (m_peak-3, m_peak+3), (0.1, 5), (-10, 10)]),
    ("Poly (2)",     f_poly2,     [(-50, 50), (-500, 500), (-500, 500)]),
    ("Poly (3)",     f_poly3,     [(-50, 50), (-500, 500), (-500, 500), (-500, 500)]),
    ("Exponential",  f_exponential, [(0, 1e5), (-5, 5), (-100, 100)]),
    ("Schechter",    f_schechter, [(0, 1e5), (Ms_min, Ms_max), (-3, 3)]),
    ("Ciardullo (free)", f_ciar_free, [(0, 1e6), (Ms_min, Ms_max), (0.05, 8), (1, 30)]),
]

results = []
for name, func, bounds in models:
    r = safe_fit(func, bounds, name)
    if r is not None:
        results.append(r)
        print(f"    {name:<22s}  R²={r['r2']:.4f}  AIC={r['aic']:.1f}  red-χ²={r['redchi']:.3f}")

# Also fit Ciardullo canonical separately
r_can = safe_fit(f_ciar_canonical, [(0, 1e6), (Ms_min, Ms_max)], "Ciardullo (canonical)")
if r_can:
    results.append(r_can)
    print(f"    {'Ciardullo (canonical)':<22s}  R²={r_can['r2']:.4f}  AIC={r_can['aic']:.1f}  red-χ²={r_can['redchi']:.3f}")

# Also fit Ciardullo with alpha free, beta fixed
def f_ciar_afree(m, N0, Ms, alpha):
    v = N0 * np.exp(alpha*(m - Ms)) * (1 - np.exp(3.0*(Ms - m)))
    return np.where(m > Ms, np.maximum(v, 1e-4), 1e-4)

r_af = safe_fit(f_ciar_afree, [(0, 1e6), (Ms_min, Ms_max), (0.05, 8)], "Ciardullo (free α)")
if r_af:
    results.append(r_af)
    print(f"    {'Ciardullo (free α)':<22s}  R²={r_af['r2']:.4f}  AIC={r_af['aic']:.1f}  red-χ²={r_af['redchi']:.3f}")

# Sort by AIC
results.sort(key=lambda r: r["aic"])

print(f"\n    Best by AIC: {results[0]['name']}")

#  FIGURE 1 — paper_pnlf_overlay.pdf  (top 4 empirical + data)
print("\n[4] Generating step07c_pnlf_model_overlay.pdf ...")

# Select top 4 non-Ciardullo models + best Ciardullo
top_empirical = [r for r in results if "Ciardullo" not in r["name"]][:4]
top_ciar = [r for r in results if r["name"] == "Ciardullo (canonical)"]

fig, ax = plt.subplots(figsize=(8.5, 6))
fig.patch.set_facecolor("white")

# Plot data: filled = complete, open = excluded
ax.errorbar(mx_f, my_f, yerr=mye_f, fmt="ko", capsize=4,
            markersize=7, elinewidth=1.2, capthick=1.2, zorder=20,
            label="Complete bins (fitted)")
ax.errorbar(mx_o, my_o, yerr=mye_o, fmt="o", capsize=4,
            markersize=7, mfc="white", mec="black", ecolor="black",
            elinewidth=1.0, capthick=1.0, linestyle="none", zorder=20,
            label="Excluded bins")

# Plot top empirical fits
COLORS_EMP = ["#D32F2F", "#1976D2", "#388E3C", "#F57C00"]
# added 2026-08-30: Poly(2) and Exponential are numerically coincident here,
# so colour alone cannot separate them; give each model its own dash pattern
STYLES_EMP = ["-", "--", "-.", ":"]
for i, r in enumerate(top_empirical):
    mplot = m_fine[m_fine <= M_LIM]
    yf = r["func"](mplot, *r["popt"])
    # ax.plot(mplot, np.where(yf > 0.3, yf, np.nan), "-",
    #         color=COLORS_EMP[i], lw=2.0, alpha=0.85,
    ax.plot(mplot, np.where(yf > 0.3, yf, np.nan), STYLES_EMP[i % len(STYLES_EMP)],
            color=COLORS_EMP[i], lw=2.0, alpha=0.85,
            label=f"{r['name']} ($R^2={r['r2']:.3f}$)")

# Plot canonical Ciardullo
if top_ciar:
    rc = top_ciar[0]
    mplot = m_fine[(m_fine > rc["popt"][1]+0.01) & (m_fine <= M_LIM)]
    yf = rc["func"](mplot, *rc["popt"])
    ax.plot(mplot, np.where(yf > 0.3, yf, np.nan), "--",
            color="black", lw=2.2, alpha=0.9,
            label=f"Ciardullo canonical ($R^2={rc['r2']:.3f}$)")

# Completeness limit
ax.axvline(M_LIM, color="#666", ls=":", lw=1.3, zorder=1)
ax.text(M_LIM + 0.08, 0.93,
        f"$M_{{\\rm lim}} = {M_LIM}$\n($5\\sigma \\approx {S_MLIM_UJY:.0f}\\,\\mu$Jy)",
        fontsize=9, color="#444", ha="left", va="top",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

# Format
ax.set_xlabel(r"$M_{\rm radio}$ / mag", fontsize=13)
ax.set_ylabel("$N$ (per 0.3 mag bin)", fontsize=13)
ax.set_yscale("log")
ax.set_xlim(M_all.min() - 0.5, M_all.max() + 0.5)
ax.set_ylim(0.4, my.max() * 6)
# ax.tick_params(labelsize=11, which="both", direction="in", top=True, right=True)
# changed 2026-08-30: the top and right spines are hidden below, so those ticks floated
ax.tick_params(labelsize=11, which="both", direction="in", top=False, right=False)
ax.xaxis.set_minor_locator(AutoMinorLocator(3))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8.5, framealpha=0.95, loc="upper left",
          handlelength=2.2, labelspacing=0.4)

# Source count annotation
ax.text(0.97, 0.03, f"$N = {N_src}$ sources",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"))

plt.tight_layout()
out1 = os.path.join(FIGS, "step07c_pnlf_model_overlay.pdf")
plt.savefig(out1, format="pdf", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    -> {out1}")


#  FIGURE 2 — paper_pnlf_ciardullo.pdf  (3 Ciardullo variants)
print("\n[5] Generating step07c_pnlf_ciardullo_comparison.pdf ...")

# Re-do the 3 Ciardullo fits properly
def ciar_full(m, N0, Mstar, alpha, beta):
    val = N0 * np.exp(alpha*(m - Mstar)) * (1.0 - np.exp(beta*(Mstar - m)))
    return np.where(m > Mstar, np.maximum(val, 1e-4), 1e-4)

def fit_ciar(alpha_fix, beta_fix, alpha_free=False, beta_free=False):
    if not alpha_free and not beta_free:
        def model(m, N0, Mstar): return ciar_full(m, N0, Mstar, alpha_fix, beta_fix)
        lb = [0, Ms_min]; ub = [1e6, Ms_max]; npar = 2
        lbl = f"$\\alpha={alpha_fix}$, $\\beta={beta_fix:.0f}$ (canonical)"
    elif alpha_free and not beta_free:
        def model(m, N0, Mstar, alpha): return ciar_full(m, N0, Mstar, alpha, beta_fix)
        lb = [0, Ms_min, 0.05]; ub = [1e6, Ms_max, 8.0]; npar = 3
        lbl = f"$\\alpha$ free, $\\beta={beta_fix:.0f}$ (fixed)"
    else:
        def model(m, N0, Mstar, alpha, beta): return ciar_full(m, N0, Mstar, alpha, beta)
        lb = [0, Ms_min, 0.05, 1.0]; ub = [1e6, Ms_max, 8.0, 30.0]; npar = 4
        lbl = "$\\alpha$, $\\beta$ free"

    def resid(p): return np.sum(((my_f - model(mx_f, *p))/mye_f)**2)
    try:
        de = differential_evolution(resid, bounds=list(zip(lb, ub)), seed=42,
                                     maxiter=10000, tol=1e-12, popsize=40,
                                     mutation=(0.4, 1.8))
        popt, _ = curve_fit(model, mx_f, my_f, p0=de.x, bounds=(lb, ub),
                             sigma=mye_f, absolute_sigma=True, maxfev=200000)
        yf  = model(mx_f, *popt)
        r2  = 1 - np.sum((my_f - yf)**2)/np.sum((my_f - my_f.mean())**2)
        c2r = np.sum(((my_f - yf)/mye_f)**2)/(len(my_f) - npar)
        return popt, model, r2, c2r, lbl, True
    except Exception as e:
        print(f"    FAILED: {e}")
        return None, None, None, None, lbl, False

p1, m1, r1, c1, l1, ok1 = fit_ciar(0.307, 3.0, False, False)
p2, m2, r2_, c2, l2, ok2 = fit_ciar(0.307, 3.0, True, False)
p3, m3, r3, c3, l3, ok3 = fit_ciar(0.307, 3.0, True, True)

if ok1: print(f"    Canonical : M*={p1[1]:.3f}  R²={r1:.3f}")
if ok2: print(f"    Free α    : M*={p2[1]:.3f}  α={p2[2]:.3f}  R²={r2_:.3f}")
if ok3: print(f"    Free α,β  : M*={p3[1]:.3f}  α={p3[2]:.3f}  β={p3[3]:.2f}  R²={r3:.3f}")

fig, ax = plt.subplots(figsize=(8.5, 6))
fig.patch.set_facecolor("white")

COLORS = ["black", "#2166AC", "#D6604D"]
fits = [(p1, m1, r1, c1, l1, ok1),
        (p2, m2, r2_, c2, l2, ok2),
        (p3, m3, r3, c3, l3, ok3)]

for (p, m, r, c, l, ok), col in zip(fits, COLORS):
    if not ok: continue
    mf = np.linspace(p[1] + 0.01, M_LIM, 600)
    yf = np.where(m(mf, *p) > 0.15, m(mf, *p), np.nan)
    ax.plot(mf, yf, "-", color=col, lw=2.3, zorder=7,
            label=f"{l}$\\quad R^2={r:.3f}$")

# Data
ax.errorbar(mx_f, my_f, yerr=mye_f, fmt="ko", capsize=4,
            markersize=7, elinewidth=1.2, capthick=1.2, zorder=20)
ax.errorbar(mx_o, my_o, yerr=mye_o, fmt="o", capsize=4,
            markersize=7, mfc="white", mec="black", ecolor="black",
            elinewidth=1.0, capthick=1.0, linestyle="none", zorder=20)

# Completeness limit
ax.axvline(M_LIM, color="#555", ls=":", lw=1.3, zorder=1)
ax.text(M_LIM + 0.08, 0.93,
        f"$M_{{\\rm lim}} = {M_LIM}$\n($5\\sigma \\approx {S_MLIM_UJY:.0f}\\,\\mu$Jy)",
        fontsize=9, color="#444", ha="left", va="top",
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

# Format
ax.set_xlabel(r"$M_{\rm radio}$ / mag", fontsize=13)
ax.set_ylabel("$N$ (per 0.3 mag bin)", fontsize=13)
ax.set_yscale("log")
ax.set_xlim(M_all.min() - 0.5, M_all.max() + 0.5)
ax.set_ylim(0.4, my.max() * 6)
# ax.tick_params(labelsize=11, which="both", direction="in", top=True, right=True)
# changed 2026-08-30: the top and right spines are hidden below, so those ticks floated
ax.tick_params(labelsize=11, which="both", direction="in", top=False, right=False)
ax.xaxis.set_minor_locator(AutoMinorLocator(3))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9, framealpha=0.95, loc="upper left",
          handlelength=2.5, labelspacing=0.5, borderpad=0.6)

# Parameter box (compact)
pbox = [f"$N = {N_src}$ sources"]
if ok1: pbox.append(f"Black: $M^* = {p1[1]:.2f}$")
if ok2: pbox.append(f"Blue: $M^* = {p2[1]:.2f}$, $\\alpha = {p2[2]:.3f}$")
if ok3: pbox.append(f"Red: $M^* = {p3[1]:.2f}$, $\\alpha = {p3[2]:.3f}$, $\\beta = {p3[3]:.2f}$")
ax.text(0.97, 0.03, "\n".join(pbox), transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  alpha=0.93, ec="#ccc", lw=0.7))

plt.tight_layout()
out2 = os.path.join(FIGS, "step07c_pnlf_ciardullo_comparison.pdf")
plt.savefig(out2, format="pdf", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"    -> {out2}")


#  PRINT TABLE DATA — for pasting into LaTeX
print("\n[6] Model comparison table (for LaTeX):")
print(f"{'Rk':<3} {'Model':<24} {'k':>3} {'R²':>7} {'AIC':>8} {'BIC':>8} {'χ²ν':>7}")
print("-" * 65)
for i, r in enumerate(results):
    print(f"{i+1:<3} {r['name']:<24} {r['k']:>3} {r['r2']:>7.4f} "
          f"{r['aic']:>8.2f} {r['bic']:>8.2f} {r['redchi']:>7.3f}")

print("\nstep07c complete")
print("  paper figures written")
print(f"  -> step07c_pnlf_model_overlay.pdf")
print(f"  -> step07c_pnlf_ciardullo_comparison.pdf")
