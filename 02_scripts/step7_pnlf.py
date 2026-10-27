"""Step 7: the radio planetary nebula luminosity function.

Every MeerKAT detection is converted to M_radio = -2.5 log10(L / L_ref) with
L = 4 pi d^2 S and binned.  Bins brighter than the flux gate and fainter than
the 5 sigma completeness limit are drawn but not fitted.  Ten functional forms
are fitted to the binned counts by minimising the Cash (1979) statistic, the
Poisson likelihood, so that bins with few or zero counts are treated
correctly.  Models are ranked by AIC = C + 2k.

Fitting: differential evolution (global search, fixed seed), then a bounded
local minimisation.  Parameter errors come from the inverse Hessian of C/2.

Outputs: catalogue columns M_radio, pnlf_fitted; figures f5_pnlf_data.pdf,
         f6_pnlf_fits.pdf, f7_pnlf_overlay.pdf, f8_pnlf_canonical.pdf;
         table t6_pnlf_models.tex; numbers block [step7].
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2, skewnorm, t as student_t

import common as c

cfg = c.CFG
pl = cfg["pnlf"]
plt = c.plot_style()
BIN = pl["bin_mag"]
M_LIM = pl["faint_limit_mag"]
M_CEIL = float(c.magnitude(cfg["flux_gate"]["ceiling_mjy"] * 1e-3))
S_LIM_UJY = float(c.flux_at_magnitude(M_LIM)) * 1e6
CA, CB = pl["ciardullo_alpha"], pl["ciardullo_beta"]

cat = c.read_catalogue()
flux_jy = np.asarray(cat["mkt_flux_mJy"], float) * 1e-3
has = np.isfinite(flux_jy) & (flux_jy > 0)
M = np.where(has, c.magnitude(np.where(has, flux_jy, 1.0)), np.nan)
lum = 4 * np.pi * c.distance_cm() ** 2 * flux_jy * 1e-23
c.set_column(cat, "M_radio", M)

# --- binning and masks -----------------------------------------------------------
Mv = M[has]
edges = np.arange(np.floor(Mv.min() / BIN) * BIN, np.ceil(Mv.max() / BIN) * BIN + BIN / 2, BIN)
counts, _ = np.histogram(Mv, bins=edges)
centres = 0.5 * (edges[:-1] + edges[1:])
in_fit = (centres >= M_CEIL) & (centres <= M_LIM)
x, n_obs = centres[in_fit], counts[in_fit].astype(float)
fitted_src = has & (M >= edges[:-1][in_fit].min()) & (M < edges[1:][in_fit].max())
c.set_column(cat, "pnlf_fitted", fitted_src)
c.write_catalogue(cat)


def poisson_interval(n):
    """68 per cent Garwood interval, used only for drawing error bars."""
    n = np.asarray(n, float)
    lo = np.where(n > 0, chi2.ppf(0.16, 2 * n) / 2, 0.0)
    hi = chi2.ppf(0.84, 2 * (n + 1)) / 2
    return n - lo, hi - n


# --- models ------------------------------------------------------------------------
def gaussian(m, A, mu, s): return A * np.exp(-0.5 * ((m - mu) / s) ** 2)
def student(m, A, mu, s, nu): return A * student_t.pdf(m, nu, loc=mu, scale=s)
def cauchy(m, A, mu, g): return A / (1 + ((m - mu) / g) ** 2)
def skew(m, A, loc, s, a): return A * skewnorm.pdf(m, a, loc=loc, scale=s)
def poly2(m, a, b, c0): return a * m ** 2 + b * m + c0
def poly3(m, a, b, c0, d): return a * m ** 3 + b * m ** 2 + c0 * m + d
def expo(m, a, b, c0): return a * np.exp(b * m) + c0


def schechter(m, phi, ms, al):
    # Schechter (1976) in magnitudes: exponential cutoff on the bright side of M*.
    d = ms - m
    return phi * 10 ** (0.4 * (al + 1) * d) * np.exp(-10 ** (0.4 * d))


def ciardullo(m, N0, Ms, al, be):
    v = N0 * np.exp(al * (m - Ms)) * (1 - np.exp(be * (Ms - m)))
    return np.where(m > Ms, v, 0.0)


def ciar_canonical(m, N0, Ms): return ciardullo(m, N0, Ms, CA, CB)
def ciar_alpha(m, N0, Ms, al): return ciardullo(m, N0, Ms, al, CB)


def model_list(x, n):
    top, lo, hi = n.max(), x.min(), x.max()
    peak = x[np.argmax(n)]
    ms_hi, ms_lo = lo - 0.05, lo - 5.0        # M* brighter than the brightest fitted bin
    return [
        ("Gaussian", gaussian, [0, lo - 3, 0.1], [top * 4, hi + 2, 8], 3),
        ("Student-$t$", student, [0, lo - 3, 0.1, 0.5], [top * 3, hi + 2, 10, 200], 4),
        ("Cauchy", cauchy, [0, lo - 3, 0.05], [top * 4, hi + 2, 10], 3),
        ("Skew-normal", skew, [0, lo - 4, 0.1, -30], [top * 6, hi + 3, 15, 30], 4),
        ("Polynomial (2)", poly2, [-500, -200, 0], [500, 200, top * 3], 3),
        ("Polynomial (3)", poly3, [-500, -500, -200, 0], [500, 500, 200, top * 3], 4),
        ("Exponential", expo, [0, 0, -50], [top * 10, 5, top], 3),
        ("Schechter", schechter, [0, lo - 4, -5], [top * 10, lo, 5], 3),
        ("Ciardullo (free)", ciardullo, [0, ms_lo, 0.05, 1.0], [1e6, ms_hi, 8, 30], 4),
        ("Ciardullo (canonical)", ciar_canonical, [0, ms_lo], [1e6, ms_hi], 2),
    ]


def cash(n, mu):
    """Cash statistic in its deviance form: 2 sum[mu - n + n ln(n/mu)]."""
    mu = np.maximum(mu, 1e-10)
    term = np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / mu), 0.0)
    return 2.0 * np.sum(mu - n + term)


def hessian(f, p, lb, ub):
    k = len(p)
    h = np.maximum(np.abs(p) * 1e-4, 1e-5)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            def at(di, dj):
                q = p.copy()
                q[i] += di * h[i]
                q[j] += dj * h[j]
                return f(np.clip(q, lb, ub))
            H[i, j] = (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (4 * h[i] * h[j])
    return H


def fit_model(func, lb, ub, x, n):
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    f = lambda p: cash(n, func(x, *p))
    de = differential_evolution(f, list(zip(lb, ub)), seed=pl["seed"], maxiter=12000,
                                tol=pl["de_tolerance"], popsize=pl["de_popsize"],
                                mutation=(0.3, 1.9), polish=False)
    loc = minimize(f, de.x, method="L-BFGS-B", bounds=list(zip(lb, ub)))
    p = loc.x if loc.fun <= de.fun else de.x
    C = f(p)
    try:
        cov = np.linalg.inv(0.5 * hessian(f, p, lb, ub))
        perr = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except np.linalg.LinAlgError:
        perr = np.full(len(p), np.nan)
    tol = lambda b: 1e-3 * np.maximum(np.abs(b), 1e-3)
    at_bound = (np.abs(p - lb) <= tol(lb)) | (np.abs(p - ub) <= tol(ub))
    return dict(func=func, p=p, perr=perr, C=C, k=len(p), aic=C + 2 * len(p),
                bound=at_bound, dof=len(n) - len(p))


results = []
for name, func, lb, ub, k in model_list(x, n_obs):
    r = fit_model(func, lb, ub, x, n_obs)
    r["name"] = name
    results.append(r)
results.sort(key=lambda r: r["aic"])
best_aic = results[0]["aic"]
param_names = {
    "Gaussian": ["A", "mu", "sigma"], "Student-$t$": ["A", "mu", "sigma", "nu"],
    "Cauchy": ["A", "mu", "gamma"], "Skew-normal": ["A", "loc", "scale", "a"],
    "Polynomial (2)": ["a", "b", "c"], "Polynomial (3)": ["a", "b", "c", "d"],
    "Exponential": ["a", "b", "c"], "Schechter": ["phi", "M*", "alpha"],
    "Ciardullo (free)": ["N0", "M*", "alpha", "beta"], "Ciardullo (canonical)": ["N0", "M*"],
}

# Functional forms, written once for the table and the fit figure.
FORMS = {
    "Gaussian": r"$A\,e^{-(M-\mu)^2/2\sigma^2}$",
    "Student-$t$": r"$\frac{A}{\sigma}\,t_\nu\!\left(\frac{M-\mu}{\sigma}\right)$",
    "Cauchy": r"$A\,/\,[1+((M-\mu)/\gamma)^2]$",
    "Skew-normal": r"$\frac{2A}{\omega}\,\phi(z)\,\Phi(az)$, $z=(M-\xi)/\omega$",
    "Polynomial (2)": r"$aM^2+bM+c$",
    "Polynomial (3)": r"$aM^3+bM^2+cM+d$",
    "Exponential": r"$a\,e^{bM}+c$",
    "Schechter": r"$\phi\,10^{0.4(\alpha+1)(M^*-M)}\,\exp[-10^{0.4(M^*-M)}]$",
    "Ciardullo (free)": r"$N_0\,e^{\alpha(M-M^*)}\,[1-e^{\beta(M^*-M)}]$",
    "Ciardullo (canonical)": rf"$N_0\,e^{{{CA}(M-M^*)}}\,[1-e^{{{CB:g}(M^*-M)}}]$",
}
TEX_NAMES = {"mu": "\\mu", "sigma": "\\sigma", "gamma": "\\gamma", "alpha": "\\alpha",
             "beta": "\\beta", "nu": "\\nu", "phi": "\\phi", "N0": "N_0", "M*": "M^*",
             "loc": "\\xi", "scale": "\\omega"}


def tex_pars(r):
    return ", ".join(f"${TEX_NAMES.get(nm, nm)}={v:.3g}$" for nm, v in zip(param_names[r["name"]], r["p"]))


# --- numbers -----------------------------------------------------------------------
gate = np.asarray(cat["gate"], str)
lines = [
    ("Distance (kpc)", cfg["distance_kpc"]),
    ("MeerKAT sources with M_radio", int(has.sum())),
    ("  of which above the flux gate", int((has & (gate == "rejected")).sum())),
    ("M_radio range", f"{Mv.min():.2f} to {Mv.max():.2f}"),
    ("Bright mask M_ceil (2.2 mJy)", f"{M_CEIL:.2f}"),
    ("Faint mask M_lim / flux (uJy) / sigma", f"{M_LIM} / {S_LIM_UJY:.1f} / {S_LIM_UJY / cfg['meerkat']['median_rms_ujy']:.1f}"),
    ("Bins total / fitted", f"{len(counts)} / {in_fit.sum()}"),
    ("Fitted bins with zero counts", int((n_obs == 0).sum())),
    ("Sources in fitted bins", int(n_obs.sum())),
    ("Sources brighter than fit range", int((Mv < edges[:-1][in_fit].min()).sum())),
    ("Sources fainter than fit range", int((Mv >= edges[1:][in_fit].max()).sum())),
    ("Fit statistic", "Cash (Poisson deviance); AIC = C + 2k"),
]
for i, r in enumerate(results):
    names = param_names[r["name"]]
    pars = ", ".join(f"{nm}={v:.4g}+-{e:.2g}" + ("[bound]" if b else "")
                     for nm, v, e, b in zip(names, r["p"], r["perr"], r["bound"]))
    lines.append((f"{i + 1}. {r['name']}",
                  f"C={r['C']:.2f} dof={r['dof']} AIC={r['aic']:.2f} dAIC={r['aic'] - best_aic:.2f} | {pars}"))

# --- t6_pnlf_models.tex: all models ---------------------------------------------------------------
rows = []
for i, r in enumerate(results):
    rows.append([i + 1, r["name"], FORMS[r["name"]], r["k"], f"{r['C']:.2f}", f"{r['aic']:.2f}",
                 f"{r['aic'] - best_aic:.2f}", tex_pars(r)])
c.write_tex("t6_pnlf_models.tex", ["Rank", "Model", "Form", "$k$", "$C$", "AIC", "$\\Delta$AIC", "Parameters"],
            rows, align="rllrrrrl", note="$M$ is $M_{\\rm radio}$; $t_\\nu$ is the Student-$t$ density, "
            "$\\phi$ and $\\Phi$ the normal density and distribution.")


# --- figures -------------------------------------------------------------------------
lo_err, hi_err = poisson_interval(counts)
grid = np.linspace(x.min() - BIN / 2, M_LIM, 600)
xlim = (Mv.min() - 0.4, Mv.max() + 0.4)
ytop = counts.max() * 6


def draw_masks(ax, labels=False):
    """Shade and mark the two masks: flux gate (bright) and 5 sigma limit (faint)."""
    ax.axvspan(xlim[0], M_CEIL, color="#C23B33", alpha=0.08, zorder=0)
    ax.axvspan(M_LIM, xlim[1], color="0.5", alpha=0.10, zorder=0)
    ax.axvline(M_CEIL, color="#C23B33", ls="--", lw=1.2, zorder=1)
    ax.axvline(M_LIM, color="0.4", ls="--", lw=1.2, zorder=1)
    if labels:
        ax.text(M_CEIL + 0.05, 0.95, f"flux gate\n2.2 mJy\n$M$={M_CEIL:.2f}", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color="#8E2820")
        ax.text(M_LIM - 0.05, 0.95, f"$5\\sigma$ limit\n{S_LIM_UJY:.0f} $\\mu$Jy\n$M$={M_LIM}",
                transform=ax.get_xaxis_transform(), va="top", ha="right", fontsize=8, color="0.3")


def draw_data(ax, ms=6):
    ax.errorbar(centres[in_fit], counts[in_fit], yerr=[lo_err[in_fit], hi_err[in_fit]], fmt="o",
                color="k", ms=ms, capsize=3, zorder=10, label="Fitted bins")
    out = ~in_fit & (counts > 0)
    ax.errorbar(centres[out], counts[out], yerr=[lo_err[out], hi_err[out]], fmt="o", mfc="white",
                mec="k", ecolor="0.4", ms=ms, capsize=3, zorder=9, label="Masked bins")
    draw_masks(ax)
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.4, ytop)


def curve(ax, r, **kw):
    y = r["func"](grid, *r["p"])
    ax.plot(grid, np.where(y > 0.05, y, np.nan), **kw)


# f5_pnlf_data.pdf: data preparation.
fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
ax = axes[0, 0]
ax.hist(np.log10(lum[has]), bins=30, color="#1976D2", alpha=0.75, edgecolor="white")
ax.set(xlabel=r"$\log_{10} L_{1.295\,{\rm GHz}}$ (erg s$^{-1}$ Hz$^{-1}$)", ylabel="Number of PNe")
ax.text(0.97, 0.95, f"$N$ = {has.sum()}\n$d$ = {cfg['distance_kpc']} kpc", transform=ax.transAxes,
        ha="right", va="top", fontsize=9)
ax.set_title("(a) Luminosity", loc="left", fontsize=11)
ax = axes[0, 1]
ax.hist(Mv, bins=edges, color="#4CAF50", alpha=0.75, edgecolor="white")
draw_masks(ax, labels=True)
ax.set(xlabel=r"$M_{\rm radio}$ (mag)", ylabel="Number of PNe", xlim=xlim)
ax.set_title(f"(b) Magnitude, {BIN} mag bins", loc="left", fontsize=11)
ax = axes[1, 0]
draw_masks(ax)
ax.bar(centres, counts, width=BIN * 0.85, color="#FF9800", alpha=0.55, edgecolor="#E65100")
ax.errorbar(centres, counts, yerr=[lo_err, hi_err], fmt="o", color="k", ms=4, capsize=3,
            label="68% Poisson interval")
ax.set(xlabel=r"$M_{\rm radio}$ (mag)", ylabel=r"$N$ per bin", xlim=xlim)
ax.legend(fontsize=8, loc="upper center", frameon=False)
ax.set_title("(c) Binned counts", loc="left", fontsize=11)
ax = axes[1, 1]
draw_data(ax)
ax.set(xlabel=r"$M_{\rm radio}$ (mag)", ylabel=r"$N$ per bin")
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.set_title(f"(d) Fit input: {in_fit.sum()} bins, {int(n_obs.sum())} PNe", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(c.figure("f5_pnlf_data.pdf"))
plt.close(fig)

# f6_pnlf_fits.pdf: the best models, ranked, sized to fill one page.
k = pl["grid_models"]
rows = int(np.ceil(k / 2))
fig, axes = plt.subplots(rows, 2, figsize=(7.2, 1.75 * rows), sharex=True, sharey=True)
for i, ax in enumerate(np.atleast_1d(axes).flat):
    if i >= k:
        ax.axis("off")
        continue
    r = results[i]
    draw_data(ax, ms=4)
    curve(ax, r, color="#C0392B" if i == 0 else "#1F6FB4", lw=1.8)
    ax.set_title(f"{i + 1}. {r['name']}", fontsize=8.5, pad=3)
    ax.text(0.03, 0.94, f"$\\Delta$AIC = {r['aic'] - best_aic:.2f}\n$C$ = {r['C']:.1f}, $k$ = {r['k']}",
            transform=ax.transAxes, va="top", fontsize=6.5)
    ax.text(0.55, 0.04, f"{FORMS[r['name']]}\n{tex_pars(r)}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.5, linespacing=1.5)
    ax.tick_params(labelsize=7)
    if i % 2 == 0:
        ax.set_ylabel(r"$N$ per bin", fontsize=8)
    if i >= k - 2:
        ax.set_xlabel(r"$M_{\rm radio}$ (mag)", fontsize=8)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
fig.savefig(c.figure("f6_pnlf_fits.pdf"))
plt.close(fig)

# f7_pnlf_overlay.pdf: overlay of the best empirical models and the canonical Ciardullo form.
fig, ax = plt.subplots(figsize=(8.5, 6))
draw_data(ax, ms=7)
empirical = [r for r in results if not r["name"].startswith("Ciardullo")][:pl["overlay_models"]]
for r, col, ls in zip(empirical, ["#D32F2F", "#1976D2", "#388E3C", "#F57C00"], ["-", "--", "-.", ":"]):
    curve(ax, r, color=col, ls=ls, lw=2, label=f"{r['name']} ($\\Delta$AIC {r['aic'] - best_aic:.1f})")
canon = next(r for r in results if r["name"] == "Ciardullo (canonical)")
curve(ax, canon, color="k", ls="--", lw=2.2, label=f"Ciardullo canonical ($\\Delta$AIC {canon['aic'] - best_aic:.1f})")
ax.set(xlabel=r"$M_{\rm radio}$ (mag)", ylabel=r"$N$ per 0.3 mag bin")
ax.legend(fontsize=8.5, loc="upper left")
ax.text(0.97, 0.03, f"$N$ = {has.sum()}", transform=ax.transAxes, ha="right", fontsize=9)
fig.savefig(c.figure("f7_pnlf_overlay.pdf"))
plt.close(fig)

# f8_pnlf_canonical.pdf: the canonical Ciardullo form alone, the main-text figure.
fig, ax = plt.subplots(figsize=(8.5, 6))
draw_data(ax, ms=7)
rank = results.index(canon) + 1
curve(ax, canon, color="k", lw=2.2,
      label=f"Ciardullo canonical, rank {rank} ($\\Delta$AIC {canon['aic'] - best_aic:.2f})")
ax.axvline(canon["p"][1], color="k", ls=":", lw=1.2)
ax.text(canon["p"][1] + 0.05, 0.05, f"$M^*$ = {canon['p'][1]:.2f}", transform=ax.get_xaxis_transform(), fontsize=9)
ax.set(xlabel=r"$M_{\rm radio}$ (mag)", ylabel=r"$N$ per 0.3 mag bin")
ax.legend(fontsize=8.5, loc="upper left")
ax.text(0.97, 0.03, f"$N$ = {has.sum()}", transform=ax.transAxes, ha="right", fontsize=9)
fig.savefig(c.figure("f8_pnlf_canonical.pdf"))
plt.close(fig)

# The Ciardullo form with increasing freedom: fits for the numbers file only.
lo_x = x.min()
variants = [
    (r"$\alpha$, $\beta$ fixed", ciar_canonical, [0, lo_x - 5], [1e6, lo_x - 0.05]),
    (r"$\alpha$ free, $\beta$ fixed", ciar_alpha, [0, lo_x - 5, 0.05], [1e6, lo_x - 0.05, 8]),
    (r"$\alpha$, $\beta$ free", ciardullo, [0, lo_x - 5, 0.05, 1.0], [1e6, lo_x - 0.05, 8, 30]),
]
for label, func, lb, ub in variants:
    r = fit_model(func, lb, ub, x, n_obs)
    extra = "".join(f", {nm}={v:.2f}" for nm, v in zip(["alpha", "beta"], r["p"][2:]))
    lines.append((f"Ciardullo variant: {label}",
                  f"C={r['C']:.2f} AIC={r['aic']:.2f} M*={r['p'][1]:.3f}{extra}"
                  + (" [M* at bound]" if r["bound"][1] else "")))

c.write_numbers("step7", lines)
print(f"step7: best model {results[0]['name']}, {in_fit.sum()} bins fitted")
