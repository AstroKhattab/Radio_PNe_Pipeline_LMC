"""
step05_pnlf.py
==============
Step 5 - The radio planetary nebula luminosity function.

This is the point of the paper: the first PNLF measured in the radio for a
galaxy other than our own.  It is presented for two samples, and the same
question is asked of each.

    Sample A, the full radio sample.  Every radio detection of a parent PN.
              This is what a radio survey of the LMC actually delivers, and it
              is the sample a reader would reproduce.

    Sample B, the strong sample.  Only the detections that passed every
              criterion that could be evaluated for them (Step 4e).  Cleaner,
              smaller, and less likely to contain a background radio galaxy
              sitting on top of an optical PN.

Neither is obviously the right one, which is why both are fitted and both are
shown.  If the same model wins in both, the choice of sample does not matter
and the full sample can be quoted with the strong one as a robustness check.
If different models win, that disagreement is itself the result, and the
strong sample is the one to trust for the characteristic luminosity.

What is masked
--------------
Two things, and nothing else:

  * the completeness limit.  Each source carries its own 5 sigma limit from
    the local noise, and the likelihood is conditioned on it (see
    _support/pnlf_models.py).  Detections that fall below their own limit
    cannot be treated this way and are excluded.

  * sources above the physical flux ceiling from Step 4d.  A handful of
    objects are brighter than an LMC PN can be.  Whatever they are, they are
    not drawn from the luminosity function we are trying to measure, and a
    single over-luminous object at the bright end drags the cutoff with it.

Nothing else is trimmed.  Earlier versions of this analysis masked
considerably more, and the model ranking turned out to depend on what was
masked, which is not a property a result should have.

Luminosity
----------
    L = 4 pi d^2 S,  d = 49.59 kpc (Pietrzynski et al. 2019)
and the function is fitted in log10 L, in W/Hz.

Outputs
-------
    03_Outputs/step05_pnlf_catalogue.vot     luminosities and limits
    03_Outputs/step05_pnlf_models.csv        every model on every sample
    03_Outputs/step05_pnlf_summary.csv       the adopted result
    04_Figures/step05_pnlf_sampleA.pdf       full radio sample
    04_Figures/step05_pnlf_sampleB.pdf       strong sample
    04_Figures/step05_pnlf_comparison.pdf    the two side by side

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import pnlf_models as M

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

IN_FILE = os.path.join(OUTS, "step04f_optical_comparison.vot")
OUT_TABLE = os.path.join(OUTS, "step05_pnlf_catalogue.vot")
OUT_MODELS = os.path.join(OUTS, "step05_pnlf_models.csv")
OUT_SUMMARY = os.path.join(OUTS, "step05_pnlf_summary.csv")
FIG_A = os.path.join(FIGS, "step05_pnlf_sampleA.pdf")
FIG_B = os.path.join(FIGS, "step05_pnlf_sampleB.pdf")
FIG_C = os.path.join(FIGS, "step05_pnlf_comparison.pdf")

DISTANCE_KPC = 49.59
DISTANCE_M = DISTANCE_KPC * 3.085677581491367e19
JY_TO_W_HZ = 4.0 * np.pi * DISTANCE_M ** 2 * 1.0e-26
DETECTION_SIGMA = 5.0
COLOUR_A = "#22539b"
COLOUR_B = "#c44e52"

print("=" * 64)
print("  Step 5  -  radio planetary nebula luminosity function")
print("=" * 64)

cat = Table.read(IN_FILE, format="votable")
n = len(cat)

flux_mjy = np.asarray(cat["flux_1295_mJy"], dtype=float)
lum = flux_mjy * 1e-3 * JY_TO_W_HZ
log_lum = np.where(lum > 0, np.log10(np.maximum(lum, 1e-300)), np.nan)

# Each source's own detection limit.  Catalogue detections carry the local RMS
# from the source finder; forced-photometry sources carry the local scatter
# measured in Step 3a.  Where neither exists we fall back on the sample median,
# which affects only a few objects.
rms = np.asarray(cat["mkt_rms_mJy_beam"], dtype=float)
fp_rms = np.asarray(cat["fp_rms_mJy_beam"], dtype=float) \
    if "fp_rms_mJy_beam" in cat.colnames else np.full(n, np.nan)
rms = np.where(np.isfinite(rms), rms, fp_rms)
median_rms = float(np.nanmedian(rms))
rms = np.where(np.isfinite(rms) & (rms > 0), rms, median_rms)
limit_mjy = DETECTION_SIGMA * rms
log_limit = np.log10(limit_mjy * 1e-3 * JY_TO_W_HZ)

print(f"\ndetections in the catalogue : {n}")
print(f"median local RMS            : {median_rms * 1e3:.1f} uJy/beam")
print(f"median {DETECTION_SIGMA:.0f} sigma limit         : "
      f"{np.median(limit_mjy) * 1e3:.0f} uJy   "
      f"(log L = {np.median(log_limit):.2f})")

cat["radio_luminosity_W_Hz"] = Column(lum)
cat["log_radio_luminosity"] = Column(log_lum)
cat["log_luminosity_limit"] = Column(log_limit)
cat["pnlf_local_rms_mJy"] = Column(rms.astype("f4"), unit="mJy/beam")

# ------------------------------------------------------------------- masking
above_ceiling = np.asarray(cat["ceiling_class"]) == "above_ceiling"
usable = np.isfinite(log_lum) & np.isfinite(log_limit) & (log_lum >= log_limit)

mask_reason = np.full(n, "kept", dtype="U22")
mask_reason[~usable] = "below own 5sigma limit"
mask_reason[above_ceiling] = "above flux ceiling"
cat["pnlf_mask"] = Column(mask_reason)

in_fit = usable & ~above_ceiling
sample_A = in_fit
sample_B = in_fit & (np.asarray(cat["strong_sample"]) == 1)

print(f"\nmasked, below own 5 sigma limit : {int((~usable).sum())}")
print(f"masked, above the flux ceiling  : {int(above_ceiling.sum())}")
print(f"Sample A, full radio sample     : {int(sample_A.sum())}")
print(f"Sample B, strong sample         : {int(sample_B.sum())}")

cat["in_sample_A"] = Column(sample_A.astype(np.int16))
cat["in_sample_B"] = Column(sample_B.astype(np.int16))
cat.write(OUT_TABLE, format="votable", overwrite=True)

# ==================================================================== fitting
SAMPLES = [("A", "full radio sample", sample_A, COLOUR_A),
           ("B", "strong sample", sample_B, COLOUR_B)]

rows = []
results = {}
for key, label, mask, colour in SAMPLES:
    x = log_lum[mask]
    u = log_limit[mask]
    print(f"\n{'=' * 64}")
    print(f"  Sample {key}  -  {label}   ({len(x)} PNe)")
    print(f"{'=' * 64}")
    print(f"  {'model':<24}{'k':>3}{'logL':>10}{'AICc':>10}"
          f"{'dAICc':>9}{'BIC':>10}{'KS D':>8}{'KS p':>8}")

    fits = []
    for model in M.MODEL_ORDER:
        fits.append(M.fit_model(model, x, u))
    best = min(f.aicc for f in fits)
    best_bic = min(f.bic for f in fits)
    order = np.argsort([f.aicc for f in fits])

    for rank, i in enumerate(order, start=1):
        f = fits[i]
        print(f"  {f.model:<24}{f.k:>3}{f.loglike:>10.2f}{f.aicc:>10.2f}"
              f"{f.aicc - best:>9.2f}{f.bic:>10.2f}{f.ks_d:>8.3f}"
              f"{f.ks_p_naive:>8.3f}")
        rows.append(dict(sample=key, sample_name=label, rank=rank,
                         model=f.model, n_pne=len(x), n_parameters=f.k,
                         log_likelihood=round(f.loglike, 3),
                         AICc=round(f.aicc, 3),
                         delta_AICc=round(f.aicc - best, 3),
                         BIC=round(f.bic, 3),
                         delta_BIC=round(f.bic - best_bic, 3),
                         KS_D=round(f.ks_d, 4),
                         KS_p=round(f.ks_p_naive, 4),
                         parameters=M.pstring(f.params, f.names)))

    winner = fits[order[0]]
    canonical = next(f for f in fits if f.model == "Canonical Ciardullo")
    print(f"\n  best by AICc      : {winner.model}")
    print(f"  canonical PNLF    : rank "
          f"{int(np.where(order == fits.index(canonical))[0][0]) + 1}"
          f" of {len(fits)},  dAICc = {canonical.aicc - best:.2f}")
    print(f"  log L* (canonical): {canonical.params[0]:.3f}")
    results[key] = dict(label=label, mask=mask, x=x, u=u, fits=fits,
                        order=order, winner=winner, canonical=canonical,
                        colour=colour, best=best)

Table(rows=rows).write(OUT_MODELS, format="ascii.csv", overwrite=True)

# ------------------------------------------------------- does the choice matter
wa = results["A"]["winner"].model
wb = results["B"]["winner"].model
print(f"\n{'=' * 64}")
if wa == wb:
    print(f"  Both samples prefer the same model: {wa}")
    print("  The choice of sample does not change the conclusion; quote the")
    print("  full sample and cite the strong sample as a robustness check.")
else:
    print(f"  The two samples prefer different models:")
    print(f"      Sample A (full)   -> {wa}")
    print(f"      Sample B (strong) -> {wb}")
    print("  The disagreement is the result. Adopt the strong sample for the")
    print("  characteristic luminosity; it is the less contaminated of the two.")
print(f"{'=' * 64}")

summary_rows = []
for key in ("A", "B"):
    r = results[key]
    can = r["canonical"]
    summary_rows.append(dict(
        sample=key, sample_name=r["label"], n_pne=len(r["x"]),
        best_model=r["winner"].model,
        best_model_delta_AICc=0.0,
        canonical_rank=int(np.where(r["order"] ==
                                    r["fits"].index(can))[0][0]) + 1,
        canonical_delta_AICc=round(can.aicc - r["best"], 3),
        canonical_logLstar=round(float(can.params[0]), 4),
        canonical_KS_D=round(can.ks_d, 4),
        canonical_KS_p=round(can.ks_p_naive, 4),
        median_log_limit=round(float(np.median(r["u"])), 4)))
Table(rows=summary_rows).write(OUT_SUMMARY, format="ascii.csv", overwrite=True)

# ==================================================================== figures
def draw_pnlf(ax, r, show_models=3):
    """Binned luminosity function with the best-fitting curves over it."""
    x, u = r["x"], r["u"]
    lo, hi = np.floor(np.min(x) * 2) / 2, np.ceil(np.max(x) * 2) / 2
    bins = np.linspace(lo, hi, max(9, int((hi - lo) / 0.25) + 1))
    centres = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(x, bins=bins)
    width = np.diff(bins)

    ax.bar(bins[:-1], counts, width=width, align="edge", color=r["colour"],
           alpha=0.30, edgecolor=r["colour"], lw=0.8,
           label=f"observed ({len(x)} PNe)")
    ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="none",
                ecolor=r["colour"], lw=1.0, capsize=2.5)

    grid = np.linspace(lo, hi, 400)
    styles = ["-", "--", ":"]
    for j, i in enumerate(r["order"][:show_models]):
        f = r["fits"][i]
        dens = M.average_conditional_density(f.model, f.params, grid, u)
        dens = dens / np.trapz(dens, grid) * len(x) * np.mean(width)
        ax.plot(grid, dens, styles[j % 3], lw=1.7 if j == 0 else 1.2,
                color="k" if j == 0 else "0.45",
                label=f"{f.model}  ($\\Delta$AICc = {f.aicc - r['best']:.1f})")

    # only draw the canonical PNLF separately when it is not already one of
    # the curves above, otherwise it appears twice in the legend
    can = r["canonical"]
    shown = {r["fits"][i].model for i in r["order"][:show_models]}
    if can.model not in shown:
        dens = M.average_conditional_density(can.model, can.params, grid, u)
        dens = dens / np.trapz(dens, grid) * len(x) * np.mean(width)
        ax.plot(grid, dens, "-", lw=1.5, color="#2c7d3f",
                label=f"Canonical Ciardullo  "
                      f"($\\Delta$AICc = {can.aicc - r['best']:.1f})")

    ax.axvline(np.median(u), color="0.3", ls="-.", lw=1.1)
    ax.text(np.median(u), ax.get_ylim()[1] * 0.97,
            f" median {DETECTION_SIGMA:.0f}$\\sigma$ limit", fontsize=7.5,
            va="top", rotation=90)
    ax.set_xlabel(r"$\log_{10}(L_{1.295\,\mathrm{GHz}}\ /\ \mathrm{W\,Hz^{-1}})$")
    ax.set_ylabel("number of PNe per bin")
    ax.legend(fontsize=7.5, framealpha=0.92)
    ax.tick_params(labelsize=8)


def draw_cumulative(ax, r):
    """Observed cumulative distribution against the fitted models."""
    x, u = r["x"], r["u"]
    xs = np.sort(x)
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
            color=r["colour"], lw=1.6, label="observed")
    grid = np.linspace(np.min(x), np.max(x), 300)
    for j, i in enumerate(r["order"][:2]):
        f = r["fits"][i]
        cdf = M.average_conditional_cdf(f.model, f.params, grid, u)
        ax.plot(grid, cdf, "--" if j else "-", lw=1.3,
                color="k" if j == 0 else "0.5",
                label=f"{f.model}  (D = {f.ks_d:.3f})")
    can = r["canonical"]
    shown = {r["fits"][i].model for i in r["order"][:2]}
    if can.model not in shown:
        cdf = M.average_conditional_cdf(can.model, can.params, grid, u)
        ax.plot(grid, cdf, "-", lw=1.3, color="#2c7d3f",
                label=f"Canonical Ciardullo  (D = {can.ks_d:.3f})")
    ax.set_xlabel(r"$\log_{10}(L_{1.295\,\mathrm{GHz}}\ /\ \mathrm{W\,Hz^{-1}})$")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, framealpha=0.92, loc="lower right")
    ax.tick_params(labelsize=8)


for key, figpath in (("A", FIG_A), ("B", FIG_B)):
    r = results[key]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    draw_pnlf(axes[0], r)
    axes[0].set_title(f"Sample {key}: {r['label']}  ({len(r['x'])} PNe)",
                      fontsize=10)
    draw_cumulative(axes[1], r)
    axes[1].set_title("Cumulative distribution", fontsize=10)
    fig.tight_layout()
    fig.savefig(figpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {os.path.basename(figpath)}")

# side-by-side comparison
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))
draw_pnlf(axes[0], results["A"])
axes[0].set_title(f"Sample A: full radio sample ({len(results['A']['x'])} PNe)",
                  fontsize=10)
draw_pnlf(axes[1], results["B"])
axes[1].set_title(f"Sample B: strong sample ({len(results['B']['x'])} PNe)",
                  fontsize=10)

ax = axes[2]
models = M.MODEL_ORDER
xs = np.arange(len(models))
w = 0.38
for off, key, colour in ((-w / 2, "A", COLOUR_A), (w / 2, "B", COLOUR_B)):
    r = results[key]
    d = [r["fits"][models.index(m)].aicc - r["best"] for m in models]
    ax.bar(xs + off, np.clip(d, 0, 60), width=w, color=colour,
           label=f"Sample {key}")
ax.axhline(2, color="0.4", ls="--", lw=1.0)
ax.text(len(models) - 0.4, 2.4, r"$\Delta$AICc = 2", fontsize=7.5, ha="right")
ax.set_xticks(xs)
ax.set_xticklabels(models, rotation=32, ha="right", fontsize=7.5)
ax.set_ylabel(r"$\Delta$AICc  (clipped at 60)")
ax.set_title("Which model wins, in each sample", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(FIG_C, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"wrote {os.path.basename(FIG_C)}")
print(f"wrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_MODELS)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
