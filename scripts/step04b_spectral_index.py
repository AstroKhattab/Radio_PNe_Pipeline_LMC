"""
step04b_spectral_index.py
=========================
Step 4b - First criterion: a thermal radio spectrum.

Planetary nebulae radiate free-free emission from ionised gas.  Between about
0.9 and 1.7 GHz an optically thin nebula has a nearly flat spectrum,
alpha = -0.1, and an optically thick one rises towards alpha = +2.  Background
radio galaxies, which are the main contaminant at these flux levels, are
synchrotron sources with alpha near -0.7.  The spectral index therefore
separates a genuine PN from a chance alignment with an extragalactic source
better than anything else we can measure.

The fit
-------
log10 S = alpha log10 nu + c, weighted least squares, with the weights taken
from the flux uncertainties propagated into log space.  The points are:

    twelve MeerKAT sub-bands, 908 to 1656 MHz
    the ASKAP-EMU 888 MHz integrated flux, where the PN is detected there

MeerKAT channels 8 and 9 are not used; they are contaminated by
radio-frequency interference and were excluded from the published fits for
the same reason.  Adding the ASKAP point roughly doubles the frequency lever
arm, which is what makes an index measurable at all for the fainter sources.

A sub-band only enters the fit if the source is actually detected in it,
peak / local RMS >= MIN_BAND_SNR.  This matters more than it sounds.  A source
picked up at three sigma in the full-bandwidth image has, by construction,
about one sigma in each of twelve sub-bands, so its sub-band fluxes are
essentially noise.  Fitting them anyway returns a slope drawn from the noise,
and because the fitted uncertainty on that slope can still come out small, the
result passes a formal error cut and lands in the catalogue as a confident
measurement.  Left uncorrected this produced a large spurious population of
steep-spectrum sources confined entirely to the faintest flux bin.  Requiring
each band to be individually detected removes it.

Reliability
-----------
A fitted index is only used if it rests on at least MIN_RELIABLE_POINTS
independent flux measurements, has a formal error below MAX_ALPHA_ERROR, and
a reduced chi-square below MAX_REDUCED_CHISQ.  Fits that fail any of these
are recorded with the reason; they are not silently dropped and they are not
counted as either thermal or steep.

How much the index can actually tell us
---------------------------------------
The criterion is only as good as the measurement, and the measurement gets
worse as the source gets fainter.  To show where the line falls, the same
fitting cuts are applied to the whole MeerKAT 5 sigma catalogue, which at
these flux levels is almost entirely background radio galaxies.  The median
index of that field population runs from about -0.74 at several mJy to about
-0.99 below 0.3 mJy.  Our PNe are compared against it bin by bin: where the
two distributions sit on top of each other the index carries no information,
and where the PNe are flatter than the field it does.  That comparison is
plotted alongside the histogram and is the honest answer to the question of
how far this criterion can be pushed.

Classification
--------------
    thermal    -0.2 <= alpha <= +2.0   consistent with free-free emission
    uncertain  -0.5 <= alpha <  -0.2   between the two populations
    steep             alpha <  -0.5    synchrotron; argues against a PN

Outputs
-------
    03_Outputs/step04b_spectral_index.vot   detections with alpha
    03_Outputs/step04b_spectral_summary.csv counts behind the paper table
    04_Figures/step04b_spectral_index.pdf   alpha distribution and SED quality

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

DET_FILE = os.path.join(OUTS, "step04a_radio_detections.vot")
MKT5_FILE = os.path.join(DATA, "MeerKAT_Final_5sigma.vot")
OUT_TABLE = os.path.join(OUTS, "step04b_spectral_index.vot")
OUT_SUMMARY = os.path.join(OUTS, "step04b_spectral_summary.csv")
OUT_FIG = os.path.join(FIGS, "step04b_spectral_index.pdf")

STEEP_BOUNDARY = -0.5
THERMAL_BOUNDARY = -0.2
THERMAL_MAXIMUM = 2.0
MIN_FIT_POINTS = 2
MIN_RELIABLE_POINTS = 4
MAX_ALPHA_ERROR = 0.5
MAX_REDUCED_CHISQ = 10.0
MIN_BAND_SNR = 3.0     # a sub-band must be detected to be fitted
ASKAP_FREQ_GHZ = 0.888

# MeerKAT sub-band centres in MHz.  Channels 8 and 9 are absent: RFI.
CHANNELS_MHZ = {
    "ch2": 908.037, "ch3": 952.342, "ch4": 996.646, "ch5": 1043.46,
    "ch6": 1092.78, "ch7": 1144.61, "ch10": 1317.23, "ch11": 1381.18,
    "ch12": 1448.05, "ch13": 1519.94, "ch14": 1593.92, "ch15": 1656.2,
}


def weighted_fit(freq_ghz, flux, err):
    """Weighted straight-line fit in log-log space.

    Returns (alpha, intercept, sigma_alpha, reduced chi-square, N).  The flux
    errors are converted to log-space errors by err / (S ln10), which is the
    correct first-order propagation as long as the fractional error is small;
    points with a fractional error above unity are not usable anyway and are
    dropped by the positivity test.
    """
    ok = (np.isfinite(freq_ghz) & (freq_ghz > 0) & np.isfinite(flux)
          & (flux > 0) & np.isfinite(err) & (err > 0))
    x = np.log10(freq_ghz[ok])
    y = np.log10(flux[ok])
    sy = err[ok] / (flux[ok] * np.log(10.0))
    if len(x) < MIN_FIT_POINTS:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    w = 1.0 / sy ** 2
    A = np.column_stack([x, np.ones_like(x)])
    try:
        cov = np.linalg.inv(A.T @ (w[:, None] * A))
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    a, c = cov @ (A.T @ (w * y))
    resid = y - (a * x + c)
    dof = len(x) - 2
    rchi = float(np.sum((resid / sy) ** 2) / dof) if dof > 0 else np.nan
    return float(a), float(c), float(np.sqrt(max(cov[0, 0], 0.0))), rchi, len(x)


print("=" * 60)
print("  Step 4b  -  spectral index")
print("=" * 60)

det = Table.read(DET_FILE, format="votable")
mkt5 = Table.read(MKT5_FILE, format="votable")
n = len(det)
print(f"\ndetections : {n}")

# resolve the sub-band column names; astropy suffixes duplicate FIELD ids
bands = {}
for ch, f_mhz in CHANNELS_MHZ.items():
    for fc, ec in ((f"{ch}_int_flux_1", f"{ch}_err_int_flux"),
                   (f"{ch}_int_flux", f"{ch}_err_int_flux_1")):
        if fc in mkt5.colnames and ec in mkt5.colnames:
            bands[ch] = (f_mhz, fc, ec, f"{ch}_peak_flux", f"{ch}_local_rms")
            break
print(f"MeerKAT sub-bands available : {len(bands)} of {len(CHANNELS_MHZ)}")

row5 = np.asarray(det["mkt5_row"], dtype=int)
ask_flux = np.asarray(det["askap_int_mJy"], dtype=float)
ask_err = np.asarray(det["askap_e_int_mJy"], dtype=float)
ask_ok = np.asarray(det["askap_detected"]) == 1

alpha = np.full(n, np.nan)
alpha_err = np.full(n, np.nan)
alpha_c = np.full(n, np.nan)
rchisq = np.full(n, np.nan)
npts = np.zeros(n, dtype=np.int16)
n_mkt_pts = np.zeros(n, dtype=np.int16)
has_askap = np.zeros(n, dtype=np.int16)

for k in range(n):
    f_list, s_list, e_list = [], [], []
    if row5[k] >= 0:
        r = mkt5[row5[k]]
        for f_mhz, fc, ec, pc, rc in bands.values():
            try:
                s = float(r[fc])
                e = float(r[ec])
                pk = float(r[pc]) if pc in mkt5.colnames else np.nan
                rms = float(r[rc]) if rc in mkt5.colnames else np.nan
            except Exception:
                continue
            if not (np.isfinite(s) and s > 0 and np.isfinite(e) and e > 0):
                continue
            # keep only bands in which the source is genuinely detected
            if np.isfinite(pk) and np.isfinite(rms) and rms > 0:
                if pk / rms < MIN_BAND_SNR:
                    continue
            f_list.append(f_mhz / 1000.0)
            s_list.append(s)
            e_list.append(e)
    n_mkt_pts[k] = len(f_list)

    if ask_ok[k] and np.isfinite(ask_flux[k]) and ask_flux[k] > 0 \
            and np.isfinite(ask_err[k]) and ask_err[k] > 0:
        # ASKAP fluxes are in mJy, MeerKAT sub-bands in Jy; the fit only cares
        # about the slope, but the two must be on the same scale for the
        # intercept and the chi-square to mean anything
        f_list.append(ASKAP_FREQ_GHZ)
        s_list.append(ask_flux[k] * 1e-3)
        e_list.append(ask_err[k] * 1e-3)
        has_askap[k] = 1

    a, c, ae, rc, np_used = weighted_fit(np.array(f_list), np.array(s_list),
                                         np.array(e_list))
    alpha[k], alpha_c[k], alpha_err[k], rchisq[k], npts[k] = a, c, ae, rc, np_used

# ------------------------------------------------------------ reliability
status = np.full(n, "no_fit", dtype="U22")
sp_class = np.full(n, "no_alpha", dtype="U12")
for k in range(n):
    if not np.isfinite(alpha[k]):
        continue
    if npts[k] < MIN_RELIABLE_POINTS:
        status[k] = "too_few_points"
    elif not (np.isfinite(alpha_err[k]) and alpha_err[k] < MAX_ALPHA_ERROR):
        status[k] = "alpha_error_too_large"
    elif not (np.isfinite(rchisq[k]) and rchisq[k] < MAX_REDUCED_CHISQ):
        status[k] = "poor_fit"
    elif alpha[k] > THERMAL_MAXIMUM:
        status[k] = "alpha_unphysical"
    else:
        status[k] = "reliable"
        if alpha[k] < STEEP_BOUNDARY:
            sp_class[k] = "steep"
        elif alpha[k] < THERMAL_BOUNDARY:
            sp_class[k] = "uncertain"
        else:
            sp_class[k] = "thermal"

det["alpha"] = Column(alpha)
det["alpha_err"] = Column(alpha_err)
det["alpha_intercept"] = Column(alpha_c)
det["alpha_redchisq"] = Column(rchisq)
det["alpha_n_points"] = Column(npts)
det["alpha_n_meerkat"] = Column(n_mkt_pts)
det["alpha_has_askap"] = Column(has_askap)
det["alpha_status"] = Column(status)
det["sp_class"] = Column(sp_class)
det.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("radio detections", n),
        (f"sub-bands required at S/N >= {MIN_BAND_SNR:.0f}", MIN_RELIABLE_POINTS),
        ("with any fit", int(np.isfinite(alpha).sum())),
        ("reliable fits", int((status == "reliable").sum())),
        ("    thermal (-0.2 to +2.0)", int((sp_class == "thermal").sum())),
        ("    uncertain (-0.5 to -0.2)", int((sp_class == "uncertain").sum())),
        ("    steep (< -0.5)", int((sp_class == "steep").sum())),
        ("rejected, too few points", int((status == "too_few_points").sum())),
        ("rejected, alpha error too large",
         int((status == "alpha_error_too_large").sum())),
        ("rejected, poor fit", int((status == "poor_fit").sum())),
        ("rejected, alpha above +2", int((status == "alpha_unphysical").sum())),
        ("no fit possible", int((status == "no_fit").sum())),
        ("fits including the ASKAP point", int(has_askap.sum()))]
Table(rows=rows, names=("quantity", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                                overwrite=True)
print("\n" + "-" * 54)
for label, v in rows:
    print(f"  {label:<44}{v:>6}")
print("-" * 54)

# ====================================================================== figure
# ------------------------------------- field comparison (see docstring above)
field = Table.read(MKT5_FILE, format="votable")
f_a = np.asarray(field["new_alpha_odr"], dtype=float)
f_ae = np.asarray(field["new_err_alpha_odr"], dtype=float)
f_n = np.asarray(field["new_n_points"], dtype=float)
f_s = np.asarray(field["ch0_int_flux_1"], dtype=float) * 1e3
f_ok = (np.isfinite(f_a) & np.isfinite(f_s) & (f_s > 0) & np.isfinite(f_ae)
        & (f_ae < MAX_ALPHA_ERROR) & (f_n >= MIN_RELIABLE_POINTS))

FLUX_BINS = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1e5)]
bin_lab, pn_med, fld_med, pn_n = [], [], [], []
det_flux = np.asarray(det["flux_1295_mJy"], dtype=float)
for lo, hi in FLUX_BINS:
    m = (status == "reliable") & (det_flux >= lo) & (det_flux < hi)
    fm = f_ok & (f_s >= lo) & (f_s < hi)
    bin_lab.append(f"{lo:g}-{hi:g}" if hi < 1e4 else f">{lo:g}")
    pn_med.append(np.nanmedian(alpha[m]) if m.sum() else np.nan)
    fld_med.append(np.nanmedian(f_a[fm]) if fm.sum() else np.nan)
    pn_n.append(int(m.sum()))

print("\nmedian alpha, our PNe against the general radio field:")
print(f"    {'flux (mJy)':<12}{'N':>5}{'PNe':>9}{'field':>9}{'offset':>9}")
for lab, nn, pm, fmd in zip(bin_lab, pn_n, pn_med, fld_med):
    print(f"    {lab:<12}{nn:>5}{pm:>9.2f}{fmd:>9.2f}{pm - fmd:>9.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.0))
cols = {"thermal": "#22539b", "uncertain": "#e8a33d", "steep": "#c44e52"}

ax = axes[0]
rel = status == "reliable"
bins = np.linspace(-2.5, 2.0, 46)
bottom = np.zeros(len(bins) - 1)
for cls in ("steep", "uncertain", "thermal"):
    m = sp_class == cls
    h, _ = np.histogram(alpha[m], bins=bins)
    ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
           color=cols[cls], label=f"{cls} ({int(m.sum())})")
    bottom += h
for b in (STEEP_BOUNDARY, THERMAL_BOUNDARY):
    ax.axvline(b, color="0.3", ls="--", lw=1.0)
ax.axvline(-0.1, color="k", ls=":", lw=1.0)
ax.text(-0.1, ax.get_ylim()[1] * 0.94, r" optically thin free-free",
        fontsize=7.5, va="top")
ax.set_xlabel(r"spectral index $\alpha$   ($S_\nu \propto \nu^{\alpha}$)")
ax.set_ylabel("number of PNe")
ax.set_title(f"Reliable spectral indices ({int(rel.sum())} of {n})", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.tick_params(labelsize=8)

ax2 = axes[1]
ax2.scatter(np.asarray(det["flux_1295_mJy"])[~rel], alpha[~rel], s=12,
            c="0.8", edgecolors="none", label="not reliable")
for cls in ("thermal", "uncertain", "steep"):
    m = sp_class == cls
    ax2.errorbar(np.asarray(det["flux_1295_mJy"])[m], alpha[m],
                 yerr=alpha_err[m], fmt="o", ms=3.4, lw=0.6, color=cols[cls],
                 alpha=0.85, label=cls)
ax2.axhline(STEEP_BOUNDARY, color="0.3", ls="--", lw=1.0)
ax2.axhline(THERMAL_BOUNDARY, color="0.3", ls="--", lw=1.0)
ax2.set_xscale("log")
ax2.set_ylim(-3.0, 2.5)
ax2.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax2.set_ylabel(r"$\alpha$")
ax2.set_title("Spectral index against brightness", fontsize=10)
ax2.legend(fontsize=8, framealpha=0.9, ncol=2)
ax2.tick_params(labelsize=8)

ax3 = axes[2]
xp = np.arange(len(bin_lab))
ax3.plot(xp, fld_med, "s--", color="0.45", ms=7,
         label="MeerKAT field (mostly radio galaxies)")
ax3.plot(xp, pn_med, "o-", color="#22539b", ms=8, label="LMC PNe, this work")
for i, nn in enumerate(pn_n):
    ax3.annotate(f"N={nn}", (xp[i], pn_med[i]), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=7.5, color="#22539b")
ax3.axhline(-0.1, color="k", ls=":", lw=1.0)
ax3.text(len(bin_lab) - 1, -0.06, "optically thin free-free", ha="right",
         fontsize=7.5, va="bottom")
ax3.set_xticks(xp)
ax3.set_xticklabels(bin_lab)
ax3.set_xlabel("integrated flux density at 1.295 GHz (mJy)")
ax3.set_ylabel(r"median $\alpha$")
ax3.set_title("Where the criterion has discriminating power", fontsize=10)
ax3.legend(fontsize=8, framealpha=0.9, loc="lower right")
ax3.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
