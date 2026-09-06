"""
step2a_crossmatch_meerkat.py
============================
Step 2a (MeerKAT) — Cross-match the unified Reid LMC PN catalogue against the
MeerKAT 1.295 GHz radio catalogues.

Inputs (01_Data/)
-----------------
    03_Outputs/step1_parent_catalogue.vot — Unified Reid catalogue from Step 01a
                               Columns: RA, Dec, reid_class, has_phot, photometry
    MeerKAT_3sigma.vot       — Aegean 3σ broadband catalogue (cross-match target)
    MeerKAT_Final_5sigma.vot — Aegean 5σ multi-band catalogue (spectral index)

Outputs
-------
    03_Outputs/step2a_meerkat_matched.vot    — Matched sources + MeerKAT columns
    03_Outputs/step2a_meerkat_unmatched.vot  — Unmatched sources
    05_Figures/step2a_meerkat_offsets.pdf    — Positional offset QC (hexbin)

Match parameters
----------------
    Primary match radius : 4.5 arcsec  (3 MeerKAT pixels @ 1.5"/pix)
    5σ lookup radius     : 4.5 arcsec

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from matplotlib.offsetbox import AnchoredText
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config
from _support.plot_style import apply_paper_style

# The VOTables carry non-standard FIELD ids; nothing else is silenced.
warnings.simplefilter("ignore", AstropyWarning)
apply_paper_style()

cfg = config.load()
MKT3_FILE = cfg.data("meerkat_3sigma")
MKT5_FILE = cfg.data("meerkat_5sigma")
REID_FILE = cfg.out("step1_parent_catalogue.vot")

OUT_MATCHED   = cfg.out("step2a_meerkat_matched.vot")
OUT_UNMATCHED = cfg.out("step2a_meerkat_unmatched.vot")
OUT_STATS     = cfg.out("step2a_meerkat_match_stats.csv")
FIG_OFFSETS   = cfg.fig("step2a_meerkat_offsets.pdf")

MATCH_RAD  = cfg["crossmatch"]["accept_arcsec"]
LOOKUP_RAD = cfg["crossmatch"]["accept_arcsec"]
AREA_DEG2  = cfg["crossmatch"]["meerkat_area_deg2"]

STYLE  = "ggplot"
HEXBIN = {"gridsize": 30, "cmap": "plasma", "mincnt": 1}
HIST   = {"bins": "auto", "color": "#D32F2F", "alpha": 0.55}
XLIM   = (-5.0, 5.0)
YLIM   = (-5.0, 5.0)
FONTS  = {"axes_label": 13, "axes_title": 13, "tick": 11, "cbar": 11, "stats": 10}

print("step2a: cross-matching the parent catalogue against MeerKAT 1.295 GHz")

print("\n[1] Loading catalogues...")
print(f"    Using : {os.path.basename(REID_FILE)}")

reid = Table.read(REID_FILE, format="votable")
mkt3 = Table.read(MKT3_FILE, format="votable")
mkt5 = Table.read(MKT5_FILE, format="votable")

print(f"    Reid catalogue      : {len(reid):>6d} sources")
print(f"    MeerKAT 3σ          : {len(mkt3):>6d} sources")
print(f"    MeerKAT 5σ          : {len(mkt5):>6d} sources")

# Coordinates
ra_reid  = np.array(reid["RA"],  dtype=float)
dec_reid = np.array(reid["Dec"], dtype=float)
cls_reid = np.array(reid["reid_class"], dtype=str)

valid_reid = np.isfinite(ra_reid) & np.isfinite(dec_reid)
coords_reid = SkyCoord(ra=ra_reid[valid_reid]  * u.deg,
                       dec=dec_reid[valid_reid] * u.deg)
reid_valid_idx = np.where(valid_reid)[0]

ra_mkt3  = np.array(mkt3["ra"],  dtype=float)
dec_mkt3 = np.array(mkt3["dec"], dtype=float)
coords_mkt3 = SkyCoord(ra=ra_mkt3 * u.deg, dec=dec_mkt3 * u.deg)

# Cross-match
print(f"\n[2] Cross-matching Reid x MeerKAT 3σ (≤ {MATCH_RAD}\")...")
idx_m, sep_m, _ = coords_reid.match_to_catalog_sky(coords_mkt3)
sep_m_arcsec    = sep_m.to(u.arcsec).value

matched_flag  = np.zeros(len(reid), dtype=bool)
sep_full      = np.full(len(reid), np.nan)
mkt3_idx_full = np.full(len(reid), -1, dtype=int)

for i, (midx, sep) in enumerate(zip(idx_m, sep_m_arcsec)):
    ri = reid_valid_idx[i]
    sep_full[ri] = sep
    if sep <= MATCH_RAD:
        matched_flag[ri]  = True
        mkt3_idx_full[ri] = midx

n_matched   = int(matched_flag.sum())
n_unmatched = len(reid) - n_matched
print(f"    Matched  (≤{MATCH_RAD}\"): {n_matched}")
print(f"    Unmatched          : {n_unmatched}")

# Nearest-neighbour matching can in principle hand the same radio source to two
# PNe.  It does not here, and the pipeline would be wrong if it ever did.
used = mkt3_idx_full[matched_flag]
assert len(set(used)) == len(used), "one MeerKAT source matched to two PNe"

# Chance coincidences expected if the radio sources were distributed at random.
density = len(mkt3) / AREA_DEG2                       # per deg2
chance = np.pi * (MATCH_RAD / 3600.0) ** 2 * density * len(reid)
print(f"    Expected by chance : {chance:.2f} over the whole parent sample")

print("\n    Detection by optical class:")
print(f"    {'Class':<12} {'Total':>6} {'Detected':>9} {'Fraction':>9}")
print(f"    {'-'*40}")
for cls in ["True", "Known", "Likely", "Possible", "Unknown"]:
    mask = cls_reid == cls
    tot  = int(mask.sum())
    det  = int((mask & matched_flag).sum())
    frac = det / tot * 100 if tot > 0 else 0.0
    print(f"    {cls:<12} {tot:>6} {det:>9} {frac:>8.1f}%")

# SNR
print("\n[3] Computing SNR...")
peak_all = np.array(mkt3["peak_flux"],    dtype=float)
rms_all  = np.array(mkt3["local_rms"],    dtype=float)
int_all  = np.array(mkt3["int_flux"],     dtype=float)
errint   = np.array(mkt3["err_int_flux"], dtype=float)
maj_all  = np.array(mkt3["a"],            dtype=float)

snr_arr = np.full(len(reid), np.nan)
for ri in range(len(reid)):
    midx = mkt3_idx_full[ri]
    if midx >= 0 and rms_all[midx] > 0:
        snr_arr[ri] = peak_all[midx] / rms_all[midx]

n_5sig = int(np.sum(snr_arr[matched_flag] >= 5.0))
n_3sig = int(np.sum((snr_arr[matched_flag] >= 3.0) & (snr_arr[matched_flag] < 5.0)))
print(f"    SNR >= 5   : {n_5sig}")
print(f"    3 <= SNR < 5 : {n_3sig}")

# Spectral index
print(f"\n[4] Spectral index lookup x MeerKAT 5σ (≤ {LOOKUP_RAD}\")...")

def find_col(tbl, candidates):
    for c in candidates:
        if c in tbl.colnames:
            return c
    raise KeyError(f"None of {candidates} found. Available: {tbl.colnames[:15]}")

ra5  = find_col(mkt5, ["RAJ2000_deg",  "dRAJ2000",  "ra"])
dec5 = find_col(mkt5, ["DECJ2000_deg", "dDECJ2000", "dec"])
sp   = find_col(mkt5, ["spindex",   "new_alpha_odr"])
spe  = find_col(mkt5, ["spindex_err","new_err_alpha_odr"])
rchi = find_col(mkt5, ["redchisq",  "new_red_chisq_odr"])
npt  = find_col(mkt5, ["n_points",  "new_n_points"])
print(f"    RA={ra5}, Dec={dec5}, alpha={sp}")

coords_mkt5 = SkyCoord(ra=np.array(mkt5[ra5],  dtype=float) * u.deg,
                       dec=np.array(mkt5[dec5], dtype=float) * u.deg)

valid_matched = np.where(matched_flag & np.isfinite(ra_reid))[0]
coords_matched = SkyCoord(
    ra=ra_reid[valid_matched]  * u.deg,
    dec=dec_reid[valid_matched] * u.deg
)
idx_5, sep_5, _ = coords_matched.match_to_catalog_sky(coords_mkt5)
sep_5_arcsec    = sep_5.to(u.arcsec).value

alpha_arr   = np.full(len(reid), np.nan)
alphaerr    = np.full(len(reid), np.nan)
redchisq    = np.full(len(reid), np.nan)
npoints     = np.zeros(len(reid), dtype=np.int64)
spflag      = np.full(len(reid), "", dtype="U16")

for j, (ri, s5idx, s5sep) in enumerate(zip(valid_matched, idx_5, sep_5_arcsec)):
    if s5sep <= LOOKUP_RAD:
        alpha_arr[ri] = float(mkt5[sp][s5idx])
        alphaerr[ri]  = float(mkt5[spe][s5idx])
        redchisq[ri]  = float(mkt5[rchi][s5idx])
        npoints[ri]   = int(mkt5[npt][s5idx])
        spflag[ri]    = "5sigma"
    else:
        spflag[ri]    = "3sigma_only"

n_with = int(np.sum(spflag == "5sigma"))
n_no   = int(np.sum(spflag == "3sigma_only"))
print(f"    With spindex (5σ)  : {n_with}")
print(f"    3σ-only (no spidx) : {n_no}")

# Build output tables
print("\n[5] Writing catalogues...")

def mkt3_col(src, dtype=float):
    out = np.full(len(reid), np.nan if dtype == float else 0, dtype=dtype)
    for ri in range(len(reid)):
        midx = mkt3_idx_full[ri]
        if midx >= 0:
            out[ri] = src[midx]
    return out

out_tbl = reid.copy()
out_tbl.add_column(Column(sep_full.astype("f4"),          name="separation_arcsec"))
out_tbl.add_column(Column(
    np.where(matched_flag, "MATCHED", "UNMATCHED"),        name="match_flag"))
out_tbl.add_column(Column(mkt3_col(ra_mkt3),              name="mkt_ra"))
out_tbl.add_column(Column(mkt3_col(dec_mkt3),             name="mkt_dec"))
out_tbl.add_column(Column(mkt3_col(peak_all),             name="mkt_peak_flux_Jy"))
out_tbl.add_column(Column(mkt3_col(int_all),              name="mkt_int_flux_Jy"))
out_tbl.add_column(Column(mkt3_col(errint),               name="mkt_err_int_flux_Jy"))
out_tbl.add_column(Column(mkt3_col(rms_all),              name="mkt_local_rms_Jy"))
out_tbl.add_column(Column(mkt3_col(maj_all),              name="mkt_maj_arcsec"))
out_tbl.add_column(Column(snr_arr.astype("f4"),           name="mkt_snr"))
out_tbl.add_column(Column(alpha_arr,                      name="alpha"))
out_tbl.add_column(Column(alphaerr,                       name="alpha_err"))
out_tbl.add_column(Column(redchisq,                       name="alpha_redchisq"))
out_tbl.add_column(Column(npoints,                        name="alpha_n_points"))
out_tbl.add_column(Column(spflag,                         name="spindex_flag"))

matched_tbl   = out_tbl[matched_flag]
unmatched_tbl = out_tbl[~matched_flag]

# Unmatched: strip MeerKAT-specific columns
drop = ["mkt_ra","mkt_dec","mkt_peak_flux_Jy","mkt_int_flux_Jy",
        "mkt_err_int_flux_Jy","mkt_local_rms_Jy","mkt_maj_arcsec",
        "mkt_snr","alpha","alpha_err","alpha_redchisq","alpha_n_points","spindex_flag"]
unmatched_out = unmatched_tbl[[c for c in unmatched_tbl.colnames if c not in drop]]

matched_tbl.write(OUT_MATCHED,     format="votable", overwrite=True)
unmatched_out.write(OUT_UNMATCHED, format="votable", overwrite=True)
print(f"    {os.path.basename(OUT_MATCHED)}   -> {n_matched} rows")
print(f"    {os.path.basename(OUT_UNMATCHED)} -> {n_unmatched} rows")

# Positional offset figure
print("\n[6] Generating offset figure...")

plot_rows = matched_flag & np.isfinite(ra_reid) & np.isfinite(dec_reid)
ra_opt  = ra_reid[plot_rows]
dec_opt = dec_reid[plot_rows]
ra_rad  = mkt3_col(ra_mkt3)[plot_rows]
dec_rad = mkt3_col(dec_mkt3)[plot_rows]

c_opt = SkyCoord(ra=ra_opt * u.deg, dec=dec_opt * u.deg)
c_rad = SkyCoord(ra=ra_rad * u.deg, dec=dec_rad * u.deg)
dra_raw, ddec_raw = c_opt.spherical_offsets_to(c_rad)
x = dra_raw.to(u.arcsec).value
y = ddec_raw.to(u.arcsec).value
finite = np.isfinite(x) & np.isfinite(y)
x = x[finite]; y = y[finite]

plt.style.use(STYLE)
fig = plt.figure(figsize=(7, 7))
gs_outer = GridSpec(1, 1, figure=fig)
gs_inner = GridSpecFromSubplotSpec(
    2, 3, subplot_spec=gs_outer[0],
    width_ratios=[6, 1, 0.35], height_ratios=[1, 6],
    wspace=0.10, hspace=0.10,
)
ax_histx = fig.add_subplot(gs_inner[0, 0])
ax_main  = fig.add_subplot(gs_inner[1, 0])
ax_histy = fig.add_subplot(gs_inner[1, 1], sharey=ax_main)
ax_cbar  = fig.add_subplot(gs_inner[1, 2])

hb = ax_main.hexbin(x, y, **HEXBIN)
ax_main.set_xlabel("ΔRA · cos(δ)  (arcsec)", fontsize=FONTS["axes_label"])
ax_main.set_ylabel("Δδ  (arcsec)",            fontsize=FONTS["axes_label"])
ax_main.set_xlim(*XLIM); ax_main.set_ylim(*YLIM)
ax_main.axhline(0, color="black", lw=1.5)
ax_main.axvline(0, color="black", lw=1.5)
ax_main.grid(axis="both", linestyle="--", alpha=0.3)

ax_histx.hist(x, **HIST)
ax_histx.axvline(np.mean(x), color="black", ls="--", lw=1.2)
ax_histx.set_ylabel("Counts", fontsize=FONTS["tick"])
ax_histx.set_xlim(*XLIM)
plt.setp(ax_histx.get_xticklabels(), visible=False)

ax_histy.hist(y, **HIST, orientation="horizontal")
ax_histy.axhline(np.mean(y), color="black", ls="--", lw=1.2)
ax_histy.set_xlabel("Counts", fontsize=FONTS["tick"])
ax_histy.set_ylim(*YLIM)
plt.setp(ax_histy.get_yticklabels(), visible=False)

cbar = fig.colorbar(hb, cax=ax_cbar)
cbar.set_label("Counts per bin", fontsize=FONTS["cbar"])
cbar.ax.tick_params(labelsize=FONTS["tick"])

stats_txt = "\n".join([
    fr"$\langle\Delta\mathrm{{RA}}\rangle = {np.mean(x):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\mathrm{{RA}}) = {np.std(x):.2f}^{{\prime\prime}}$",
    fr"$\langle\Delta\delta\rangle = {np.mean(y):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\delta) = {np.std(y):.2f}^{{\prime\prime}}$",
    f"$N = {len(x)}$",
])
at = AnchoredText(stats_txt, prop=dict(size=FONTS["stats"]),
                  frameon=True, loc="upper left")
at.patch.set_boxstyle("round,pad=0.3"); at.patch.set_alpha(0.9)
ax_main.add_artist(at)

# NO title — added in Overleaf
plt.savefig(FIG_OFFSETS, format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    {os.path.basename(FIG_OFFSETS)}  done")

sep_matched = sep_full[matched_flag]
stats = Table()
stats["quantity"] = ["n_parent", "n_matched", "n_unmatched", "median_offset_arcsec",
                     "mean_dra_cosdec_arcsec", "mean_ddec_arcsec",
                     "std_dra_cosdec_arcsec", "std_ddec_arcsec",
                     "expected_chance_matches", "n_snr_ge_5", "n_snr_3_to_5"]
stats["value"] = [len(reid), n_matched, n_unmatched, float(np.nanmedian(sep_matched)),
                  float(np.mean(x)), float(np.mean(y)),
                  float(np.std(x)), float(np.std(y)),
                  float(chance), n_5sig, n_3sig]
stats.write(OUT_STATS, format="ascii.csv", overwrite=True)
print(f"    {os.path.basename(OUT_STATS)}")

print("\nstep2a complete")
print(f"  Reid sources   : {len(reid)}")
print(f"  Matched        : {n_matched} ({n_matched/len(reid)*100:.1f}%)")
print(f"  Unmatched      : {n_unmatched}")
print(f"  With spindex   : {n_with}")
print(f"  Median offset  : {np.nanmedian(sep_matched):.2f} arcsec")
print(f"  SNR >= 5       : {n_5sig}")
