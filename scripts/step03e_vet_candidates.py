"""
step03e_vet_candidates.py
=========================
Step 3b - Vetting the forced-photometry candidates.

Step 3a measures a flux at every undetected parent position and keeps the ones
that clear 5 sigma in both peak and aperture significance.  A significance cut
alone cannot tell a compact PN from a knot of diffuse H II emission or from the
sidelobe of a bright neighbour, so each candidate is checked here on two
grounds:

  1. compactness.  For an unresolved source the aperture flux stops growing
     once the aperture passes about 1.5 beam FWHM.  We measure the flux in
     12 and 24 arcsec radii and take the ratio.  Anything above GROWTH_MAX is
     still gaining flux at 24 arcsec, which means the emission is extended and
     the 12 arcsec measurement is not a source flux at all.

  2. contamination by a bright neighbour.  The 3 sigma MeerKAT catalogue is
     dense enough that most positions on the sky have some catalogued source
     within half an arcminute, so mere proximity means nothing.  What does
     matter is a bright neighbour: a source above BRIGHT_MJY within
     NEIGHBOUR_RAD can put its skirt or a sidelobe on the PN position and
     produce exactly the compact excess we are looking for.

Both tests are written down and applied to every candidate, so the surviving
list is reproducible.  The cutouts are still rendered, because the numbers are
a filter and not a substitute for looking at the images, but the images confirm
the decision rather than making it.

Every candidate that survives is recorded as a tentative detection, not as a
detection equal to the catalogue ones.  The measured false-positive rate from
the Step 3a control fields applies to this list and is quoted with it.

Outputs
-------
    03_Outputs/step03e_visual_detections.vot  the vetted tentative detections
    03_Outputs/step03e_vet_summary.csv        counts behind the paper table
    04_Figures/step03e_candidate_cutouts.pdf  contact sheet of every candidate
    04_Figures/step03e_vetting.pdf            growth curves and outcome

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
import astropy.units as u

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

FP_FILE = os.path.join(OUTS, "step03d_forced_photometry.vot")
IMG_FILE = os.path.join(DATA, "LMC_I_mosaic_ch0_beam.fits")
MKT3_FILE = os.path.join(DATA, "MeerKAT_3sigma.vot")

OUT_TABLE = os.path.join(OUTS, "step03e_visual_detections.vot")
OUT_SUMMARY = os.path.join(OUTS, "step03e_vet_summary.csv")
OUT_SHEET = os.path.join(FIGS, "step03e_candidate_cutouts.pdf")
OUT_FIG = os.path.join(FIGS, "step03e_vetting.pdf")

BEAM_FWHM = 8.0
APER_RAD = 12.0
WIDE_RAD = 24.0
GROWTH_MAX = 1.6      # S(24") / S(12"); an unresolved source gives about 1.0
NEIGHBOUR_RAD = 60.0  # arcsec
BRIGHT_MJY = 5.0      # mJy/beam; only neighbours this bright can contaminate
STAMP = 90.0          # arcsec, cutout side

print("=" * 60)
print("  Step 3b  -  vetting the forced-photometry candidates")
print("=" * 60)

fp = Table.read(FP_FILE, format="votable")
cand = fp[np.asarray(fp["fp_candidate"]) == 1]
print(f"\ncandidates from Step 3a : {len(cand)}")

hdu = fits.open(IMG_FILE, memmap=True)[0]
img = hdu.data
while img.ndim > 2:
    img = img[0]
wcs = WCS(hdu.header).celestial
pscale = float(np.abs(wcs.pixel_scale_matrix[1, 1]) * 3600.0)
beam_area_pix = np.pi * BEAM_FWHM ** 2 / (4.0 * np.log(2) * pscale ** 2)

pos = SkyCoord(ra=np.asarray(cand["fp_ra"]) * u.deg,
               dec=np.asarray(cand["fp_dec"]) * u.deg)

# --------------------------------------------------- catalogued neighbours
mkt3 = Table.read(MKT3_FILE, format="votable")
c_mkt = SkyCoord(ra=np.asarray(mkt3["ra"], dtype=float) * u.deg,
                 dec=np.asarray(mkt3["dec"], dtype=float) * u.deg)
peak_mkt = np.asarray(mkt3["peak_flux"], dtype=float) * 1e3
bright = peak_mkt >= BRIGHT_MJY
c_bright = c_mkt[bright]
print(f"catalogued sources above {BRIGHT_MJY:.0f} mJy/beam : {int(bright.sum())}")
_, sep_nb, _ = pos.match_to_catalog_sky(c_bright)
sep_nb = sep_nb.arcsec
neighbour = sep_nb <= NEIGHBOUR_RAD
print(f"with a bright neighbour inside {NEIGHBOUR_RAD:.0f}\" : "
      f"{int(neighbour.sum())}")

# ------------------------------------------------------------ growth curves
n = len(cand)
stamps = []
centres = []
growth = np.full(n, np.nan)
f12 = np.full(n, np.nan)
f24 = np.full(n, np.nan)
box_pix = int(round(STAMP / pscale))

for i in range(n):
    cut = Cutout2D(img, pos[i], (box_pix, box_pix), wcs=wcs, mode="partial",
                   fill_value=np.nan)
    st = np.asarray(cut.data, dtype=float)
    stamps.append(st)
    ny, nx = st.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    # take the centre from the WCS rather than assuming the middle pixel;
    # a partial cutout at the mosaic edge is not centred on the request
    cx, cy = cut.wcs.world_to_pixel(pos[i])
    centres.append((float(cx), float(cy)))
    r = np.hypot(xx - cx, yy - cy) * pscale
    edge = (r >= 36.0) & np.isfinite(st)
    bkg = float(np.median(st[edge])) if edge.sum() > 30 else 0.0
    f12[i] = float(np.nansum(st[(r <= APER_RAD) & np.isfinite(st)] - bkg)) / beam_area_pix
    f24[i] = float(np.nansum(st[(r <= WIDE_RAD) & np.isfinite(st)] - bkg)) / beam_area_pix
    if f12[i] > 0:
        growth[i] = f24[i] / f12[i]

extended = np.isfinite(growth) & (growth > GROWTH_MAX)
print(f"extended (S24/S12 > {GROWTH_MAX}) : {int(extended.sum())}")

verdict = np.full(n, "tentative", dtype="U12")
verdict[extended] = "extended"
verdict[neighbour & ~extended] = "confused"
keep = verdict == "tentative"
print(f"surviving as tentative detections : {int(keep.sum())}")

# ------------------------------------------------------------------ outputs
out = cand.copy()
out["vet_f12_mJy"] = Column((f12 * 1e3).astype("f4"), unit="mJy")
out["vet_f24_mJy"] = Column((f24 * 1e3).astype("f4"), unit="mJy")
out["vet_growth"] = Column(growth.astype("f4"))
out["vet_bright_neighbour_sep"] = Column(sep_nb.astype("f4"), unit="arcsec")
out["fp_visual"] = Column(verdict)
out["visual_detected"] = Column(keep.astype(np.int16))
out.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("Step 3a candidates", n),
        (f"rejected, extended (S24/S12 > {GROWTH_MAX})", int(extended.sum())),
        (f"rejected, bright neighbour < {NEIGHBOUR_RAD:.0f}\"",
         int((verdict == "confused").sum())),
        ("tentative detections retained", int(keep.sum()))]
Table(rows=rows, names=("outcome", "N")).write(OUT_SUMMARY, format="ascii.csv",
                                               overwrite=True)
print("\n" + "-" * 52)
for label, v in rows:
    print(f"  {label:<42}{v:>6}")
print("-" * 52)

# ------------------------------------------------------------ contact sheet
ncol = 6
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(1.85 * ncol, 2.05 * nrow))
axes = np.atleast_2d(axes).ravel()
z = ZScaleInterval()
colours = {"tentative": "#2c7d3f", "extended": "#c44e52", "confused": "#d68910"}

for i in range(len(axes)):
    ax = axes[i]
    ax.set_xticks([])
    ax.set_yticks([])
    if i >= n:
        ax.axis("off")
        continue
    st = stamps[i]
    finite = st[np.isfinite(st)]
    lo, hi = z.get_limits(finite) if finite.size else (0, 1)
    ax.imshow(st, origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    cx, cy = centres[i]
    ax.add_patch(plt.Circle((cx, cy), APER_RAD / pscale,
                            fill=False, ec="w", lw=0.9))
    ax.add_patch(plt.Circle((cx, cy), WIDE_RAD / pscale,
                            fill=False, ec="w", lw=0.5, ls=":"))
    v = verdict[i]
    ax.set_title(f"{cand['PN_ID'][i]}  {cand['fp_snr'][i]:.1f}$\\sigma$",
                 fontsize=7.5, pad=2.5)
    for sp in ax.spines.values():
        sp.set_color(colours[v])
        sp.set_linewidth(2.0)
    ax.text(0.03, 0.03, v, transform=ax.transAxes, fontsize=6.5, color="w",
            bbox=dict(fc=colours[v], ec="none", alpha=0.85,
                      boxstyle="round,pad=0.2"))

fig.suptitle(f"Forced-photometry candidates, MeerKAT 1.295 GHz  "
             f"({STAMP:.0f}\" on a side; circles 12\" and 24\")", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(OUT_SHEET, bbox_inches="tight", dpi=170)
plt.close(fig)

# --------------------------------------------------------------- vet figure
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
for v, c in colours.items():
    m = verdict == v
    ax.scatter(np.asarray(cand["fp_snr"])[m], growth[m], s=26, c=c,
               edgecolors="none", label=f"{v} ({int(m.sum())})")
ax.axhline(GROWTH_MAX, color="0.35", ls="--", lw=1.1)
ax.axhline(1.0, color="0.75", lw=0.8)
ax.set_xlabel("peak signal-to-noise ratio")
ax.set_ylabel(r"$S(24^{\prime\prime})\,/\,S(12^{\prime\prime})$")
ax.set_title("Compactness of the candidates", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.tick_params(labelsize=8)

ax2 = axes[1]
labels = ["tentative", "extended", "confused"]
vals = [int((verdict == v).sum()) for v in labels]
bars = ax2.bar(labels, vals, color=[colours[v] for v in labels], width=0.6)
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
             ha="center", fontsize=9)
ax2.set_ylabel("number of candidates")
ax2.set_ylim(0, max(vals) * 1.2)
ax2.set_title(f"Outcome of vetting {n} candidates", fontsize=10)
ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_SHEET)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
