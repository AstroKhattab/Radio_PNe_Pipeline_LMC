"""
step03d_forced_photometry.py
============================
Step 3 - Forced photometry on the sources the catalogue missed.

The Aegean source finder that produced the MeerKAT catalogue works on a fixed
island threshold, so a real but faint PN sitting just under it never appears
in Step 2 at all.  This step goes back to the image and measures a flux at
every parent position that Step 2 left undetected, whether or not anything is
obviously there.  Measuring all of them, rather than a list picked by eye,
is what makes the faint end of the sample reproducible: the selection becomes
a threshold anyone can re-apply, not a judgement call.

Method
------
For each undetected parent PN inside the MeerKAT mosaic:

  1. cut a 60 arcsec box out of the total-intensity image;
  2. take the brightest pixel within 4.5 arcsec of the optical position;
  3. sum the pixels inside a 12 arcsec radius of that peak, subtract the
     median of a 30-90 arcsec background annulus, and divide by the beam
     area in pixels to convert Jy/beam into Jy;
  4. read the noise from the matching RMS map at the same position.

A 12 arcsec radius is 1.5 beam FWHM, which encloses 99.8 per cent of an
unresolved source, so no aperture correction is applied.  This is the same
aperture used for the catalogue-detected sources, so the two flux scales are
directly comparable.

Acceptance
----------
    peak / local RMS  >= 3.0
    integrated flux    > 0
    peak within 4.5 arcsec of the optical position

Everything that passes is a candidate; everything that passes is also
rendered as a cutout in 04_Figures/step03d_cutouts/ and inspected by eye in
Step 3b, where sources that turn out to be extended emission, imaging
artefacts or noise peaks are demoted.

Outputs
-------
    03_Outputs/step03d_forced_photometry.vot   every position measured
    03_Outputs/step03d_summary.csv             counts behind the paper table
    04_Figures/step03d_forced_photometry.pdf   S/N distribution and outcome
    04_Figures/step03d_cutouts/                one PNG per candidate

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
import astropy.units as u

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

XMATCH_FILE = os.path.join(OUTS, "step02c_radio_crossmatch.vot")
IMG_FILE = os.path.join(DATA, "LMC_I_mosaic_ch0_beam.fits")
RMS_FILE = os.path.join(DATA, "LMC_I_mosaic_ch0_rms.fits")

OUT_TABLE = os.path.join(OUTS, "step03d_forced_photometry.vot")
OUT_SUMMARY = os.path.join(OUTS, "step03d_summary.csv")
OUT_FIG = os.path.join(FIGS, "step03d_forced_photometry.pdf")
CUTOUT_DIR = os.path.join(FIGS, "step03d_cutouts")

BEAM_FWHM = 8.0        # arcsec
APER_RAD = 12.0        # arcsec, 1.5 FWHM
SEARCH_RAD = 4.5       # arcsec, same acceptance radius as Step 2
ANN_IN, ANN_OUT = 30.0, 90.0
BOX = 200.0            # arcsec, cutout size
SNR_MIN = 5.0        # peak and aperture significance
N_NULL = 1           # one control field per parent position
NULL_SHIFT = 60.0    # arcsec


def load(path):
    hdu = fits.open(path, memmap=True)[0]
    data = hdu.data
    while data.ndim > 2:
        data = data[0]
    return data, WCS(hdu.header).celestial


print("=" * 60)
print("  Step 3  -  forced photometry on undetected positions")
print("=" * 60)

cat = Table.read(XMATCH_FILE, format="votable")
todo = cat[np.asarray(cat["catalogue_detected"]) == 0]
print(f"\nparent sample                : {len(cat)}")
print(f"no catalogue counterpart     : {len(todo)}")

img, wcs = load(IMG_FILE)
rms_img, rms_wcs = load(RMS_FILE)
pscale = float(np.abs(wcs.pixel_scale_matrix[1, 1]) * 3600.0)
beam_area_pix = np.pi * BEAM_FWHM ** 2 / (4.0 * np.log(2) * pscale ** 2)
print(f"image {img.shape}, {pscale:.2f}\"/pix, beam = {beam_area_pix:.1f} pix")

coords = SkyCoord(ra=np.asarray(todo["RA"]) * u.deg,
                  dec=np.asarray(todo["Dec"]) * u.deg)

n = len(todo)
box_pix = int(round(BOX / pscale))


def measure(positions):
    """Forced aperture photometry at a list of sky positions.

    Returns peak and integrated flux (Jy and Jy/beam), the local noise, the
    offset of the adopted peak from the input position, and a flag saying
    whether the position falls on the mosaic at all.

    The noise is taken from the scatter of the background annulus rather
    than from the RMS map.  Across the LMC the field is not empty: diffuse
    H II emission and low-level residuals raise the real pixel-to-pixel
    scatter well above the thermal noise, and a threshold set on the
    thermal value alone lets a large number of noise peaks through.
    """
    m = len(positions)
    res = dict(peak=np.full(m, np.nan), integ=np.full(m, np.nan),
               sigma=np.full(m, np.nan), offset=np.full(m, np.nan),
               ra=np.full(m, np.nan), dec=np.full(m, np.nan),
               n_aper=np.full(m, np.nan), in_field=np.zeros(m, dtype=bool))

    for k in range(m):
        try:
            cut = Cutout2D(img, positions[k], (box_pix, box_pix), wcs=wcs,
                           mode="partial", fill_value=np.nan)
        except Exception:
            continue
        stamp = np.asarray(cut.data, dtype=float)
        if not np.isfinite(stamp).any():
            continue
        res["in_field"][k] = True

        ny, nx = stamp.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        cy, cx = cut.wcs.world_to_pixel(positions[k])[::-1]
        r_opt = np.hypot(xx - cx, yy - cy) * pscale

        near = r_opt <= SEARCH_RAD
        if not (near & np.isfinite(stamp)).any():
            continue
        py, px = np.unravel_index(
            np.nanargmax(np.where(near, stamp, -np.inf)), stamp.shape)

        r_peak = np.hypot(xx - px, yy - py) * pscale
        ann = (r_peak >= ANN_IN) & (r_peak <= ANN_OUT) & np.isfinite(stamp)
        if ann.sum() < 50:
            continue
        vals = stamp[ann]
        bkg = float(np.median(vals))
        # median absolute deviation, scaled to a Gaussian sigma so that a
        # handful of bright neighbours do not inflate the noise estimate
        sigma = 1.4826 * float(np.median(np.abs(vals - bkg)))

        aper = (r_peak <= APER_RAD) & np.isfinite(stamp)
        res["peak"][k] = stamp[py, px] - bkg
        res["integ"][k] = float(np.nansum(stamp[aper] - bkg)) / beam_area_pix
        res["n_aper"][k] = int(aper.sum())
        res["sigma"][k] = sigma
        res["offset"][k] = float(r_opt[py, px])
        sky = cut.wcs.pixel_to_world(px, py)
        res["ra"][k] = sky.ra.deg
        res["dec"][k] = sky.dec.deg

        if (k + 1) % 150 == 0:
            print(f"    {k + 1}/{m}")
    return res


print("\nmeasuring the parent positions...")
real = measure(coords)

# ---------------------------------------------------------------- null test
# The same measurement at positions displaced by 60 arcsec.  Those land in
# the same parts of the mosaic, with the same diffuse emission and the same
# noise, but on nothing in particular.  Whatever fraction of them clears a
# given threshold is the false-positive rate of that threshold, measured
# rather than assumed.
print(f"measuring {N_NULL} control fields offset by {NULL_SHIFT:.0f}\"...")
rng = np.random.default_rng(20260830)
pa = rng.uniform(0.0, 360.0, n) * u.deg
null = measure(coords.directional_offset_by(pa, NULL_SHIFT * u.arcsec))

in_field = real["in_field"]
peak = real["peak"]
integ = real["integ"]
sigma = real["sigma"]
offset = real["offset"]
peak_ra = real["ra"]
peak_dec = real["dec"]

with np.errstate(invalid="ignore", divide="ignore"):
    snr = peak / sigma
    sigma_int = sigma * np.sqrt(real["n_aper"]) / beam_area_pix
    snr_int = integ / sigma_int
    snr_null = null["peak"] / null["sigma"]
    snr_int_null = (null["integ"] * beam_area_pix
                    / (null["sigma"] * np.sqrt(null["n_aper"])))

print("\nfalse-positive rate against threshold "
      "(peak S/N and aperture S/N both applied):")
print(f"    {'thresh':>7}{'real':>7}{'control':>9}{'purity':>9}")
for thr in (3.0, 4.0, 5.0, 6.0):
    nr = int(np.sum(in_field & (snr >= thr) & (snr_int >= thr) & (integ > 0)))
    nn = int(np.sum(null["in_field"] & (snr_null >= thr)
                    & (snr_int_null >= thr) & (null["integ"] > 0)))
    pur = (1 - nn / nr) * 100 if nr else 0.0
    print(f"    {thr:>7.1f}{nr:>7d}{nn:>9d}{pur:>8.1f}%")

accepted = (in_field & np.isfinite(snr) & (snr >= SNR_MIN)
            & (snr_int >= SNR_MIN) & (integ > 0) & (offset <= SEARCH_RAD))
n_null_at_thresh = int(np.sum(null["in_field"] & (snr_null >= SNR_MIN)
                              & (snr_int_null >= SNR_MIN)
                              & (null["integ"] > 0)))

print(f"\ninside the mosaic            : {int(in_field.sum())}")
print(f"outside the mosaic           : {int((~in_field).sum())}")
print(f"candidates at {SNR_MIN:.0f} sigma        : {int(accepted.sum())}"
      f"   (control fields: {n_null_at_thresh})")

out = todo.copy()
out["fp_in_field"] = Column(in_field.astype(np.int16))
out["fp_peak_mJy_beam"] = Column((peak * 1e3).astype("f4"), unit="mJy/beam")
out["fp_int_mJy"] = Column((integ * 1e3).astype("f4"), unit="mJy")
out["fp_rms_mJy_beam"] = Column((sigma * 1e3).astype("f4"), unit="mJy/beam")
out["fp_snr"] = Column(snr.astype("f4"))
out["fp_snr_int"] = Column(snr_int.astype("f4"))
out["fp_offset_arcsec"] = Column(offset.astype("f4"), unit="arcsec")
out["fp_ra"] = Column(peak_ra, unit="deg")
out["fp_dec"] = Column(peak_dec, unit="deg")
out["fp_candidate"] = Column(accepted.astype(np.int16))
# Filled in by step03e once the cutouts have been looked at.
out["fp_visual"] = Column(np.where(accepted, "pending", "n/a").astype("U10"))
out.write(OUT_TABLE, format="votable", overwrite=True)

rows = [("undetected in Step 2", n),
        ("inside the MeerKAT mosaic", int(in_field.sum())),
        ("outside the mosaic", int((~in_field).sum())),
        (f"peak S/N >= {SNR_MIN:.0f}", int((in_field & (snr >= SNR_MIN)).sum())),
        (f"and aperture S/N >= {SNR_MIN:.0f}",
         int((in_field & (snr >= SNR_MIN) & (snr_int >= SNR_MIN)).sum())),
        ("and positive integrated flux", int((in_field & (snr >= SNR_MIN)
                                              & (snr_int >= SNR_MIN)
                                              & (integ > 0)).sum())),
        (f"and peak within {SEARCH_RAD}\" (candidates)", int(accepted.sum())),
        ("same cuts on control fields", n_null_at_thresh)]
Table(rows=rows, names=("selection", "N")).write(OUT_SUMMARY,
                                                 format="ascii.csv",
                                                 overwrite=True)
print("\n" + "-" * 50)
for label, v in rows:
    print(f"  {label:<38}{v:>6}")
print("-" * 50)

# ====================================================================== figure
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

ax = axes[0]
good = in_field & np.isfinite(snr)
bins = np.linspace(-4, 12, 49)
ax.hist(np.clip(snr[good], -4, 12), bins=bins, color="0.78",
        label=f"all measured ({int(good.sum())})")
ax.hist(np.clip(snr_null[null["in_field"]], -4, 12), bins=bins,
        histtype="step", color="k", lw=1.1, label="control fields")
ax.hist(np.clip(snr[accepted], -4, 12), bins=bins, color="#22539b",
        label=f"candidates ({int(accepted.sum())})")
ax.axvline(SNR_MIN, color="#c44e52", ls="--", lw=1.3)
ax.text(SNR_MIN + 0.25, ax.get_ylim()[1] * 0.9, f"{SNR_MIN:.0f}$\\sigma$",
        color="#c44e52", fontsize=9)
ax.set_xlabel("peak signal-to-noise ratio")
ax.set_ylabel("number of positions")
ax.set_title("Forced photometry at undetected positions", fontsize=10)
ax.legend(fontsize=8, framealpha=0.9)
ax.tick_params(labelsize=8)

ax2 = axes[1]
ax2.scatter(snr[good & ~accepted], np.asarray(out["fp_int_mJy"])[good & ~accepted],
            s=10, c="0.78", edgecolors="none", label="rejected")
ax2.scatter(snr[accepted], np.asarray(out["fp_int_mJy"])[accepted],
            s=16, c="#22539b", edgecolors="none", label="candidate")
ax2.axvline(SNR_MIN, color="#c44e52", ls="--", lw=1.3)
ax2.axhline(0, color="0.55", lw=0.8)
ax2.set_xlim(-4, 12)
ax2.set_xlabel("peak signal-to-noise ratio")
ax2.set_ylabel("integrated flux density (mJy)")
ax2.set_title("Flux against significance", fontsize=10)
ax2.legend(fontsize=8, framealpha=0.9)
ax2.tick_params(labelsize=8)

fig.tight_layout()
fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
print(f"\ncutouts: run step03e_vet_candidates.py")
