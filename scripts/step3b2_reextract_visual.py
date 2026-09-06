"""
step3b2_reextract_visual.py

Independent check on the step 3b photometry, with a wide aperture.

Step 3b measures the sub-threshold sources in an aperture of one beam
half-width and divides by the fraction of a point source that aperture
encloses, which is the method the paper describes.  This script repeats the
measurement in a 2 x FWHM aperture with a local background subtracted, where
no enclosed-flux correction is needed at all, and reports both so the
correction can be checked against a measurement that does not use it.

Nothing downstream reads this file.  The wide aperture is noise dominated at
these flux levels -- most of these sources are detected between 2 and 6 sigma
-- which is why the narrow, corrected aperture is the one the analysis uses.

Inputs
------
    01_Data/LMC_I_mosaic_ch0_beam.fits    beam-corrected mosaic (Jy/beam)
    01_Data/LMC_I_mosaic_ch0_rms.fits     RMS map (Jy/beam)
    03_Outputs/step3a_meerkat_to_check.vot

Outputs
-------
    03_Outputs/step3b2_meerkat_visual_reextracted.vot
    05_Figures/step3b2_visual_photometry_cutouts/<RP_ID>.png

step3b_extract_meerkat_visual.py is left untouched.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style
import matplotlib.patches as mpatches
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, PercentileInterval, ImageNormalize
import astropy.units as u

warnings.filterwarnings("ignore")
apply_paper_style()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()
ENTRIES = cfg.visual_ids()
VISUAL_IDS = [rp for rp, _ in ENTRIES]
OVERRIDES = {rp: radius for rp, radius in ENTRIES if radius is not None}

PNGD = cfg.inspect("figures/step3b2_visual_photometry_cutouts")

BEAM_FILE = cfg.data("meerkat_mosaic")
RMS_FILE  = cfg.data("meerkat_rms")
TO_CHECK  = cfg.out("step3a_meerkat_to_check.vot")
OUT_VOT   = cfg.out("step3b2_meerkat_visual_reextracted.vot")

SEARCH_RAD   = cfg["visual"]["search_arcsec"]
BEAM_FWHM    = cfg["meerkat"]["beam_fwhm_arcsec"]
APER_RAD     = 2.0 * BEAM_FWHM   # the wide check aperture; 2 x FWHM
BKG_ANN_IN   = 24.0   # arcsec, starts clear of the wide aperture
BKG_ANN_OUT  = 48.0   # arcsec, wide enough for a stable median, still local
NARROW_RAD   = cfg["visual"]["aperture_arcsec"]   # the aperture step 3b uses
FLUX_SCALE_ERR = 0.05 # fractional flux-scale term added in quadrature
CUTOUT_SIZE  = 90.0   # arcsec, side of the inspection PNG
PHOT_BOX     = 2.4 * BKG_ANN_OUT  # arcsec, must comfortably contain the annulus

print("step3b2: wide-aperture check on the step 3b photometry")
print("Visual IDs: %d" % len(VISUAL_IDS))

os.makedirs(PNGD, exist_ok=True)

print("Loading step3a_meerkat_to_check.vot ...")
tbl   = Table.read(TO_CHECK, format="votable")
rpids = np.array(tbl["RP_ID"], dtype=str)

rows, not_found = [], []
for vid in VISUAL_IDS:
    m = np.where(rpids == vid)[0]
    if len(m) == 0:
        not_found.append(vid)
    else:
        rows.append(int(m[0]))

if not_found:
    print("WARNING: not found in step03a: %s" % not_found)

visual = tbl[rows]
n_vis  = len(visual)
print("Matched: %d sources" % n_vis)

# memmap keeps the 3.5 GB mosaic on disk; only the cutouts are ever read.
print("Opening mosaic and RMS map (memmap) ...")
hdul_beam = fits.open(BEAM_FILE, memmap=True)
hdul_rms  = fits.open(RMS_FILE,  memmap=True)
data_beam = np.squeeze(hdul_beam[0].data)
data_rms  = np.squeeze(hdul_rms[0].data)
wcs_beam  = WCS(hdul_beam[0].header, naxis=2)
wcs_rms   = WCS(hdul_rms[0].header,  naxis=2)

pscale = float(np.mean(proj_plane_pixel_scales(wcs_beam))) * 3600.0
beam_area_pix = (np.pi * BEAM_FWHM * BEAM_FWHM) / (4.0 * np.log(2) * pscale**2)
aper_area_pix = np.pi * APER_RAD**2 / pscale**2
n_beams_aper  = aper_area_pix / beam_area_pix

# Fraction of a circular Gaussian of this FWHM falling inside a radius r:
# 1 - exp(-4 ln2 r^2 / FWHM^2).  Quoted rather than assumed.
def enclosed_fraction(radius):
    return 1.0 - np.exp(-4.0 * np.log(2) * radius**2 / BEAM_FWHM**2)

f_new = enclosed_fraction(APER_RAD)
f_narrow = enclosed_fraction(NARROW_RAD)

print("Pixel scale     : %.4f arcsec/pix" % pscale)
print("Beam area       : %.2f pix" % beam_area_pix)
print("Aperture r=%.1f\" : %.2f pix, %.2f beams, encloses %.4f of a point source"
      % (APER_RAD, aper_area_pix, n_beams_aper, f_new))
print("Narrow r=%.1f\"   : encloses %.4f, so step 3b divides its aperture sum by that"
      % (NARROW_RAD, f_narrow))

ras_opt  = np.array(visual["RA"],  dtype=float)
decs_opt = np.array(visual["Dec"], dtype=float)

col_peak     = np.full(n_vis, np.nan)
col_int_new  = np.full(n_vis, np.nan)
col_err_new  = np.full(n_vis, np.nan)
col_int_narrow  = np.full(n_vis, np.nan)
col_int_narrow_c = np.full(n_vis, np.nan)
col_bkg      = np.full(n_vis, np.nan)
col_rms      = np.full(n_vis, np.nan)
col_snr      = np.full(n_vis, np.nan)
col_ra_radio = np.full(n_vis, np.nan)
col_dec_radio = np.full(n_vis, np.nan)
col_offset   = np.full(n_vis, np.nan)
col_flag     = np.full(n_vis, "", dtype="U24")

print("\nExtracting (aperture r=%.0f\", background %.0f-%.0f\") ...\n"
      % (APER_RAD, BKG_ANN_IN, BKG_ANN_OUT))

for k in range(n_vis):
    rpid  = str(visual["RP_ID"][k]).strip()
    ra_o  = ras_opt[k]
    dec_o = decs_opt[k]

    if not (np.isfinite(ra_o) and np.isfinite(dec_o)):
        col_flag[k] = "no_position"
        print("  %-10s no optical position" % rpid)
        continue

    try:
        coord = SkyCoord(ra=ra_o * u.deg, dec=dec_o * u.deg)
        box   = u.Quantity(PHOT_BOX, u.arcsec)

        cut_b = Cutout2D(data_beam, coord, box, wcs=wcs_beam,
                         mode="partial", fill_value=np.nan)
        cut_n = Cutout2D(data_rms, coord, box, wcs=wcs_rms,
                         mode="partial", fill_value=np.nan)
        rd = np.asarray(cut_b.data, dtype=float)
        nd = np.asarray(cut_n.data, dtype=float)

        if not np.any(np.isfinite(rd)):
            col_flag[k] = "no_data"
            print("  %-10s no valid pixels" % rpid)
            continue

        pix_o = cut_b.wcs.all_world2pix([[ra_o, dec_o]], 0)[0]
        cx_o, cy_o = float(pix_o[0]), float(pix_o[1])

        ny, nx = rd.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        dist_opt = np.hypot(xx - cx_o, yy - cy_o) * pscale

        search_rad = OVERRIDES.get(rpid, SEARCH_RAD)
        if search_rad == 0.0:
            py, px = int(round(cy_o)), int(round(cx_o))
            if py < 0 or py >= ny or px < 0 or px >= nx:
                col_flag[k] = "optical_out_of_bounds"
                print("  %-10s optical position off the mosaic" % rpid)
                continue
        else:
            search = np.where(dist_opt <= search_rad, rd, np.nan)
            if not np.any(np.isfinite(search)):
                col_flag[k] = "no_data_in_search"
                print("  %-10s no valid pixels in search radius" % rpid)
                continue
            py, px = np.unravel_index(np.nanargmax(search), rd.shape)

        col_peak[k] = float(rd[py, px]) * 1000.0

        world_pk = cut_b.wcs.all_pix2world([[px, py]], 0)[0]
        col_ra_radio[k]  = float(world_pk[0])
        col_dec_radio[k] = float(world_pk[1])
        c_pk = SkyCoord(ra=col_ra_radio[k] * u.deg, dec=col_dec_radio[k] * u.deg)
        col_offset[k] = float(coord.separation(c_pk).to(u.arcsec).value)

        dist_pk  = np.hypot(xx - px, yy - py) * pscale
        aper     = (dist_pk <= APER_RAD) & np.isfinite(rd)
        ann      = (dist_pk >= BKG_ANN_IN) & (dist_pk <= BKG_ANN_OUT) & np.isfinite(rd)

        if not np.any(aper):
            col_flag[k] = "empty_aperture"
            print("  %-10s empty aperture" % rpid)
            continue

        if np.count_nonzero(ann) >= 50:
            _, bkg_med, _ = sigma_clipped_stats(rd[ann], sigma=3.0, maxiters=5)
        else:
            bkg_med = 0.0
            col_flag[k] = "no_bkg_annulus"
        col_bkg[k] = float(bkg_med) * 1000.0

        n_aper = int(np.count_nonzero(aper))
        aper_sum = float(np.sum(rd[aper])) - float(bkg_med) * n_aper
        col_int_new[k] = aper_sum / beam_area_pix * 1000.0

        # The step 3b measurement, reproduced here for comparison: sum inside
        # r = FWHM/2, no background, divided by the beam area and then by the
        # enclosed fraction.
        narrow_mask = (dist_pk <= NARROW_RAD) & np.isfinite(rd)
        if np.any(narrow_mask):
            col_int_narrow[k] = float(np.sum(rd[narrow_mask])) / beam_area_pix * 1000.0
            col_int_narrow_c[k] = col_int_narrow[k] / f_narrow

        ann_rms = nd[(dist_pk >= BKG_ANN_IN) & (dist_pk <= BKG_ANN_OUT) & np.isfinite(nd)]
        if ann_rms.size > 0:
            col_rms[k] = float(np.median(ann_rms)) * 1000.0
        elif np.any(np.isfinite(nd)):
            col_rms[k] = float(np.nanmedian(nd)) * 1000.0

        if np.isfinite(col_rms[k]):
            # Aperture noise scales with the number of independent beams it
            # covers; the flux-scale term dominates for the brighter sources.
            sigma_ap = col_rms[k] * np.sqrt(n_beams_aper)
            col_err_new[k] = float(np.hypot(sigma_ap,
                                            FLUX_SCALE_ERR * abs(col_int_new[k])))
            if col_rms[k] > 0:
                col_snr[k] = col_peak[k] / col_rms[k]

        if col_flag[k] == "":
            col_flag[k] = "extracted"

        ratio = col_int_new[k] / col_int_narrow[k] if col_int_narrow[k] else np.nan
        print("  %-10s peak=%8.4f  wide=%8.4f +/- %6.4f  narrow=%8.4f  wide/narrow=%5.2f  "
              "bkg=%+8.5f  SNR=%6.1f  off=%4.2f\"  [%s]"
              % (rpid, col_peak[k], col_int_new[k], col_err_new[k], col_int_narrow[k],
                 ratio, col_bkg[k], col_snr[k], col_offset[k], col_flag[k]))

        # Inspection cutout.
        cut_p = Cutout2D(data_beam, coord, u.Quantity(CUTOUT_SIZE, u.arcsec),
                         wcs=wcs_beam, mode="partial", fill_value=np.nan)
        pd_mjy = np.asarray(cut_p.data, dtype=float) * 1000.0
        pw = cut_p.wcs

        fig = plt.figure(figsize=(5.4, 5.4))
        ax  = fig.add_subplot(1, 1, 1, projection=pw)
        norm = ImageNormalize(pd_mjy, interval=PercentileInterval(99.5),
                              stretch=AsinhStretch(0.1))
        im = ax.imshow(pd_mjy, origin="lower", cmap="magma", norm=norm,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mJy/beam", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        tr = ax.get_transform("pixel")
        po = pw.all_world2pix([[ra_o, dec_o]], 0)[0]
        ox, oy = float(po[0]), float(po[1])
        arm = 6.0 / pscale
        ax.plot([ox - arm, ox + arm], [oy, oy], color="cyan", lw=0.9,
                transform=tr, zorder=5)
        ax.plot([ox, ox], [oy - arm, oy + arm], color="cyan", lw=0.9,
                transform=tr, zorder=5)

        pp = pw.all_world2pix([[col_ra_radio[k], col_dec_radio[k]]], 0)[0]
        pkx, pky = float(pp[0]), float(pp[1])
        ax.plot([pkx], [pky], marker="x", color="white", ms=6, mew=1.2,
                transform=tr, zorder=6)
        for rad, colr, style in ((APER_RAD, "lime", "-"),
                                 (BKG_ANN_IN, "deepskyblue", "--"),
                                 (BKG_ANN_OUT, "deepskyblue", "--")):
            ax.add_patch(mpatches.Circle((pkx, pky), radius=rad / pscale,
                                         edgecolor=colr, facecolor="none",
                                         linewidth=1.2, linestyle=style,
                                         transform=tr, zorder=6))

        ax.set_xlabel("RA (J2000)", fontsize=9)
        ax.set_ylabel("Dec (J2000)", fontsize=9)
        ax.coords[0].set_major_formatter("hh:mm:ss")
        ax.coords[1].set_major_formatter("dd:mm:ss")
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, color="white", ls=":", lw=0.5)
        ax.set_title("%s\n$S_\\mathrm{int}$ = %.3f $\\pm$ %.3f mJy   "
                     "$S_\\mathrm{peak}$ = %.3f mJy/beam\n"
                     "S/N = %.1f   offset = %.2f\"" %
                     (rpid, col_int_new[k], col_err_new[k], col_peak[k],
                      col_snr[k], col_offset[k]), fontsize=9)
        fig.savefig(os.path.join(PNGD, "%s.png" % rpid), dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        col_flag[k] = "error"
        print("  %-10s ERROR: %s" % (rpid, e))

print("\nWriting %s ..." % os.path.basename(OUT_VOT))

keep = [c for c in ("RA", "Dec", "RP_ID", "Name", "reid_class") if c in visual.colnames]
out = visual[keep].copy()
out.add_column(Column(np.full(n_vis, "visual", dtype="U10"), name="detection_method"))
out.add_column(Column(col_ra_radio,  name="mkt_ra"))
out.add_column(Column(col_dec_radio, name="mkt_dec"))
out.add_column(Column(col_offset.astype("f4"),     name="offset_arcsec"))
out.add_column(Column(col_peak.astype("f4"),       name="mkt_peak_flux_mJy"))
out.add_column(Column(col_rms.astype("f4"),        name="mkt_local_rms_mJy"))
out.add_column(Column(col_snr.astype("f4"),        name="mkt_snr"))
out.add_column(Column(col_int_new.astype("f4"),    name="mkt_int_flux_mJy_new"))
out.add_column(Column(col_err_new.astype("f4"),    name="mkt_err_int_flux_mJy_new"))
out.add_column(Column(col_int_narrow.astype("f4"),    name="mkt_int_flux_mJy_narrow_raw"))
out.add_column(Column(col_int_narrow_c.astype("f4"),   name="mkt_int_flux_mJy_narrow_corrected"))
out.add_column(Column(col_bkg.astype("f4"),        name="bkg_median_mJy"))
out.add_column(Column(np.full(n_vis, APER_RAD, dtype="f4"), name="aperture_radius_arcsec"))
out.add_column(Column(np.full(n_vis, n_beams_aper, dtype="f4"), name="n_beams_in_aperture"))
out.add_column(Column(np.full(n_vis, f_new, dtype="f4"), name="enclosed_flux_fraction"))
out.add_column(Column(col_flag, name="flux_flag"))

for name, unit in (("mkt_peak_flux_mJy", "mJy/beam"),
                   ("mkt_local_rms_mJy", "mJy/beam"),
                   ("bkg_median_mJy", "mJy/beam"),
                   ("mkt_int_flux_mJy_new", "mJy"),
                   ("mkt_err_int_flux_mJy_new", "mJy"),
                   ("mkt_int_flux_mJy_narrow_raw", "mJy"),
                   ("mkt_int_flux_mJy_narrow_corrected", "mJy"),
                   ("offset_arcsec", "arcsec"),
                   ("aperture_radius_arcsec", "arcsec")):
    out[name].unit = unit

out["bkg_median_mJy"].description = "sigma-clipped median pixel value in the background annulus"
out["mkt_int_flux_mJy_narrow_raw"].description = "raw narrow-aperture sum, no background, no enclosed-flux correction"
out["mkt_int_flux_mJy_narrow_corrected"].description = "the step 3b value: narrow aperture divided by its enclosed fraction"

out.write(OUT_VOT, format="votable", overwrite=True)
print("  -> %s  (%d rows)" % (OUT_VOT, n_vis))

ok = col_flag == "extracted"
ratios = np.where((col_int_narrow != 0) & ok, col_int_new / col_int_narrow, np.nan)
print("\nExtracted: %d / %d" % (int(np.sum(ok)), n_vis))
if np.any(np.isfinite(ratios)):
    print("wide/narrow ratio  median %.3f   min %.3f   max %.3f"
          % (np.nanmedian(ratios), np.nanmin(ratios), np.nanmax(ratios)))
    print("sources with ratio > 3 or < 1/3: %d"
          % int(np.sum((ratios > 3.0) | (ratios < 1.0 / 3.0))))
print("Median new integrated flux: %.4f mJy" % np.nanmedian(col_int_new[ok]))
print("Cutouts: %s" % PNGD)
