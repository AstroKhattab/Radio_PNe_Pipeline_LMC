"""
step3b_extract_visual.py
========================
Step 3b — Extract radio fluxes for visually confirmed PNe.

For each source:
1. Start at Reid optical position.
2. Search within 6 arcsec (4 pixels) for peak flux pixel in MeerKAT image.
3. That peak pixel = radio centroid.
4. Extract peak flux (Jy/beam → mJy/beam) at peak pixel.
5. Extract integrated flux within 8"×8" beam centred on peak:
      int_flux = sum(pixels in beam) / beam_area_pixels
6. Extract local RMS (median in 30"–90" annulus from peak).
7. SNR = peak_flux / local_rms.
8. Generate cutout: 8" beam circle at PEAK, crosshair at REID position.

Outputs
-------
    03_Outputs/step3b_meerkat_visual.vot — Catalogue with fluxes
    04_Inspect/figures/step3b_meerkat_visual_beams.pdf — Cutouts for verification

After verifying the cutouts, run step3c to merge these into the detection
catalogue.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config
from _support.plot_style import apply_paper_style

warnings.simplefilter("ignore", AstropyWarning)
apply_paper_style()

cfg = config.load()
ENTRIES     = cfg.visual_ids()
VISUAL_IDS  = [rp for rp, _ in ENTRIES]
OVERRIDES   = {rp: radius for rp, radius in ENTRIES if radius is not None}

BEAM_FILE = cfg.data("meerkat_mosaic")
RMS_FILE  = cfg.data("meerkat_rms")
HA_FILE   = cfg.data("mcels_ha")
OIII_FILE = cfg.data("mcels_oiii")
TO_CHECK  = cfg.out("step3a_meerkat_to_check.vot")

OUT_VOT   = cfg.out("step3b_meerkat_visual.vot")
OUT_PDF   = cfg.inspect("figures/step3b_meerkat_visual_beams.pdf")
os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)

SEARCH_RAD  = cfg["visual"]["search_arcsec"]
BEAM_FWHM   = cfg["meerkat"]["beam_fwhm_arcsec"]
BEAM_RAD    = cfg["visual"]["aperture_arcsec"]
RMS_ANN_IN, RMS_ANN_OUT = cfg["visual"]["rms_annulus_arcsec"]
CUTOUT_SIZE = 90.0   # arcsec — inspection cutout side, cosmetic only
FLUX_SCALE_ERR = 0.05   # fractional flux-scale term, added in quadrature

# A circular aperture of one beam half-width catches only half the flux of an
# unresolved source.  Correcting for that is the difference between the
# integrated flux density and half of it, and it propagates all the way to the
# luminosity function.
ENCLOSED = cfg.enclosed_fraction(BEAM_RAD)

print("step3b: extracting MeerKAT fluxes for the visual detections")
print(f"  Visual IDs: {len(VISUAL_IDS)}")
print(f"  Aperture r = {BEAM_RAD}\" on an {BEAM_FWHM}\" beam encloses {ENCLOSED:.4f}"
      f" of a point source; fluxes are divided by that")

if not VISUAL_IDS:
    print("  No IDs. Exiting."); exit(0)

# Load to-check catalogue
print("\n[1] Loading step3a_meerkat_to_check.vot...")
tbl   = Table.read(TO_CHECK, format="votable")
rpids = np.array(tbl["RP_ID"], dtype=str)

found_mask = np.zeros(len(tbl), dtype=bool)
not_found  = []
for vid in VISUAL_IDS:
    m = np.where(rpids == vid)[0]
    if len(m) == 0:
        not_found.append(vid)
    else:
        found_mask[m[0]] = True

if not_found:
    print(f"  inspected but not in the parent sample ({len(not_found)}): "
          f"{', '.join(not_found)}")
    print("  (HASH V/163 does not list these as true or probable PNe)")

visual = tbl[found_mask]
n_vis  = len(visual)
print(f"  Matched: {n_vis} sources")

# Load FITS images
print("\n[2] Loading FITS images...")

def load_img(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found"); return None, None, None
    # Keep the array memory-mapped and in its native dtype.  The MeerKAT
    # mosaic is 24235 x 36281 pixels; promoting it to float64 up front asks
    # for 6.5 GB and fails on anything but a very large machine.  Every use
    # below is a small cutout, which is cast to float where it is taken.
    h = fits.open(path, memmap=True)
    for hdu in h:
        if hdu.data is not None and len(hdu.data.shape) >= 2:
            d = hdu.data
            while d.ndim > 2:
                d = d[0]
            w = WCS(hdu.header, naxis=2)
            print(f"  {os.path.basename(path):40s} {d.shape}")
            return d, w, hdu.header
    return None, None, None

data_beam, wcs_beam, hdr_beam = load_img(BEAM_FILE)
data_rms,  wcs_rms,  _        = load_img(RMS_FILE)
data_ha,   wcs_ha,   _        = load_img(HA_FILE)
data_oiii, wcs_oiii, _        = load_img(OIII_FILE)

# Pixel scale
pscale = abs(float(hdr_beam.get("CDELT1", 0.000417))) * 3600  # arcsec/pix
print(f"  Pixel scale: {pscale:.3f}\"/pix")

# Beam area in pixels  (Gaussian beam: A = pi*bmaj*bmin / (4*ln2) / pixarea)
beam_area_pix = (np.pi * BEAM_FWHM * BEAM_FWHM) / (4.0 * np.log(2) * pscale**2)
aper_area_pix = np.pi * BEAM_RAD**2 / pscale**2
n_beams_aper  = aper_area_pix / beam_area_pix
print(f"  Beam area: {beam_area_pix:.1f} pixels")

# Flux extraction
print(f"\n[3] Extracting fluxes (search {SEARCH_RAD}\", beam {BEAM_FWHM}\"×{BEAM_FWHM}\")...\n")

ras_reid  = np.array(visual["RA"],  dtype=float)
decs_reid = np.array(visual["Dec"], dtype=float)

# Output arrays
col_peak     = np.full(n_vis, np.nan)   # mJy/beam
col_int      = np.full(n_vis, np.nan)   # mJy (integrated)
col_rms      = np.full(n_vis, np.nan)   # mJy/beam
col_snr      = np.full(n_vis, np.nan)
col_ra_radio = np.full(n_vis, np.nan)
col_dec_radio= np.full(n_vis, np.nan)
col_offset   = np.full(n_vis, np.nan)   # arcsec
col_err_int  = np.full(n_vis, np.nan)   # mJy
col_flag     = np.full(n_vis, "", dtype="U20")

# Store peak pixel positions for cutout plotting
peak_pix_x = np.full(n_vis, np.nan)
peak_pix_y = np.full(n_vis, np.nan)

for k in range(n_vis):
    ra_r  = ras_reid[k]
    dec_r = decs_reid[k]
    rpid  = str(visual["RP_ID"][k]).strip()

    if not (np.isfinite(ra_r) and np.isfinite(dec_r)):
        col_flag[k] = "no_position"; continue

    try:
        coord    = SkyCoord(ra=ra_r * u.deg, dec=dec_r * u.deg)
        cut_size = u.Quantity(CUTOUT_SIZE, u.arcsec)

        cut_b = Cutout2D(data_beam, coord, cut_size,
                         wcs=wcs_beam, mode="partial", fill_value=np.nan)
        cut_n = Cutout2D(data_rms,  coord, cut_size,
                         wcs=wcs_rms,  mode="partial", fill_value=np.nan)

        rd = cut_b.data
        nd = cut_n.data

        if rd is None or np.all(np.isnan(rd)):
            col_flag[k] = "no_data"; continue

        # Reid pixel in cutout
        pix_r = cut_b.wcs.all_world2pix([[ra_r, dec_r]], 0)[0]
        cx_r, cy_r = float(pix_r[0]), float(pix_r[1])

        # Distance grid from Reid position
        ny, nx = rd.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        dist_from_reid = np.sqrt((xx - cx_r)**2 + (yy - cy_r)**2) * pscale

        # Step A: Search for peak (or use Reid position if override)
        src_search_rad = OVERRIDES.get(rpid, SEARCH_RAD)
        if src_search_rad == 0.0:
            # Extract directly at Reid position
            py, px = int(round(cy_r)), int(round(cx_r))
            if py < 0 or py >= ny or px < 0 or px >= nx:
                col_flag[k] = "reid_out_of_bounds"; continue
        else:
            search = rd.copy()
            search[dist_from_reid > src_search_rad] = np.nan
            if np.all(np.isnan(search)):
                col_flag[k] = "no_data_in_search"; continue
            peak_idx = np.unravel_index(np.nanargmax(search), rd.shape)
            py, px   = peak_idx
        peak_pix_x[k] = px
        peak_pix_y[k] = py

        # Step B: Peak flux (Jy/beam → mJy/beam)
        pf_jy = float(rd[py, px])
        col_peak[k] = pf_jy * 1000.0   # mJy/beam

        # Step C: Radio centroid (world coords of peak pixel)
        world_pk = cut_b.wcs.all_pix2world([[px, py]], 0)[0]
        col_ra_radio[k]  = float(world_pk[0])
        col_dec_radio[k] = float(world_pk[1])

        # Step D: Offset Reid → peak
        c_pk = SkyCoord(ra=col_ra_radio[k] * u.deg,
                        dec=col_dec_radio[k] * u.deg)
        col_offset[k] = float(coord.separation(c_pk).to(u.arcsec).value)

        # Step E: Integrated flux within 8"×8" beam at peak
        dist_from_peak = np.sqrt((xx - px)**2 + (yy - py)**2) * pscale
        beam_mask      = dist_from_peak <= BEAM_RAD
        beam_pixels    = rd[beam_mask]
        beam_pixels    = beam_pixels[np.isfinite(beam_pixels)]

        if len(beam_pixels) == 0:
            col_flag[k] = "empty_aperture"; continue
        # Aperture sum in beams, then corrected for the flux the aperture misses.
        col_int[k] = (float(np.sum(beam_pixels)) / beam_area_pix
                      / ENCLOSED * 1000.0)  # mJy

        # Step F: Local RMS (annulus around peak)
        ann_mask = (dist_from_peak >= RMS_ANN_IN) & (dist_from_peak <= RMS_ANN_OUT)
        rms_vals = nd[ann_mask]
        rms_vals = rms_vals[np.isfinite(rms_vals)]

        if len(rms_vals) > 0:
            col_rms[k] = float(np.nanmedian(rms_vals)) * 1000.0
        else:
            col_rms[k] = float(np.nanmedian(nd[np.isfinite(nd)])) * 1000.0

        # Step G: SNR, and the uncertainty on the aperture flux.  The noise
        # scales with the number of independent beams the aperture covers and
        # is corrected alongside the flux; the flux-scale term dominates for
        # the brighter sources.
        if col_rms[k] > 0:
            col_snr[k] = col_peak[k] / col_rms[k]
            sigma_ap = col_rms[k] * np.sqrt(n_beams_aper) / ENCLOSED
            col_err_int[k] = float(np.hypot(sigma_ap,
                                            FLUX_SCALE_ERR * abs(col_int[k])))

        col_flag[k] = "extracted"

        print(f"  {rpid:<10s}  peak={col_peak[k]:8.4f}  int={col_int[k]:8.4f}  "
              f"rms={col_rms[k]:7.4f}  SNR={col_snr[k]:5.1f}  "
              f"offset={col_offset[k]:4.2f}\"  [{col_flag[k]}]")

    except Exception as e:
        col_flag[k] = "error"
        print(f"  {rpid:<10s}  ERROR: {e}")

# Build output catalogue
print(f"\n[4] Writing step3b_meerkat_visual.vot...")

visual.add_column(Column(np.full(n_vis, "visual", dtype="U10"), name="detection_method"))
visual.add_column(Column(col_ra_radio,                name="mkt_ra"))
visual.add_column(Column(col_dec_radio,               name="mkt_dec"))
visual.add_column(Column(col_peak.astype("f4"),       name="mkt_peak_flux_mJy"))
visual.add_column(Column(col_int.astype("f4"),        name="mkt_int_flux_mJy"))
visual.add_column(Column(col_err_int.astype("f4"),    name="mkt_err_int_flux_mJy"))
visual.add_column(Column(col_rms.astype("f4"),        name="mkt_local_rms_mJy"))
visual.add_column(Column(col_snr.astype("f4"),        name="mkt_snr"))
visual.add_column(Column(col_offset.astype("f4"),     name="offset_arcsec"))
visual.add_column(Column(np.full(n_vis, BEAM_RAD, dtype="f4"), name="aperture_radius_arcsec"))
visual.add_column(Column(np.full(n_vis, ENCLOSED, dtype="f4"), name="enclosed_flux_fraction"))
visual.add_column(Column(col_flag,                    name="flux_flag"))
for name, unit in (("mkt_peak_flux_mJy", "mJy/beam"), ("mkt_local_rms_mJy", "mJy/beam"),
                   ("mkt_int_flux_mJy", "mJy"), ("mkt_err_int_flux_mJy", "mJy"),
                   ("offset_arcsec", "arcsec"), ("aperture_radius_arcsec", "arcsec")):
    visual[name].unit = unit
visual["mkt_int_flux_mJy"].description = (
    "aperture sum inside one beam half-width, divided by the enclosed fraction")

visual.write(OUT_VOT, format="votable", overwrite=True)
print(f"  -> {OUT_VOT}  ({n_vis} rows)")

# Cutout PDF
print(f"\n[5] Generating cutouts with beam circles...")

DATASETS = [
    (data_beam, wcs_beam, "Radio",   "mJy/beam", True),
    (data_ha,   wcs_ha,   "H\u03b1", "Flux",     False),
    (data_oiii, wcs_oiii, "[O III]", "Flux",      False),
]

with PdfPages(OUT_PDF) as pdf:
    for k in range(n_vis):
        ra_r  = ras_reid[k]; dec_r = decs_reid[k]
        if not (np.isfinite(ra_r) and np.isfinite(dec_r)):
            continue

        rpid = str(visual["RP_ID"][k]).strip()
        cls  = str(visual["reid_class"][k]).strip()
        sname = f"{rpid}  [{cls}]  (RA={ra_r:.5f}°, Dec={dec_r:.5f}°)"
        snr_str = f"SNR={col_snr[k]:.1f}" if np.isfinite(col_snr[k]) else "SNR=N/A"
        title = f"{sname}\npeak={col_peak[k]:.4f} mJy  int={col_int[k]:.4f} mJy  {snr_str}  offset={col_offset[k]:.2f}\""

        coord    = SkyCoord(ra=ra_r * u.deg, dec=dec_r * u.deg)
        cut_size = u.Quantity(CUTOUT_SIZE, u.arcsec)

        fig = plt.figure(figsize=(15, 5))
        for pi, (data, wcs_img, label, unit, is_radio) in enumerate(DATASETS):
            try:
                cut = Cutout2D(data, coord, cut_size, wcs=wcs_img,
                               mode="partial", fill_value=np.nan)
                cd = cut.data.astype(float)
                if is_radio:
                    cd = cd * 1000.0  # Jy → mJy
                cw = cut.wcs
            except Exception:
                cd = np.full((60, 60), np.nan); cw = wcs_img

            ax = fig.add_subplot(1, 3, pi + 1, projection=cw)

            vmin = 0.0
            vmax = float(np.nanpercentile(cd, 99.99)) if np.any(np.isfinite(cd)) else 1.0
            if vmax <= vmin: vmax = vmin + 1e-9

            im = ax.imshow(cd, origin="lower", cmap="magma",
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit, fontsize=8)
            cb.ax.tick_params(labelsize=7)

            ax.set_title(label, fontsize=10)
            ax.set_xlabel("RA", fontsize=8)
            ax.set_ylabel("Dec", fontsize=8)
            ax.grid(True, alpha=0.2, color="white", ls=":", lw=0.5)
            if hasattr(ax, "coords"):
                ax.coords[0].set_major_formatter("hh:mm:ss")
                ax.coords[1].set_major_formatter("dd:mm:ss")
                ax.tick_params(labelsize=6)

            try:
                pix_tr = ax.get_transform("pixel")

                # Reid position crosshair (cyan, thin)
                pix_reid = cw.all_world2pix([[ra_r, dec_r]], 0)[0]
                rx, ry = float(pix_reid[0]), float(pix_reid[1])
                ch = 8.0 / pscale  # crosshair arm length
                ax.plot([rx - ch, rx + ch], [ry, ry],
                        color="cyan", lw=0.8, transform=pix_tr, zorder=5)
                ax.plot([rx, rx], [ry - ch, ry + ch],
                        color="cyan", lw=0.8, transform=pix_tr, zorder=5)

                # Peak position: 8" beam circle (green, thick)
                if np.isfinite(col_ra_radio[k]):
                    pix_pk = cw.all_world2pix([[col_ra_radio[k], col_dec_radio[k]]], 0)[0]
                    pkx, pky = float(pix_pk[0]), float(pix_pk[1])
                    beam_r_pix = BEAM_RAD / pscale
                    circ = mpatches.Circle(
                        (pkx, pky), radius=beam_r_pix,
                        edgecolor="lime", facecolor="none",
                        linewidth=1.8, transform=pix_tr, zorder=6)
                    ax.add_patch(circ)
            except Exception:
                pass

        fig.suptitle(title, fontsize=9, fontweight="bold", y=1.03)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, dpi=100, bbox_inches="tight")
        plt.close(fig)

        if (k + 1) % 10 == 0:
            print(f"    {k+1}/{n_vis} done...")

print(f"  -> {OUT_PDF}")

# Summary
n_ok = int(np.sum(col_flag == "extracted"))
print("\nstep3b complete")
print(f"  Visual sources: {n_vis}")
print(f"  Flux extracted: {n_ok}")
print(f"  Failed:         {n_vis - n_ok}")
if n_ok > 0:
    ok = col_flag == "extracted"
    print(f"\n  Peak flux  — median: {np.nanmedian(col_peak[ok]):.4f} mJy/beam")
    print(f"  Int  flux  — median: {np.nanmedian(col_int[ok]):.4f} mJy")
    print(f"  SNR        — median: {np.nanmedian(col_snr[ok]):.1f}")
    print(f"  Offset     — median: {np.nanmedian(col_offset[ok]):.2f}\"")
print(f"\n  Catalogue: {OUT_VOT}")
print(f"  Cutouts:   {OUT_PDF}")
print("\n  Verify the cutouts, then run step3c.")
