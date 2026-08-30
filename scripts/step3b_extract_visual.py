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

After verifying cutouts, run
step3c_build_meerkat_catalogue_and_multisurvey_figures.py to merge.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _support.plot_style import apply_paper_style
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
import astropy.units as u

warnings.filterwarnings("ignore")
apply_paper_style()

# Visual IDs
VISUAL_IDS = [
    "RP607",  "RP650",  "RP652",  "RP656",
    "RP980",  "RP1114", "RP1234",
    "RP1324", "RP1352", "RP1557", "RP1606", "RP1608",
    "RP1634", "RP1636", "RP1684", "RP1687", "RP1695",
    "RP2278", "RP2294", "RP2297", "RP2304",
    "RP2311", "RP2312", "RP2326", "RP2708", "RP3449",
    "RP3464", "RP3661", "RP4065", "RP4081",
    "RP4176", "RP4285",
]

# Per-source search radius overrides (arcsec)
# Use 0 to extract directly at Reid position (no peak search)
OVERRIDES = {
    "RP1114": 0.0,   # bright extended source nearby — extract at Reid pos
    "RP2708": 0.0,   # wrong source grabbed — extract at Reid pos
}

# Paths
BASE      = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA      = os.path.join(BASE, "01_Data")
OUTS      = os.path.join(BASE, "03_Outputs")
INSP      = os.path.join(BASE, "04_Inspect", "figures")

BEAM_FILE = os.path.join(DATA, "LMC_I_mosaic_ch0_beam.fits")
RMS_FILE  = os.path.join(DATA, "LMC_I_mosaic_ch0_rms.fits")
HA_FILE   = os.path.join(DATA, "LMC.ha.fits")
OIII_FILE = os.path.join(DATA, "LMC.oiii.fits")
TO_CHECK  = os.path.join(OUTS, "step3a_meerkat_to_check.vot")

OUT_VOT   = os.path.join(OUTS, "step3b_meerkat_visual.vot")
OUT_PDF   = os.path.join(INSP, "step3b_meerkat_visual_beams.pdf")

SEARCH_RAD  = 6.0    # arcsec — peak search radius (4 pixels @ 1.5"/pix)
BEAM_FWHM   = 8.0    # arcsec — MeerKAT beam FWHM
BEAM_RAD    = 4.0    # arcsec — half-beam
RMS_ANN_IN  = 30.0   # arcsec
RMS_ANN_OUT = 90.0   # arcsec
CUTOUT_SIZE = 90.0   # arcsec — full cutout side

print("step3b: extracting MeerKAT fluxes for the visual detections")
print(f"  Visual IDs: {len(VISUAL_IDS)}")

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
    print(f"  WARNING: not found: {not_found}")

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

        if len(beam_pixels) > 0:
            col_int[k] = float(np.nansum(beam_pixels)) / beam_area_pix * 1000.0  # mJy
        else:
            col_int[k] = col_peak[k]  # fallback: point source assumption

        # Step F: Local RMS (annulus around peak)
        ann_mask = (dist_from_peak >= RMS_ANN_IN) & (dist_from_peak <= RMS_ANN_OUT)
        rms_vals = nd[ann_mask]
        rms_vals = rms_vals[np.isfinite(rms_vals)]

        if len(rms_vals) > 0:
            col_rms[k] = float(np.nanmedian(rms_vals)) * 1000.0
        else:
            col_rms[k] = float(np.nanmedian(nd[np.isfinite(nd)])) * 1000.0

        # Step G: SNR
        if col_rms[k] > 0:
            col_snr[k] = col_peak[k] / col_rms[k]

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
visual.add_column(Column(col_rms.astype("f4"),        name="mkt_local_rms_mJy"))
visual.add_column(Column(col_snr.astype("f4"),        name="mkt_snr"))
visual.add_column(Column(col_offset.astype("f4"),     name="offset_arcsec"))
visual.add_column(Column(col_flag,                    name="flux_flag"))

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
print("\n  Verify cutouts then run: python "
      "02_Scripts/step3c_build_meerkat_catalogue_and_multisurvey_figures.py")
