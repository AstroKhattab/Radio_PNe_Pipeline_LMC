"""
step03a_inspect_meerkat.py
==========================
Step 3a — Split and inspect unmatched Reid x MeerKAT sources.

Logic
-----
    679 total Reid sources
    - 188 cross-matched in Step 02a → step02a_meerkat_matched.vot  (DONE)
    - 491 unmatched
        - 23 outside MeerKAT region → step03a_meerkat_outside.vot
        - 468 inside region, not detected → step03a_meerkat_to_check.vot + inspection PDF
              Omar checks these manually, provides IDs of visual detections.

Outputs
-------
    03_Outputs/step03a_meerkat_outside.vot — 23 sources outside MeerKAT region
    03_Outputs/step03a_meerkat_to_check.vot — 468 sources inside region to inspect
    04_Figures/Inspect/step03a_meerkat_matched_cutouts.pdf — 188 pages
    04_Figures/Inspect/step03a_meerkat_to_check_cutouts.pdf — 468 pages

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
import astropy.units as u
from datetime import datetime

warnings.filterwarnings("ignore")

# Paths
BASE  = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA  = os.path.join(BASE, "01_Data")
OUTS  = os.path.join(BASE, "03_Outputs")
INSP  = os.path.join(BASE, "04_Figures", "Inspect")
os.makedirs(INSP, exist_ok=True)

MKT3_FILE  = os.path.join(DATA, "MeerKAT_3sigma.vot")
HULL_STEP  = 50

# Image configuration
CUTOUT_SIZE       = 60      # arcsec
SCALE_PERCENTILE  = 99.99
MARKER_COLOR      = "cyan"
MARKER_SIZE       = 3       # pixels (radius)
DPI               = 100
PANEL_SIZE        = 5       # inches per panel

IMAGES = [
    {"name": "Radio",   "file": os.path.join(DATA, "LMC_I_mosaic_ch0_beam.fits"),
     "is_radio": True,  "colormap": "magma", "enabled": True},
    {"name": "H-alpha", "file": os.path.join(DATA, "LMC.ha.fits"),
     "is_radio": False, "colormap": "magma", "enabled": True},
    {"name": "[O III]", "file": os.path.join(DATA, "LMC.oiii.fits"),
     "is_radio": False, "colormap": "magma", "enabled": True},
]

print("step03a: building the MeerKAT inspection pages")
print(f"started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load catalogues
print("\n[1] Loading catalogues...")
matched   = Table.read(os.path.join(OUTS, "step02a_meerkat_matched.vot"), format="votable")
unmatched = Table.read(os.path.join(OUTS, "step02a_meerkat_unmatched.vot"), format="votable")
mkt3      = Table.read(MKT3_FILE, format="votable")

print(f"    Matched (Step 2a)  : {len(matched)}")
print(f"    Unmatched (Step 2a): {len(unmatched)}")

# Build MeerKAT footprint
print("\n[2] Building MeerKAT footprint...")
ra_mkt3  = np.array(mkt3["ra"],  dtype=float)
dec_mkt3 = np.array(mkt3["dec"], dtype=float)
pts      = np.column_stack([ra_mkt3[::HULL_STEP], dec_mkt3[::HULL_STEP]])
hull     = ConvexHull(pts)
hull_ra  = np.append(pts[hull.vertices, 0], pts[hull.vertices[0], 0])
hull_dec = np.append(pts[hull.vertices, 1], pts[hull.vertices[0], 1])
hull_path = MplPath(np.column_stack([hull_ra[:-1], hull_dec[:-1]]))

ra_um  = np.array(unmatched["RA"],  dtype=float)
dec_um = np.array(unmatched["Dec"], dtype=float)
valid  = np.isfinite(ra_um) & np.isfinite(dec_um)

inside = np.zeros(len(unmatched), dtype=bool)
inside[valid] = hull_path.contains_points(
    np.column_stack([ra_um[valid], dec_um[valid]])
)
outside  = ~inside & valid

n_outside  = int(outside.sum())
n_to_check = int(inside.sum())
print(f"    Inside  (to check): {n_to_check}")
print(f"    Outside (skip)    : {n_outside}")

# Write catalogues
print("\n[3] Writing catalogues...")
tbl_outside  = unmatched[outside]
tbl_to_check = unmatched[inside]

tbl_outside.add_column( Column(np.full(len(tbl_outside),  "outside",  dtype="U10"), name="step2_class"))
tbl_to_check.add_column(Column(np.full(len(tbl_to_check), "to_check", dtype="U10"), name="step2_class"))

tbl_outside.write(os.path.join(OUTS, "step03a_meerkat_outside.vot"), format="votable", overwrite=True)
tbl_to_check.write(os.path.join(OUTS, "step03a_meerkat_to_check.vot"), format="votable", overwrite=True)
print(f"    step03a_meerkat_outside.vot -> {n_outside} rows (excluded — outside region)")
print(f"    step03a_meerkat_to_check.vot -> {n_to_check} rows  (inspect manually)")

# Load FITS images once
print("\n[4] Loading FITS images...")

def load_fits(filepath):
    if not os.path.exists(filepath):
        print(f"    WARNING: not found — {filepath}")
        return None, None, None
    try:
        with fits.open(filepath) as hdul:
            for hdu in hdul:
                if hdu.data is not None and len(hdu.data.shape) >= 2:
                    data   = np.squeeze(hdu.data).astype(float)
                    header = hdu.header
                    wcs    = WCS(header, naxis=2)
                    print(f"    Loaded {os.path.basename(filepath):40s} shape={data.shape}")
                    return data, wcs, header
    except Exception as e:
        print(f"    ERROR loading {filepath}: {e}")
    return None, None, None

def convert_to_mjy(data, header):
    if header and "BUNIT" in header:
        bunit = header["BUNIT"].lower()
        if "jy/beam" in bunit and "m" not in bunit:
            return data * 1000.0   # Jy/beam → mJy/beam
    return data

loaded = {}
for img in IMAGES:
    if img["enabled"]:
        d, w, h = load_fits(img["file"])
        if img["is_radio"] and d is not None:
            d = convert_to_mjy(d, h)
        loaded[img["name"]] = {"data": d, "wcs": w, "header": h,
                               "is_radio": img["is_radio"],
                               "colormap": img["colormap"]}

# Cutout function
def extract_cutout(name, ra_deg, dec_deg):
    info = loaded.get(name, {})
    data = info.get("data"); wcs = info.get("wcs")
    if data is None or wcs is None:
        return None, None
    try:
        pos  = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        size = (CUTOUT_SIZE * u.arcsec, CUTOUT_SIZE * u.arcsec)
        cut  = Cutout2D(data, pos, size, wcs=wcs, mode="partial")
        return cut.data, cut.wcs
    except Exception:
        return None, None

def add_markers(ax, shape):
    """Circle + crosshair at cutout centre (pixel-based — source always centred)."""
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    circ = plt.Circle((cx, cy), MARKER_SIZE, fill=False,
                       edgecolor=MARKER_COLOR, linewidth=1.5, zorder=5)
    ax.add_patch(circ)
    ax.plot([cx - 6, cx + 6], [cy, cy], color=MARKER_COLOR, lw=0.9, alpha=0.8, zorder=5)
    ax.plot([cx, cx], [cy - 6, cy + 6], color=MARKER_COLOR, lw=0.9, alpha=0.8, zorder=5)

def make_page(fig, ra_deg, dec_deg, source_name):
    """One page per source: radio, H-alpha, [O III]."""
    enabled = [img for img in IMAGES if img["enabled"]]
    n       = len(enabled)
    for idx, img in enumerate(enabled):
        cut_data, cut_wcs = extract_cutout(img["name"], ra_deg, dec_deg)

        if cut_wcs is not None:
            ax = fig.add_subplot(1, n, idx + 1, projection=cut_wcs)
        else:
            ax = fig.add_subplot(1, n, idx + 1)

        if cut_data is not None:
            vmin = 0.0
            vmax = float(np.nanpercentile(cut_data, SCALE_PERCENTILE))
            if vmax <= vmin:
                vmax = vmin + 1e-9
            im = ax.imshow(cut_data, origin="lower", cmap=img["colormap"],
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("mJy/beam" if img["is_radio"] else "Flux", fontsize=8)
            cb.ax.tick_params(labelsize=7)
            add_markers(ax, cut_data.shape)
        else:
            ax.text(0.5, 0.5, f"Outside\n{img['name']} field",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")

        ax.set_title(img["name"], fontsize=10)
        ax.set_xlabel("RA",  fontsize=8)
        ax.set_ylabel("Dec", fontsize=8)
        ax.grid(True, alpha=0.3, color="white", linestyle=":", linewidth=0.5)

        if hasattr(ax, "coords"):
            ax.coords[0].set_major_formatter("hh:mm:ss")
            ax.coords[1].set_major_formatter("dd:mm:ss")
            ax.coords[0].set_separator(":")
            ax.tick_params(labelsize=6)
        else:
            ax.tick_params(labelsize=6)

    fig.suptitle(f"{source_name}  (RA={ra_deg:.5f}°, Dec={dec_deg:.5f}°)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()

def create_j_name(ra, dec):
    coord  = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    ra_str = coord.ra.to_string(unit=u.hour, sep="", precision=1, pad=True)[:6]
    if coord.dec.deg >= 0:
        dec_str = "+" + coord.dec.to_string(unit=u.deg, sep="", precision=0, pad=True)[:5]
    else:
        dec_str = coord.dec.to_string(unit=u.deg, sep="", precision=0, pad=True)[:6]
    return f"J{ra_str}{dec_str}"

def write_pdf(out_path, tbl, ra_col, dec_col, name_col, label):
    ras  = np.array(tbl[ra_col],  dtype=float)
    decs = np.array(tbl[dec_col], dtype=float)
    valid_rows = np.isfinite(ras) & np.isfinite(decs)
    n = int(valid_rows.sum())
    print(f"\n    {label}: {n} pages -> {os.path.basename(out_path)}")
    n_panels = sum(1 for img in IMAGES if img["enabled"])
    figsize  = (PANEL_SIZE * n_panels, PANEL_SIZE)
    with PdfPages(out_path) as pdf:
        for k in range(len(tbl)):
            if not valid_rows[k]:
                continue
            ra_s  = ras[k]; dec_s = decs[k]
            try:
                rpid = str(tbl["RP_ID"][k]).strip()
            except Exception:
                rpid = ""
            try:
                cls = str(tbl["reid_class"][k]).strip()
            except Exception:
                cls = ""
            jname = create_j_name(ra_s, dec_s)
            sname = f"{rpid}  [{cls}]  {jname}" if rpid else f"{jname}  [{cls}]"
            fig = plt.figure(figsize=figsize)
            make_page(fig, ra_s, dec_s, sname)
            pdf.savefig(fig, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            if (k + 1) % 50 == 0:
                print(f"      {k+1}/{n} done...")
    print(f"      -> {out_path}")

# Generate cutout PDFs
print("\n[5] Generating cutout PDFs...")

# Detected: centre on RADIO position
write_pdf(
    out_path  = os.path.join(INSP, "step03a_meerkat_matched_cutouts.pdf"),
    tbl       = matched,
    ra_col    = "mkt_ra",    # radio centroid
    dec_col   = "mkt_dec",
    name_col  = "RP_ID",
    label     = "Detected (cross-matched)",
)

# To-check: centre on OPTICAL (Reid) position
write_pdf(
    out_path  = os.path.join(INSP, "step03a_meerkat_to_check_cutouts.pdf"),
    tbl       = tbl_to_check,
    ra_col    = "RA",
    dec_col   = "Dec",
    name_col  = "RP_ID",
    label     = "To check (inside region, not cross-matched)",
)

# Summary
print("\nstep03a complete")
print(f"  Total Reid PNe          : 679")
print(f"  Cross-matched (radio)   : {len(matched)}")
print(f"  Outside MeerKAT region  : {n_outside}  (skipped)")
print(f"  To inspect manually     : {n_to_check}")
print(f"\n  Catalogues:")
print(f"    step03a_meerkat_outside.vot  ({n_outside} sources)")
print(f"    step03a_meerkat_to_check.vot ({n_to_check} sources)")
print(f"\n  Inspection PDFs (04_Figures/Inspect/):")
print(f"    step03a_meerkat_matched_cutouts.pdf ({len(matched)} pages)")
print(f"    step03a_meerkat_to_check_cutouts.pdf ({n_to_check} pages)")
print("\n  List the RP_IDs of the visual detections in step03b_extract_meerkat_visual.py")
print("  and run it to add them to the main catalogue.")
