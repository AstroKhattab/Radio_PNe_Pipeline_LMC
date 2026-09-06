"""
step3a_inspect_meerkat.py
=========================
Step 3a — Split and inspect the parent PNe that MeerKAT did not cross-match.

The unmatched sources are split into those that fall inside the mosaic and
those that fall outside it, and a radio / Halpha / [O III] cutout page is
written for each of the former.  Those pages are what the visual detection
list in manual/visual_detections.txt is read off; nothing here decides
anything by itself.

Outputs
-------
    03_Outputs/step3a_meerkat_outside.vot   parent PNe outside the mosaic
    03_Outputs/step3a_meerkat_to_check.vot  inside the mosaic, to inspect
    04_Inspect/figures/step3a_meerkat_matched_cutouts.pdf
    04_Inspect/figures/step3a_meerkat_to_check_cutouts.pdf

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os, sys, warnings
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
from astropy.utils.exceptions import AstropyWarning
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

warnings.simplefilter("ignore", AstropyWarning)

cfg = config.load()
OUTS  = cfg.out()
# The cutout pages take far longer than everything else in the pipeline put
# together and are quality control, not a paper figure.  --no-cutouts writes
# the tables and skips them.
MAKE_CUTOUTS = "--no-cutouts" not in sys.argv
INSP  = cfg.inspect("figures")
os.makedirs(INSP, exist_ok=True)

MKT3_FILE  = cfg.data("meerkat_3sigma")
HULL_STEP  = 50   # sample every 50th source when tracing the mosaic outline

# ── Image config (matching your PNE_Cutout_Script_flexible.py style) ──────────
CUTOUT_SIZE       = 60      # arcsec
SCALE_PERCENTILE  = 99.99
MARKER_COLOR      = "cyan"
MARKER_SIZE       = 3       # pixels (radius)
DPI               = 100
PANEL_SIZE        = 5       # inches per panel

IMAGES = [
    {"name": "Radio",   "file": cfg.data("meerkat_mosaic"),
     "is_radio": True,  "colormap": "magma", "enabled": True},
    {"name": "H-alpha", "file": cfg.data("mcels_ha"),
     "is_radio": False, "colormap": "magma", "enabled": True},
    {"name": "[O III]", "file": cfg.data("mcels_oiii"),
     "is_radio": False, "colormap": "magma", "enabled": True},
]

print("=" * 60)
print("  step3a_inspect_meerkat.py")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Load catalogues ────────────────────────────────────────────────────────────
print("\n[1] Loading catalogues...")
matched   = Table.read(os.path.join(OUTS, "step2a_meerkat_matched.vot"), format="votable")
unmatched = Table.read(os.path.join(OUTS, "step2a_meerkat_unmatched.vot"), format="votable")
mkt3      = Table.read(MKT3_FILE, format="votable")

print(f"    Matched (Step 2a)  : {len(matched)}")
print(f"    Unmatched (Step 2a): {len(unmatched)}")

# ── Build MeerKAT footprint ────────────────────────────────────────────────────
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
assert n_outside + n_to_check == len(unmatched), \
    "unmatched sources lost between inside and outside the mosaic"
print(f"    Inside  (to check): {n_to_check}")
print(f"    Outside (skip)    : {n_outside}")

# ── Write catalogues ───────────────────────────────────────────────────────────
print("\n[3] Writing catalogues...")
tbl_outside  = unmatched[outside]
tbl_to_check = unmatched[inside]

tbl_outside.add_column( Column(np.full(len(tbl_outside),  "outside",  dtype="U10"), name="step2_class"))
tbl_to_check.add_column(Column(np.full(len(tbl_to_check), "to_check", dtype="U10"), name="step2_class"))

tbl_outside.write(os.path.join(OUTS, "step3a_meerkat_outside.vot"), format="votable", overwrite=True)
tbl_to_check.write(os.path.join(OUTS, "step3a_meerkat_to_check.vot"), format="votable", overwrite=True)
print(f"    step3a_meerkat_outside.vot -> {n_outside} rows (excluded — outside region)")
print(f"    step3a_meerkat_to_check.vot -> {n_to_check} rows  (inspect manually)")

if not MAKE_CUTOUTS:
    print("\n[4] --no-cutouts: tables written, inspection pages skipped")
    raise SystemExit(0)

# ── Load FITS images once ──────────────────────────────────────────────────────
print("\n[4] Loading FITS images...")

def load_fits(filepath):
    if not os.path.exists(filepath):
        print(f"    WARNING: not found — {filepath}")
        return None, None, None
    try:
        # Keep the array memory-mapped and in its native dtype.  These mosaics
        # run to 24235 x 36281 pixels; promoting one to float64 up front asks
        # for 6.5 GB, and three of them at once will exhaust any ordinary
        # machine.  Every use below is a small cutout, cast where it is taken.
        hdul = fits.open(filepath, memmap=True)
        for hdu in hdul:
            if hdu.data is not None and len(hdu.data.shape) >= 2:
                data = hdu.data
                while data.ndim > 2:
                    data = data[0]
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

# ── Cutout function (your script's method) ────────────────────────────────────
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
    """One page: Radio | H-alpha | [O III]  — your script's style."""
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

# ── Generate cutout PDFs ───────────────────────────────────────────────────────
print("\n[5] Generating cutout PDFs...")

# Detected: centre on RADIO position
write_pdf(
    out_path  = os.path.join(INSP, "step3a_meerkat_matched_cutouts.pdf"),
    tbl       = matched,
    ra_col    = "mkt_ra",    # radio centroid
    dec_col   = "mkt_dec",
    name_col  = "RP_ID",
    label     = "Detected (cross-matched)",
)

# To-check: centre on OPTICAL (Reid) position
write_pdf(
    out_path  = os.path.join(INSP, "step3a_meerkat_to_check_cutouts.pdf"),
    tbl       = tbl_to_check,
    ra_col    = "RA",
    dec_col   = "Dec",
    name_col  = "RP_ID",
    label     = "To check (inside region, not cross-matched)",
)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  STEP 3 COMPLETE")
print(f"{'='*60}")
print(f"  Unmatched at step 2a    : {len(unmatched)}")
print(f"  Cross-matched (radio)   : {len(matched)}")
print(f"  Outside MeerKAT region  : {n_outside}  (skipped)")
print(f"  To inspect manually     : {n_to_check}")
print(f"\n  Catalogues:")
print(f"    step3a_meerkat_outside.vot  ({n_outside} sources)")
print(f"    step3a_meerkat_to_check.vot ({n_to_check} sources)")
print(f"\n  Inspection PDFs (04_Inspect/figures/):")
print(f"    step3a_meerkat_matched_cutouts.pdf ({len(matched)} pages)")
print(f"    step3a_meerkat_to_check_cutouts.pdf ({n_to_check} pages)")
print(f"\n  Read the visual detections off the to-check pages, list them in")
print(f"  manual/visual_detections.txt, then run step3b.")
print(f"{'='*60}")
