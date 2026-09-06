"""
step3c_detection_catalogue.py
=============================
Step 4a — Build the MeerKAT detected/non-detected catalogues and generate the
associated paper figures. The sky map also shows ASKAP coverage and overlap.

Inputs
------
    03_Outputs/step2a_meerkat_matched.vot   catalogue cross-matches
    03_Outputs/step3b_meerkat_visual.vot    sub-threshold recoveries
    03_Outputs/step3a_meerkat_outside.vot   outside the mosaic
    03_Outputs/step3a_meerkat_to_check.vot  inside it, inspected
    03_Outputs/step2b_askap_matched.vot     accepted ASKAP associations
    the MeerKAT and ASKAP mosaics, for the survey footprints

Outputs
-------
Catalogues (03_Outputs/):
    step3c_meerkat_detected.vot — cross-matched + visually confirmed
    step3c_meerkat_nondetected.vot — inside region, not detected

Paper figures (05_Figures/) — NO TITLES on any figure:
    step3c_multisurvey_sky_map_detections.pdf — footprints and survey overlap
    step3c_multisurvey_sky_map_reid_classes.pdf — footprints and Reid classes
    step3c_multisurvey_detection_summary.pdf — survey totals + MeerKAT class bars
    step3c_meerkat_positional_offsets.pdf — Reid-versus-radio offsets
    step3c_meerkat_snr_distribution.pdf — S/N distribution
    step3c_meerkat_detection_breakdown.pdf — source-count breakdown

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from astropy.table import Table, Column, vstack
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config
from _support.plot_style import apply_paper_style

warnings.simplefilter("ignore", AstropyWarning)
apply_paper_style()

cfg = config.load()
OUTS  = cfg.out()
FIGS  = cfg.fig()

MEERKAT_BEAM_FILE = cfg.data("meerkat_mosaic")
ASKAP_IMAGE_FILE = cfg.data("askap_image")
ASKAP_MATCHED_FILE = cfg.out("step2b_askap_matched.vot")

# Colours matched to the reference figure
COL    = {"True":"#6baed6", "Known":"#74c476", "Likely":"#fd8d3c",
          "Possible":"#fb6a4a", "Unknown":"#bdbdbd"}
MARKER = {"True":".", "Known":"s", "Likely":".", "Possible":"D", "Unknown":"."}
MSIZE  = {"True":20, "Known":12, "Likely":20, "Possible":12, "Unknown":10}
# Ordered by how secure the optical classification is, not alphabetically:
# True and Known are spectroscopically confirmed, Possible and Likely are not.
CLASSES = ["True", "Known", "Possible", "Likely"]

print("step3c: building the MeerKAT detection catalogue")

# Load all catalogues
print("\n[1] Loading catalogues...")
matched = Table.read(os.path.join(OUTS, "step2a_meerkat_matched.vot"), format="votable")
outside = Table.read(os.path.join(OUTS, "step3a_meerkat_outside.vot"), format="votable")
to_check = Table.read(os.path.join(OUTS, "step3a_meerkat_to_check.vot"), format="votable")

visual = Table.read(os.path.join(OUTS, "step3b_meerkat_visual.vot"), format="votable")
print(f"    step3b_meerkat_visual.vot : {len(visual)} visually confirmed")

print(f"    step2a_meerkat_matched.vot : {len(matched)} cross-matched")
print(f"    step3a_meerkat_outside.vot : {len(outside)} outside region")
print(f"    step3a_meerkat_to_check.vot: {len(to_check)} to check")

askap_matched = Table.read(ASKAP_MATCHED_FILE, format="votable")
print(f"    step2b_askap_matched.vot   : {len(askap_matched)} accepted")

reid_full = Table.read(os.path.join(OUTS, "step1_parent_catalogue.vot"), format="votable")
n_total   = len(reid_full)
ra_all    = np.array(reid_full["RA"],  dtype=float)
dec_all   = np.array(reid_full["Dec"], dtype=float)
cls_all   = np.array(reid_full["reid_class"], dtype=str)

# Build detected catalogue
print("\n[2] Building final catalogues...")


vis_rpids = set(np.array(visual["RP_ID"], dtype=str))
check_rpids = np.array(to_check["RP_ID"], dtype=str)
nondetected = to_check[np.array([rp not in vis_rpids for rp in check_rpids])]

# The catalogue holds fluxes in Jy under one set of names, the visual
# extraction in mJy under another.  Put the visual rows onto the catalogue's
# names and units before stacking; taking the intersection of the two column
# sets as they stand would quietly drop every flux column from the result.
vis = visual.copy()
vis["mkt_peak_flux_Jy"] = np.asarray(vis["mkt_peak_flux_mJy"], float) / 1e3
vis["mkt_int_flux_Jy"] = np.asarray(vis["mkt_int_flux_mJy"], float) / 1e3
vis["mkt_err_int_flux_Jy"] = np.asarray(vis["mkt_err_int_flux_mJy"], float) / 1e3
vis["mkt_local_rms_Jy"] = np.asarray(vis["mkt_local_rms_mJy"], float) / 1e3
vis["separation_arcsec"] = np.asarray(vis["offset_arcsec"], float)

common_cols = [c for c in matched.colnames if c in vis.colnames]
detected = vstack([matched[common_cols], vis[common_cols]])
detected["detection_method"] = np.concatenate([
    np.full(len(matched), "cross_matched"), np.full(len(vis), "visual")])

n_xmatch  = len(matched)
n_visual  = len(visual)
n_det     = len(detected)
n_nondet  = len(nondetected)
n_outside = len(outside)

detected.write(os.path.join(OUTS, "step3c_meerkat_detected.vot"), format="votable", overwrite=True)
nondetected.write(os.path.join(OUTS, "step3c_meerkat_nondetected.vot"), format="votable", overwrite=True)

print(f"    step3c_meerkat_detected.vot -> {n_det} rows "
      f"({n_xmatch} cross-matched + {n_visual} visual)")
print(f"    step3c_meerkat_nondetected.vot -> {n_nondet} rows")

# FITS-derived survey footprints
def finite_image_outline(path, stride=64):
    """Return the actual finite-data boundary of a large FITS mosaic.

    The FITS arrays contain padded blank pixels.  Sampling the finite-pixel mask
    and contouring its outer boundary preserves the real mosaic shape, including
    its rotation and non-rectangular edges, without loading a multi-GB copy.
    """
    with fits.open(path, memmap=True) as hdul:
        image = np.squeeze(hdul[0].data)
        image_wcs = WCS(hdul[0].header).celestial
        sampled = np.asarray(image[::stride, ::stride])

    valid = np.isfinite(sampled)
    if not np.any(valid):
        raise RuntimeError(f"No finite image pixels found in {path}")

    contour_fig, contour_ax = plt.subplots(figsize=(2, 2))
    contour_set = contour_ax.contour(valid.astype(float), levels=[0.5])
    contour_segments = contour_set.allsegs[0]
    plt.close(contour_fig)
    if not contour_segments:
        raise RuntimeError(f"Could not trace finite image boundary in {path}")

    boundary = max(contour_segments, key=len)
    pixel_x = boundary[:, 0] * stride
    pixel_y = boundary[:, 1] * stride
    outline_ra, outline_dec = image_wcs.pixel_to_world_values(pixel_x, pixel_y)
    if outline_ra[0] != outline_ra[-1] or outline_dec[0] != outline_dec[-1]:
        outline_ra = np.append(outline_ra, outline_ra[0])
        outline_dec = np.append(outline_dec, outline_dec[0])
    return outline_ra, outline_dec


def finite_image_membership(path, ra_deg, dec_deg):
    """Test sky positions against the original, unsampled FITS finite mask."""
    with fits.open(path, memmap=True) as hdul:
        image = np.squeeze(hdul[0].data)
        image_wcs = WCS(hdul[0].header).celestial
        pixel_x, pixel_y = image_wcs.world_to_pixel_values(ra_deg, dec_deg)
        index_x = np.rint(pixel_x).astype(int)
        index_y = np.rint(pixel_y).astype(int)
        in_bounds = (
            (index_x >= 0) & (index_y >= 0)
            & (index_x < image.shape[1]) & (index_y < image.shape[0])
        )
        covered = np.zeros(len(ra_deg), dtype=bool)
        covered[in_bounds] = np.isfinite(
            image[index_y[in_bounds], index_x[in_bounds]]
        )
    return covered


meerkat_ra, meerkat_dec = finite_image_outline(MEERKAT_BEAM_FILE)
askap_ra, askap_dec = finite_image_outline(ASKAP_IMAGE_FILE)
meerkat_coverage = finite_image_membership(MEERKAT_BEAM_FILE, ra_all, dec_all)
askap_coverage = finite_image_membership(ASKAP_IMAGE_FILE, ra_all, dec_all)
print(
    "    FITS footprints: "
    f"MeerKAT RA {np.nanmin(meerkat_ra):.2f}-{np.nanmax(meerkat_ra):.2f} deg, "
    f"Dec {np.nanmin(meerkat_dec):.2f}-{np.nanmax(meerkat_dec):.2f} deg; "
    f"ASKAP RA {np.nanmin(askap_ra):.2f}-{np.nanmax(askap_ra):.2f} deg, "
    f"Dec {np.nanmin(askap_dec):.2f}-{np.nanmax(askap_dec):.2f} deg"
)
print(
    f"    FITS coverage check: MeerKAT {meerkat_coverage.sum()}/{len(ra_all)} "
    f"(outside {(~meerkat_coverage).sum()}); ASKAP "
    f"{askap_coverage.sum()}/{len(ra_all)} "
    f"(outside {(~askap_coverage).sum()})"
)
if int((~meerkat_coverage).sum()) != n_outside:
    warnings.warn(
        "MeerKAT FITS-mask outside count no longer agrees with Step 03a; "
        "inspect the footprint before publication."
    )

# Coordinates for plotting
ra_m  = np.array(matched["RA"],  dtype=float)
dec_m = np.array(matched["Dec"], dtype=float)
cls_m = np.array(matched["reid_class"], dtype=str)
snr_m = np.array(matched["mkt_snr"],    dtype=float)

ra_nd  = np.array(nondetected["RA"],  dtype=float)
dec_nd = np.array(nondetected["Dec"], dtype=float)
cls_nd = np.array(nondetected["reid_class"], dtype=str)

ra_out  = np.array(outside["RA"],  dtype=float)
dec_out = np.array(outside["Dec"], dtype=float)

ra_v  = np.array(visual["RA"],  dtype=float)
dec_v = np.array(visual["Dec"], dtype=float)

# PAPER FIGURES — no titles (captions added in Overleaf)
print("\n[3] Generating paper figures (no titles)...")

# step3c_multisurvey_sky_map_detections.pdf
# This figure presents both survey footprints and their common-source population.
# Build survey-detection masks on the common Reid optical positions.
reid_ids = np.array([str(value).strip() for value in reid_full["RP_ID"]])
mkt_ids = {str(value).strip() for value in detected["RP_ID"] if str(value).strip()}
askap_ids = {
    str(value).strip() for value in askap_matched["RP_ID"] if str(value).strip()
}
det_mkt_map = np.array([value in mkt_ids for value in reid_ids], dtype=bool)
det_askap_map = np.array([value in askap_ids for value in reid_ids], dtype=bool)
mask_both = det_mkt_map & det_askap_map
mask_mkt_only = det_mkt_map & ~det_askap_map
mask_askap_only = ~det_mkt_map & det_askap_map
mask_neither = ~det_mkt_map & ~det_askap_map

n_both_map = int(mask_both.sum())
n_mkt_only_map = int(mask_mkt_only.sum())
n_askap_only_map = int(mask_askap_only.sum())
n_neither_map = int(mask_neither.sum())

# Colour-blind-friendly survey palette.
mkt_colour = "#0072B2"
askap_colour = "#8E5AA9"
both_colour = "#6F2DBD"
askap_only_colour = "#E69F00"
background_colour = "#B7BEC8"

# Survey totals overlap; the exclusive groups below do not.
survey_total_specs = [
    ("MeerKAT", n_det, mkt_colour, "#EAF4FB"),
    ("ASKAP", len(askap_matched), askap_colour, "#F3ECF7"),
    ("Radio union", n_mkt_only_map + n_both_map + n_askap_only_map,
     "#287A55", "#E6F4ED"),
]
exclusive_specs = [
    ("MeerKAT only", n_mkt_only_map, mkt_colour, "#EAF4FB"),
    ("Both surveys", n_both_map, both_colour, "#EEE8FA"),
    ("ASKAP only", n_askap_only_map, "#B97800", "#FFF3D6"),
]

# LMC-centred SIN projection: a Cartesian RA/Dec frame distorts these wide mosaics.
plot_wcs = WCS(naxis=2)
plot_wcs.wcs.crpix = [500.0, 400.0]
plot_wcs.wcs.cdelt = np.array([-0.025, 0.025])
plot_wcs.wcs.crval = [79.24927079022, -68.9420520621]
plot_wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
plot_wcs.wcs.cunit = ["deg", "deg"]

fig = plt.figure(figsize=(10.2, 8.2), facecolor="white")
ax = fig.add_subplot(111, projection=plot_wcs)
world = ax.get_transform("world")

# Survey footprints from populated/valid coverage.
ax.fill(askap_ra, askap_dec, color=askap_colour, alpha=0.035,
        transform=world, zorder=0)
ax.plot(askap_ra, askap_dec, color=askap_colour, lw=1.3,
        ls=(0, (5, 2, 1.3, 2)), transform=world, zorder=1)
ax.fill(meerkat_ra, meerkat_dec, color=mkt_colour, alpha=0.045,
        transform=world, zorder=0)
ax.plot(meerkat_ra, meerkat_dec, color=mkt_colour, lw=1.25, ls="--",
        transform=world, zorder=2)

# Quiet contextual layer for Reid sources without a radio detection.
ax.scatter(ra_all[mask_neither], dec_all[mask_neither],
           s=11, color=background_colour, alpha=0.55, linewidths=0,
           transform=world, zorder=3, rasterized=True)

# Single-survey and common detections use distinct colour/shape combinations.
ax.scatter(ra_all[mask_mkt_only], dec_all[mask_mkt_only],
           s=38, facecolors="white", edgecolors=mkt_colour,
           linewidths=0.9, marker="o", alpha=0.88,
           transform=world, zorder=5, rasterized=True)
ax.scatter(ra_all[mask_askap_only], dec_all[mask_askap_only],
           s=74, facecolors=askap_only_colour, edgecolors="white",
           linewidths=0.8, marker="D", alpha=0.98,
           transform=world, zorder=7, rasterized=True)
ax.scatter(ra_all[mask_both], dec_all[mask_both],
           s=72, facecolors=both_colour, edgecolors="white",
           linewidths=0.75, marker="P", alpha=0.96,
           transform=world, zorder=8, rasterized=True)

ax.coords[0].set_axislabel("Right Ascension (ICRS)", fontsize=12)
ax.coords[1].set_axislabel("Declination (ICRS)", fontsize=12)
ax.coords[0].set_major_formatter("hh:mm")
ax.coords[0].set_ticks(spacing=15 * u.deg)
ax.coords[1].set_ticks(spacing=2 * u.deg)
ax.coords[0].set_ticks_position("b")
ax.coords[0].set_ticklabel_position("b")
ax.coords[1].set_ticks_position("l")
ax.coords[1].set_ticklabel_position("l")
ax.coords[0].set_ticklabel(size=9.5)
ax.coords[1].set_ticklabel(size=9.5)
ax.coords.grid(True, color="#DDE2E8", linewidth=0.55, alpha=0.65)

# Frame the combined finite-data footprints, with modest padding so neither
# survey boundary is clipped.  Correct for RA convergence at LMC declination.
footprint_ra = np.concatenate([meerkat_ra, askap_ra])
footprint_dec = np.concatenate([meerkat_dec, askap_dec])
footprint_x, footprint_y = plot_wcs.world_to_pixel_values(
    footprint_ra, footprint_dec
)
x_span = float(np.nanmax(footprint_x) - np.nanmin(footprint_x))
y_span = float(np.nanmax(footprint_y) - np.nanmin(footprint_y))
ax.set_xlim(
    float(np.nanmin(footprint_x) - 0.07 * x_span),
    float(np.nanmax(footprint_x) + 0.07 * x_span),
)
ax.set_ylim(
    float(np.nanmin(footprint_y) - 0.08 * y_span),
    float(np.nanmax(footprint_y) + 0.08 * y_span),
)
ax.set_aspect("equal", adjustable="box")

detection_legend = [
    Line2D([0], [0], marker=".", color="none",
           markerfacecolor=background_colour, markeredgecolor=background_colour,
           markersize=7, label=f"No radio detection (N={n_neither_map})"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor=mkt_colour, markeredgewidth=1.0, markersize=7,
           label=f"MeerKAT only (N={n_mkt_only_map})"),
    Line2D([0], [0], marker="P", color="none", markerfacecolor=both_colour,
           markeredgecolor="white", markeredgewidth=0.7, markersize=8,
           label=f"MeerKAT + ASKAP (N={n_both_map})"),
    Line2D([0], [0], marker="D", color="none",
           markerfacecolor=askap_only_colour, markeredgecolor="white",
           markeredgewidth=0.7, markersize=7,
           label=f"ASKAP only (N={n_askap_only_map})"),
]
leg_det = ax.legend(
    handles=detection_legend, title="Parent-sample detections",
    fontsize=8.8, title_fontsize=9.2, framealpha=0.96,
    loc="upper left", frameon=True, borderpad=0.65,
)
leg_det.get_frame().set_edgecolor("#CAD2DC")
ax.add_artist(leg_det)

footprint_legend = [
    Line2D([0], [0], color=mkt_colour, lw=1.4, ls="--",
           label="MeerKAT 1.295 GHz finite-data footprint"),
    Line2D([0], [0], color=askap_colour, lw=1.4,
           ls=(0, (5, 2, 1.3, 2)),
           label="ASKAP 888 MHz finite-data footprint"),
]
leg_fp = ax.legend(
    handles=footprint_legend, title="Survey coverage",
    fontsize=8.2, title_fontsize=8.8, framealpha=0.96,
    loc="lower left", frameon=True, borderpad=0.6,
)
leg_fp.get_frame().set_edgecolor("#CAD2DC")

# Counts are read from the data rather than written in by hand, so the
# annotation cannot drift out of step with the sample the way it did when
# the parent catalogue changed.
info = ("FITS masks reprojected to a common ICRS SIN grid\n"
        f"MeerKAT detections: {int(n_xmatch)} catalogue + {int(n_visual)} visual")
ax.text(0.98, 0.03, info, transform=ax.transAxes, fontsize=7.6,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#BDBDBD", alpha=0.94))

fig.subplots_adjust(left=0.11, right=0.985, bottom=0.09, top=0.985)
plt.savefig(os.path.join(FIGS, "step3c_multisurvey_sky_map_detections.pdf"),
            format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("    step3c_multisurvey_sky_map_detections.pdf")

# step3c_multisurvey_sky_map_reid_classes.pdf
# Companion view of the same two FITS footprints, encoding the original Reid
# optical classes rather than the radio-detection combinations shown above.
fig = plt.figure(figsize=(11.2, 8.6), facecolor="white")
ax = fig.add_subplot(111, projection=plot_wcs)
world = ax.get_transform("world")

ax.fill(askap_ra, askap_dec, color=askap_colour, alpha=0.030,
        transform=world, zorder=0)
ax.plot(askap_ra, askap_dec, color=askap_colour, lw=1.3,
        ls=(0, (5, 2, 1.3, 2)), transform=world, zorder=1)
ax.fill(meerkat_ra, meerkat_dec, color=mkt_colour, alpha=0.040,
        transform=world, zorder=0)
ax.plot(meerkat_ra, meerkat_dec, color=mkt_colour, lw=1.25, ls="--",
        transform=world, zorder=2)

class_colours = {
    "True": "#4EA8DE", "Known": "#2A9D8F", "Likely": "#F4A261",
    "Possible": "#E76F51", "Unknown": "#6C757D",
}
class_markers = {
    "True": "o", "Known": "s", "Likely": "^", "Possible": "D",
    "Unknown": "X",
}
class_sizes = {
    "True": 10, "Known": 14, "Likely": 19, "Possible": 18, "Unknown": 21,
}
class_order = ["True", "Known", "Possible", "Likely", "Unknown"]
radio_detected = det_mkt_map | det_askap_map
for class_name in class_order:
    class_mask = cls_all == class_name
    quiet_mask = class_mask & ~radio_detected
    detected_mask = class_mask & radio_detected
    # Non-detections remain visible as context, but detections receive stronger
    # class symbols before the independent survey-status rings are added.
    ax.scatter(
        ra_all[quiet_mask], dec_all[quiet_mask],
        s=class_sizes[class_name], marker=class_markers[class_name],
        facecolors=class_colours[class_name], edgecolors="none", linewidths=0,
        alpha=0.34, transform=world, zorder=3, rasterized=True,
    )
    if np.any(detected_mask):
        ax.scatter(
            ra_all[detected_mask], dec_all[detected_mask],
            s=class_sizes[class_name] * 1.18,
            marker=class_markers[class_name],
            facecolors=class_colours[class_name], edgecolors="white",
            linewidths=0.35, alpha=0.98,
            transform=world, zorder=5, rasterized=True,
        )

# Detection-status rings are drawn over, not instead of, the Reid class symbol.
ring_colours = {
    "MeerKAT only": "#0057B8",
    "MeerKAT + ASKAP": "#7B2CBF",
    "ASKAP only": "#D97706",
}
ax.scatter(
    ra_all[mask_mkt_only], dec_all[mask_mkt_only], s=31,
    marker="o", facecolors="none", edgecolors=ring_colours["MeerKAT only"],
    linewidths=0.72, alpha=0.90, transform=world, zorder=7,
    rasterized=True,
)
ax.scatter(
    ra_all[mask_both], dec_all[mask_both], s=39,
    marker="o", facecolors="none", edgecolors=ring_colours["MeerKAT + ASKAP"],
    linewidths=1.15, alpha=0.98, transform=world, zorder=8,
    rasterized=True,
)
ax.scatter(
    ra_all[mask_askap_only], dec_all[mask_askap_only], s=47,
    marker="o", facecolors="none", edgecolors=ring_colours["ASKAP only"],
    linewidths=1.35, alpha=1.0, transform=world, zorder=9,
    rasterized=True,
)

ax.coords[0].set_axislabel("Right Ascension (ICRS)", fontsize=12)
ax.coords[1].set_axislabel("Declination (ICRS)", fontsize=12)
ax.coords[0].set_major_formatter("hh:mm")
ax.coords[0].set_ticks(spacing=15 * u.deg)
ax.coords[1].set_ticks(spacing=2 * u.deg)
ax.coords[0].set_ticks_position("b")
ax.coords[0].set_ticklabel_position("b")
ax.coords[1].set_ticks_position("l")
ax.coords[1].set_ticklabel_position("l")
ax.coords[0].set_ticklabel(size=9.5)
ax.coords[1].set_ticklabel(size=9.5)
ax.coords.grid(True, color="#DDE2E8", linewidth=0.55, alpha=0.65)
ax.set_xlim(
    float(np.nanmin(footprint_x) - 0.07 * x_span),
    float(np.nanmax(footprint_x) + 0.07 * x_span),
)
ax.set_ylim(
    float(np.nanmin(footprint_y) - 0.08 * y_span),
    float(np.nanmax(footprint_y) + 0.08 * y_span),
)
ax.set_aspect("equal", adjustable="box")

class_legend = [
    Line2D([0], [0], marker=class_markers[name], color="none",
           markerfacecolor=class_colours[name], markeredgecolor="white",
           markeredgewidth=0.5, markersize=7.5,
           label=f"{name} (N={int((cls_all == name).sum())})")
    for name in class_order
]
leg_cls = ax.legend(
    handles=class_legend, title="Optical classification",
    fontsize=8.8, title_fontsize=9.2, framealpha=0.96,
    loc="upper left", frameon=True, borderpad=0.65,
)
leg_cls.get_frame().set_edgecolor("#CAD2DC")
ax.add_artist(leg_cls)

detection_ring_legend = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor=ring_colours["MeerKAT only"],
           markeredgewidth=1.0, markersize=7,
           label=f"MeerKAT only (N={n_mkt_only_map})"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor=ring_colours["MeerKAT + ASKAP"],
           markeredgewidth=1.4, markersize=8,
           label=f"MeerKAT + ASKAP (N={n_both_map})"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor=ring_colours["ASKAP only"],
           markeredgewidth=1.5, markersize=8.5,
           label=f"ASKAP only (N={n_askap_only_map})"),
]
leg_radio = ax.legend(
    handles=detection_ring_legend, title="Coloured ring = radio detection",
    fontsize=8.6, title_fontsize=9.0, framealpha=0.96,
    loc="upper right", frameon=True, borderpad=0.65,
)
leg_radio.get_frame().set_edgecolor("#CAD2DC")
ax.add_artist(leg_radio)

leg_fp = ax.legend(
    handles=footprint_legend, title="Survey coverage",
    fontsize=8.2, title_fontsize=8.8, framealpha=0.96,
    loc="lower left", frameon=True, borderpad=0.6,
)
leg_fp.get_frame().set_edgecolor("#CAD2DC")
ax.text(
    0.98, 0.03,
    "Fill and shape = Reid class\nColoured ring = radio detection",
    transform=ax.transAxes, fontsize=7.8, va="bottom", ha="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#BDBDBD", alpha=0.94),
)
fig.subplots_adjust(left=0.11, right=0.985, bottom=0.09, top=0.985)
plt.savefig(
    os.path.join(FIGS, "step3c_multisurvey_sky_map_reid_classes.pdf"),
    format="pdf", bbox_inches="tight", facecolor="white",
)
plt.close()
print("    step3c_multisurvey_sky_map_reid_classes.pdf")

# step3c_meerkat_positional_offsets.pdf: cross-matched and visual combined
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from matplotlib.offsetbox import AnchoredText
from astropy.coordinates import SkyCoord as SC2

ra_opt_m  = ra_m[np.isfinite(ra_m)]
dec_opt_m = dec_m[np.isfinite(ra_m)]
mkt_ra_m  = np.array(matched["mkt_ra"],  dtype=float)[np.isfinite(ra_m)]
mkt_dec_m = np.array(matched["mkt_dec"], dtype=float)[np.isfinite(ra_m)]
c_o_m = SC2(ra=ra_opt_m * u.deg, dec=dec_opt_m * u.deg)
c_r_m = SC2(ra=mkt_ra_m * u.deg, dec=mkt_dec_m * u.deg)
dra_m, ddec_m = c_o_m.spherical_offsets_to(c_r_m)
x_m = dra_m.to(u.arcsec).value; y_m = ddec_m.to(u.arcsec).value

if len(visual) > 0:
    ra_ov  = np.array(visual["RA"],      dtype=float)
    dec_ov = np.array(visual["Dec"],     dtype=float)
    mra_v  = np.array(visual["mkt_ra"],  dtype=float)
    mdec_v = np.array(visual["mkt_dec"], dtype=float)
    ok_v   = np.isfinite(ra_ov) & np.isfinite(mra_v)
    c_ov = SC2(ra=ra_ov[ok_v] * u.deg, dec=dec_ov[ok_v] * u.deg)
    c_rv = SC2(ra=mra_v[ok_v] * u.deg, dec=mdec_v[ok_v] * u.deg)
    dra_v, ddec_v = c_ov.spherical_offsets_to(c_rv)
    x_v = dra_v.to(u.arcsec).value; y_v = ddec_v.to(u.arcsec).value
else:
    x_v = np.array([]); y_v = np.array([])

x_all = np.concatenate([x_m, x_v]); y_all = np.concatenate([y_m, y_v])
fin = np.isfinite(x_all) & np.isfinite(y_all)
x_all = x_all[fin]; y_all = y_all[fin]

plt.style.use("ggplot")
fig = plt.figure(figsize=(7, 7))
gs_o = GridSpec(1, 1, figure=fig)
gs_i = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_o[0],
    width_ratios=[6, 1, 0.35], height_ratios=[1, 6], wspace=0.10, hspace=0.10)
ax_hx = fig.add_subplot(gs_i[0, 0])
ax_mn = fig.add_subplot(gs_i[1, 0])
ax_hy = fig.add_subplot(gs_i[1, 1], sharey=ax_mn)
ax_cb = fig.add_subplot(gs_i[1, 2])

XLIM = (-6, 6); YLIM = (-6, 6)
hb = ax_mn.hexbin(x_all, y_all, gridsize=30, cmap="plasma", mincnt=1)
ax_mn.set_xlabel("\u0394RA \u00b7 cos(\u03b4)  (arcsec)", fontsize=13)
ax_mn.set_ylabel("\u0394\u03b4  (arcsec)", fontsize=13)
ax_mn.set_xlim(*XLIM); ax_mn.set_ylim(*YLIM)
ax_mn.axhline(0, color="black", lw=1.5); ax_mn.axvline(0, color="black", lw=1.5)
ax_mn.grid(axis="both", ls="--", alpha=0.3)
ax_hx.hist(x_all, bins="auto", color="#D32F2F", alpha=0.55)
ax_hx.axvline(np.mean(x_all), color="black", ls="--", lw=1.2)
ax_hx.set_xlim(*XLIM); plt.setp(ax_hx.get_xticklabels(), visible=False)
ax_hy.hist(y_all, bins="auto", color="#D32F2F", alpha=0.55, orientation="horizontal")
ax_hy.axhline(np.mean(y_all), color="black", ls="--", lw=1.2)
ax_hy.set_ylim(*YLIM); plt.setp(ax_hy.get_yticklabels(), visible=False)
cbar = fig.colorbar(hb, cax=ax_cb); cbar.set_label("Counts per bin", fontsize=11)
stats = "\n".join([
    fr"$\langle\Delta\mathrm{{RA}}\rangle = {np.mean(x_all):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\mathrm{{RA}}) = {np.std(x_all):.2f}^{{\prime\prime}}$",
    fr"$\langle\Delta\delta\rangle = {np.mean(y_all):+.2f}^{{\prime\prime}}$",
    fr"$\sigma(\Delta\delta) = {np.std(y_all):.2f}^{{\prime\prime}}$",
    f"$N = {len(x_all)}$",
])
at = AnchoredText(stats, prop=dict(size=10), frameon=True, loc="upper left")
at.patch.set_boxstyle("round,pad=0.3"); at.patch.set_alpha(0.9)
ax_mn.add_artist(at)
plt.savefig(os.path.join(FIGS, "step3c_meerkat_positional_offsets.pdf"),
            format="pdf", bbox_inches="tight", facecolor="white")
plt.close(); plt.style.use("default")
print(f"    step3c_meerkat_positional_offsets.pdf (N={len(x_all)}: {n_xmatch} XM + {len(x_v)} visual)")

# step3c_meerkat_snr_distribution.pdf: cross-matched and visual combined
snr_xm = snr_m[np.isfinite(snr_m)]
if len(visual) > 0:
    snr_vi = np.array(visual["mkt_snr"], dtype=float)
    snr_vi = snr_vi[np.isfinite(snr_vi)]
else:
    snr_vi = np.array([])
snr_vals = np.concatenate([snr_xm, snr_vi])

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")
log_bins = np.logspace(np.log10(max(snr_vals.min(), 0.5)),
                       np.log10(snr_vals.max() + 1), 35)
ax.hist(snr_xm, bins=log_bins, color="#455A64", edgecolor="white", lw=0.5,
        zorder=2, label=f"Cross-matched (N={len(snr_xm)})")
if len(snr_vi) > 0:
    ax.hist(snr_vi, bins=log_bins, color="#FF9800", edgecolor="white", lw=0.5,
            zorder=3, alpha=0.7, label=f"Visual (N={len(snr_vi)})")
ax.axvline(3.0, color="#F57C00", lw=2, ls="--", zorder=4, label="3\u03c3")
ax.axvline(5.0, color="#D32F2F", lw=2, ls="--", zorder=4, label="5\u03c3")
ax.set_xscale("log")
ax.set_xlabel("Signal-to-Noise Ratio  (peak flux / local RMS)", fontsize=11)
ax.set_ylabel("Number of sources", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.tick_params(labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "step3c_meerkat_snr_distribution.pdf"),
            format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    step3c_meerkat_snr_distribution.pdf (N={len(snr_vals)}: {len(snr_xm)} XM + {len(snr_vi)} visual)")

# step3c_meerkat_detection_breakdown.pdf: donut, counts only
n_inside_det   = n_det
n_inside_nondet = n_nondet

outer_vals   = [n_inside_det + n_nondet, n_outside]
outer_labels = [f"Inside\n({n_inside_det + n_nondet})",
                f"Outside\n({n_outside})"]
outer_colors = ["#1565C0", "#EE1010"]

inner_vals   = [n_xmatch, n_visual, n_nondet]
inner_labels = [f"Cross-matched\n{n_xmatch}",
                f"Visual\n{n_visual}" if n_visual > 0 else "",
                f"Non-detected\n{n_nondet}"]
inner_colors = ["#455A64", "#FF9800", "#E5E535"]

# Remove empty slices
if n_visual == 0:
    inner_vals   = [n_xmatch, n_nondet]
    inner_labels = [f"Cross-matched\n{n_xmatch}", f"Non-detected\n{n_nondet}"]
    inner_colors = ["#455A64", "#E5E535"]

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("white")
ax.set_aspect("equal")

ax.pie(outer_vals, labels=outer_labels, colors=outer_colors, radius=1.0,
       startangle=90,
       wedgeprops=dict(width=0.30, edgecolor="white", linewidth=2),
       textprops=dict(fontsize=12, fontweight="bold"), labeldistance=1.12)

wedge_in, texts_in = ax.pie(
    inner_vals, labels=inner_labels, colors=inner_colors, radius=0.68,
    startangle=90,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1.5),
    textprops=dict(fontsize=10), labeldistance=0.52)
for t in texts_in:
    t.set_color("white"); t.set_fontweight("bold")

ax.text(0, 0, f"{n_total}\nPNe", ha="center", va="center",
        fontsize=16, fontweight="bold", color="#212121")

patches = [
    mpatches.Patch(color="#455A64", label=f"Cross-matched with MeerKAT  {n_xmatch}"),
]
if n_visual > 0:
    patches.append(mpatches.Patch(color="#FF9800",
        label=f"Visually confirmed     {n_visual}"))
patches.extend([
    mpatches.Patch(color="#E5E535", label=f"Non-detected           {n_nondet}"),
    mpatches.Patch(color="#EE1010", label=f"Outside coverage       {n_outside}"),
])
ax.legend(handles=patches, fontsize=9, framealpha=0.9,
          loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "step3c_meerkat_detection_breakdown.pdf"),
            format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    step3c_meerkat_detection_breakdown.pdf")

# step3c_multisurvey_detection_summary.pdf
print("    Generating detection summary figure...")

from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

cls_match = np.array(matched["reid_class"], dtype=str)
cls_vis   = np.array(visual["reid_class"], dtype=str)
cls_nd    = np.array(nondetected["reid_class"], dtype=str)
cls_out   = np.array(outside["reid_class"], dtype=str)

class_data = []
for cls in CLASSES + ["Unknown"]:
    n_tot = int((cls_all == cls).sum())
    n_xm  = int((cls_match == cls).sum())
    n_vi  = int((cls_vis == cls).sum())
    n_d   = n_xm + n_vi
    n_n   = int((cls_nd == cls).sum())
    n_o   = int((cls_out == cls).sum())
    frac  = n_d / n_tot * 100 if n_tot > 0 else 0
    class_data.append({"cls": cls, "total": n_tot, "xm": n_xm, "vis": n_vi,
                       "det": n_d, "nondet": n_n, "outside": n_o, "frac": frac})

fig = plt.figure(figsize=(12, 7.5), facecolor="white")
gs = gridspec.GridSpec(3, 1, height_ratios=[0.95, 0.95, 4], hspace=0.18)


def draw_detection_card_row(axis, heading, specifications):
    axis.set_xlim(0, 12); axis.set_ylim(0, 1); axis.axis("off")
    axis.text(0.10, 0.91, heading, ha="left", va="center", fontsize=8.7,
              fontweight="bold", color="#4B5563")
    for centre, (label, value, accent, fill_colour) in zip(
            [2.0, 6.0, 10.0], specifications):
        box = FancyBboxPatch(
            (centre - 1.70, 0.05), 3.40, 0.72,
            boxstyle="round,pad=0.07", facecolor=fill_colour,
            edgecolor="#D5DDE6", linewidth=0.8,
        )
        axis.add_patch(box)
        axis.text(centre, 0.48, str(value), ha="center", va="center",
                  fontsize=21, fontweight="bold", color=accent)
        axis.text(centre, 0.20, label, ha="center", va="center",
                  fontsize=8.8, fontweight="bold", color="#344054")


draw_detection_card_row(
    fig.add_subplot(gs[0]),
    "Survey counts (not mutually exclusive: common detections appear in both)",
    survey_total_specs,
)
draw_detection_card_row(
    fig.add_subplot(gs[1]),
    ("Exclusive detection groups "
     f"({n_mkt_only_map} + {n_both_map} + {n_askap_only_map} = "
     f"{n_mkt_only_map + n_both_map + n_askap_only_map})"),
    exclusive_specs,
)

# Bottom: horizontal stacked bar chart
ax = fig.add_subplot(gs[2])
ax.text(0.0, 1.055, "MeerKAT detection status by optical class",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2,
        fontweight="bold", color="#4B5563")

classes_plot = [d["cls"] for d in class_data]
y_pos = np.arange(len(classes_plot))[::-1]
bar_h = 0.55

for i, d in enumerate(class_data):
    y = y_pos[i]
    t = d["total"]
    if t == 0:
        continue
    # Stacked bars: cross-matched | visual | non-detected | outside
    w_xm  = d["xm"]  / t
    w_vis = d["vis"] / t
    w_nd  = d["nondet"] / t
    w_out = d["outside"] / t

    left = 0
    if d["xm"] > 0:
        ax.barh(y, w_xm, bar_h, left=left, color="#1976D2", edgecolor="white", lw=0.5)
        if w_xm > 0.06:
            ax.text(left + w_xm/2, y, str(d["xm"]), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += w_xm
    if d["vis"] > 0:
        ax.barh(y, w_vis, bar_h, left=left, color="#66BB6A", edgecolor="white", lw=0.5)
        if w_vis > 0.04:
            ax.text(left + w_vis/2, y, str(d["vis"]), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += w_vis
    if d["nondet"] > 0:
        ax.barh(y, w_nd, bar_h, left=left, color="#E0E0E0", edgecolor="white", lw=0.5)
        if w_nd > 0.06:
            ax.text(left + w_nd/2, y, str(d["nondet"]), ha="center", va="center",
                    fontsize=8, color="#616161")
        left += w_nd
    if d["outside"] > 0:
        ax.barh(y, w_out, bar_h, left=left, color="#BDBDBD", edgecolor="white", lw=0.5)
        if w_out > 0.04:
            ax.text(left + w_out/2, y, str(d["outside"]), ha="center", va="center",
                    fontsize=8, color="#424242")

    # Fraction label on right
    ax.text(1.03, y, f"{d['det']} / {d['total']} detected  ({d['frac']:.1f}%)",
            ha="left", va="center", fontsize=9.4, fontweight="bold",
            color="#1565C0" if d["frac"] > 20 else "#616161")

# Class labels on left with colour dots
for i, d in enumerate(class_data):
    y = y_pos[i]
    cls = d["cls"]
    color = COL.get(cls, "#BDBDBD")
    ax.plot(-0.03, y, marker="o" if cls != "Possible" else "D",
            color=color, markersize=8, clip_on=False,
            markeredgecolor=color, markeredgewidth=0)
    ax.text(-0.06, y, cls, ha="right", va="center",
            fontsize=11, fontweight="bold")

ax.set_xlim(0, 1); ax.set_ylim(-0.5, len(classes_plot) - 0.5)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color="#9E9E9E")
ax.set_yticks([])
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#E0E0E0")
ax.tick_params(axis="x", colors="#E0E0E0")
ax.grid(axis="x", color="#F0F0F0", lw=0.5, zorder=0)

# Two-column legend avoids long MeerKAT labels colliding.
legend_handles = [
    mpatches.Patch(color="#1976D2",
                   label=f"MeerKAT catalogue match ({n_xmatch})"),
    mpatches.Patch(color="#66BB6A",
                   label=f"MeerKAT visual detection ({n_visual})"),
    mpatches.Patch(color="#E0E0E0",
                   label=f"MeerKAT non-detected ({n_nondet})"),
    mpatches.Patch(color="#BDBDBD",
                   label=f"Outside MeerKAT coverage ({n_outside})"),
]
ax.legend(handles=legend_handles, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.13), frameon=False, fontsize=9,
          columnspacing=2.4, handlelength=1.5, handletextpad=0.6)

plt.savefig(os.path.join(FIGS, "step3c_multisurvey_detection_summary.pdf"),
            format="pdf", bbox_inches="tight", facecolor="white")
plt.close()
print(f"    step3c_multisurvey_detection_summary.pdf")

# Detection table by Reid class (printed, not plotted)
print("\n[4] Detection summary by Reid optical class:")
print(f"    {'Class':<12} {'Total':>6} {'XMatch':>7} {'Visual':>7} {'Det':>5} {'NonDet':>7} {'Outside':>8} {'Frac':>7}")
print(f"    {'-'*62}")

cls_match = np.array(matched["reid_class"], dtype=str)
cls_vis   = np.array(visual["reid_class"], dtype=str)
cls_nd    = np.array(nondetected["reid_class"], dtype=str)
cls_out   = np.array(outside["reid_class"], dtype=str)

for cls in CLASSES + ["Unknown"]:
    n_tot = int((cls_all == cls).sum())
    n_xm  = int((cls_match == cls).sum())
    n_vi  = int((cls_vis == cls).sum())
    n_d   = n_xm + n_vi
    n_n   = int((cls_nd == cls).sum())
    n_o   = int((cls_out == cls).sum())
    frac  = n_d / n_tot * 100 if n_tot > 0 else 0
    print(f"    {cls:<12} {n_tot:>6} {n_xm:>7} {n_vi:>7} {n_d:>5} {n_n:>7} {n_o:>8} {frac:>6.1f}%")

# Summary
print("\nstep3c complete")
print(f"  Total Reid PNe          : {n_total}")
print(f"  Detected (total)        : {n_det}")
print(f"    Cross-matched         : {n_xmatch}")
print(f"    Visually confirmed    : {n_visual}")
print(f"  Non-detected (inside)   : {n_nondet}")
print(f"  Outside MeerKAT region  : {n_outside}")
print(f"\n  Catalogues:")
print(f"    step3c_meerkat_detected.vot    ({n_det} rows)")
print(f"    step3c_meerkat_nondetected.vot ({n_nondet} rows)")
print(f"\n  Paper figures (05_Figures/):")
print(f"    step3c_multisurvey_sky_map_detections.pdf")
print(f"    step3c_multisurvey_sky_map_reid_classes.pdf")
print(f"    step3c_multisurvey_detection_summary.pdf")
print(f"    step3c_meerkat_positional_offsets.pdf")
print(f"    step3c_meerkat_snr_distribution.pdf")
print(f"    step3c_meerkat_detection_breakdown.pdf")
