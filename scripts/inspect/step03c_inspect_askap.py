"""
step03c_inspect_askap.py
========================
Create multiwavelength visual-QA pages for every accepted and review-candidate
Reid x ASKAP association.  This step does not silently change match decisions;
it creates an editable VOTable queue with blank manual-decision fields.

Panels: ASKAP 888 MHz, MCELS H-alpha, MCELS [O III], SAGE IRAC 8 micron.
The Reid optical position is a cyan cross and the ASKAP catalogue position is
an orange plus.

Inputs
------
    03_Outputs/step02b_askap_matched.vot
    03_Outputs/step02b_askap_review.vot
    01_Data/ASKAP_LMC_888MHz_image.fits
    01_Data/LMC.ha.fits
    01_Data/LMC.oiii.fits
    01_Data/SAGE_LMC_IRAC8.0_2_mosaic.fits

Outputs
-------
    03_Outputs/step03c_askap_inspection_queue.vot
    04_Figures/Inspect/step03c_askap_accepted_cutouts.pdf
    04_Figures/Inspect/step03c_askap_review_cutouts.pdf

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import warnings

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Column, Table, vstack
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
INSP = os.path.join(BASE, "04_Figures", "Inspect")

ACCEPTED_FILE = os.path.join(OUTS, "step02b_askap_matched.vot")
REVIEW_FILE = os.path.join(OUTS, "step02b_askap_review.vot")
QUEUE_FILE = os.path.join(OUTS, "step03c_askap_inspection_queue.vot")

IMAGE_FILES = {
    "ASKAP 888 MHz": os.path.join(DATA, "ASKAP_LMC_888MHz_image.fits"),
    r"MCELS H$\alpha$": os.path.join(DATA, "LMC.ha.fits"),
    "MCELS [O III]": os.path.join(DATA, "LMC.oiii.fits"),
    r"SAGE 8 $\mu$m": os.path.join(DATA, "SAGE_LMC_IRAC8.0_2_mosaic.fits"),
}
CUTOUT_SIZE = 120 * u.arcsec

os.makedirs(OUTS, exist_ok=True)
os.makedirs(INSP, exist_ok=True)


def add_or_replace(table, column):
    if column.name in table.colnames:
        table.remove_column(column.name)
    table.add_column(column)


def image_plane_and_wcs(path):
    hdul = fits.open(path, memmap=True)
    data = hdul[0].data
    while data.ndim > 2:
        data = data[0]
    return hdul, data, WCS(hdul[0].header).celestial


def display_normalization(data):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return None
    return ImageNormalize(finite, interval=PercentileInterval(99.0), stretch=AsinhStretch())


def render_pdf(path, table, group, images):
    with PdfPages(path) as pdf:
        for row in table:
            optical = SkyCoord(float(row["RA"]) * u.deg, float(row["Dec"]) * u.deg)
            askap = SkyCoord(float(row["askap_ra"]) * u.deg,
                             float(row["askap_dec"]) * u.deg)
            fig = plt.figure(figsize=(11, 9))
            for panel, (title, (_, data, wcs)) in enumerate(images.items(), start=1):
                try:
                    cutout = Cutout2D(data, optical, CUTOUT_SIZE, wcs=wcs,
                                      mode="partial", fill_value=np.nan, copy=False)
                    ax = fig.add_subplot(2, 2, panel, projection=cutout.wcs)
                    norm = display_normalization(cutout.data)
                    ax.imshow(cutout.data, origin="lower", cmap="gray", norm=norm)
                    transform = ax.get_transform("world")
                    ax.scatter(optical.ra.deg, optical.dec.deg, transform=transform,
                               marker="x", s=90, linewidth=1.8, color="#00D4FF",
                               label="Reid optical")
                    ax.scatter(askap.ra.deg, askap.dec.deg, transform=transform,
                               marker="+", s=110, linewidth=1.8, color="#FF8C1A",
                               label="ASKAP catalogue")
                    ax.coords[0].set_axislabel("RA")
                    ax.coords[1].set_axislabel("Dec")
                    ax.legend(fontsize=7, loc="upper right")
                except Exception as exc:
                    ax = fig.add_subplot(2, 2, panel)
                    ax.text(0.5, 0.5, f"Cutout unavailable\n{exc}", ha="center", va="center")
                    ax.set_axis_off()
                ax.set_title(title)

            rp_id = str(row["RP_ID"]).strip()
            reid_class = str(row["reid_class"]).strip()
            separation = float(row["askap_separation_arcsec"])
            source_list = str(row["askap_source_list"]).strip()
            source_type = str(row["askap_source_type"]).strip()
            flux = float(row["askap_int_flux_mJy"])
            fig.suptitle(
                f"{rp_id} | Reid {reid_class} | {group} | sep={separation:.2f} arcsec | "
                f"{source_list}/{source_type} | S888={flux:.3f} mJy",
                fontsize=12,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=140, bbox_inches="tight")
            plt.close(fig)


print("step03c: building the ASKAP inspection pages")

accepted = Table.read(ACCEPTED_FILE, format="votable")
review = Table.read(REVIEW_FILE, format="votable")
add_or_replace(accepted, Column(np.full(len(accepted), "accepted", dtype="U12"),
                                name="inspection_group"))
add_or_replace(review, Column(np.full(len(review), "review", dtype="U12"),
                              name="inspection_group"))
queue = vstack([accepted, review], metadata_conflicts="silent")
add_or_replace(queue, Column(np.full(len(queue), "", dtype="U16"), name="manual_decision"))
add_or_replace(queue, Column(np.full(len(queue), "", dtype="U256"), name="manual_notes"))
queue.write(QUEUE_FILE, format="votable", overwrite=True)
print(f"\n[1] Inspection queue: {len(queue)} rows -> {QUEUE_FILE}")

print("\n[2] Opening survey images with memory mapping...")
images = {title: image_plane_and_wcs(path) for title, path in IMAGE_FILES.items()}

accepted_pdf = os.path.join(INSP, "step03c_askap_accepted_cutouts.pdf")
review_pdf = os.path.join(INSP, "step03c_askap_review_cutouts.pdf")
print("\n[3] Rendering accepted associations...")
render_pdf(accepted_pdf, accepted, "accepted", images)
print(f"    {accepted_pdf}")
print("\n[4] Rendering borderline/wide candidates...")
render_pdf(review_pdf, review, "review", images)
print(f"    {review_pdf}")

for hdul, _, _ in images.values():
    hdul.close()

print("\nstep03c complete")
print("    Fill manual_decision/manual_notes only after inspecting the PDFs.")
