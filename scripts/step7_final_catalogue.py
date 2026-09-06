"""
step7_final_catalogue.py
========================
Step 7 - the final radio catalogue of LMC planetary nebulae.

This is the deliverable table: one row per radio-detected PN, carrying the
optical identification, the positional agreement with each survey, and the
three radio criteria together with the classification they produce.

It is assembled, not computed.  Every value here was measured in an earlier
step; this step selects the columns a reader needs and gives them names that
mean something outside the pipeline.  Nothing is recalculated, so the final
catalogue cannot disagree with the figures in the paper.

Columns
-------
    RP_ID, Name, hash_id      identification
    RA, Dec                   optical position, degrees J2000
    reid_class                Reid & Parker optical classification
    hash_pnstat               HASH status, T (true) or P (probable)
    sep_meerkat_arcsec        optical-to-MeerKAT offset
    sep_askap_arcsec          optical-to-ASKAP offset
    S_meerkat_mJy             MeerKAT 1.295 GHz integrated flux density
    S_askap_mJy               ASKAP-EMU 888 MHz integrated flux density
    alpha, alpha_err          in-band spectral index and its 1-sigma error
    sp_class                  thermal / uncertain / steep / no_reliable_alpha
    mir_radio_ratio           adopted 8 micron to radio flux ratio
    below_flux_ceiling        True if below 2.2 mJy, False if above
    radio_class               Radio_High / Possible / Weak / Rejected

Empty cells are genuine absences: no ASKAP detection, no reliable spectral
index, no 8 micron photometry.  They are left blank rather than filled with a
sentinel so that a reader cannot mistake one for a measurement.

Inputs
------
    03_Outputs/step5d_criteria_tally.vot

Outputs
-------
    03_Outputs/Final_catalogue.vot     the catalogue
    03_Outputs/Final_catalogue_excerpt.tex  sampled rows, for the paper

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings

import numpy as np
from astropy.table import Table, Column

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

cfg = config.load()

IN_FILE = cfg.out("step5d_criteria_tally.vot")
OUT_VOT = cfg.out("Final_catalogue.vot")
OUT_TEX = cfg.table("Final_catalogue_excerpt.tex")

N_EXCERPT = 12   # rows in the published excerpt, sampled evenly in RA

print("=" * 62)
print("  Step 7  -  final radio catalogue")
print("=" * 62)

t = Table.read(IN_FILE)
n = len(t)
print(f"\nradio detections in: {n}")


def col(name, default=np.nan):
    """Return a column as a float array, or all-NaN if it is absent."""
    if name in t.colnames:
        return np.array(t[name], dtype=float)
    print(f"    note: {name} absent, column left empty")
    return np.full(n, default, dtype=float)


def scol(name, width=32):
    if name in t.colnames:
        return np.array([str(x).strip() for x in t[name]], dtype=f"U{width}")
    print(f"    note: {name} absent, column left empty")
    return np.full(n, "", dtype=f"U{width}")


# ------------------------------------------------------------------ positions
sep_mkt = col("mkt_separation_arcsec")
sep_ask = col("askap_separation_arcsec")

# ------------------------------------------------------------------- fluxes
# MeerKAT is stored in Jy, ASKAP in mJy.  Both are published here in mJy.
S_mkt = col("mkt_int_flux_Jy") * 1e3
S_ask = col("askap_int_flux_mJy")

# --------------------------------------------------------- spectral index
alpha = col("alpha")
alpha_err = col("alpha_err")
sp_class = scol("sp_class", 20)

# Only reliable indices are published.  An index that failed a quality cut is
# not a measurement of a flat spectrum, it is the absence of a measurement.
unreliable = (sp_class == "no_reliable_alpha") | ~np.isfinite(alpha)
alpha[unreliable] = np.nan
alpha_err[unreliable] = np.nan

# --------------------------------------------------------- MIR/radio ratio
mir = col("cohen_ratio_near1GHz")

# ------------------------------------------------------------ flux ceiling
# crit_ceiling: 1 below the ceiling, 0 above it, -1 not measurable.
ceil_code = col("crit_ceiling")
below_ceiling = np.full(n, "", dtype="U5")
below_ceiling[ceil_code == 1] = "True"
below_ceiling[ceil_code == 0] = "False"

# --------------------------------------------------------- classification
radio_class = scol("radio_only_class", 16)

# ================================================================== assemble
out = Table()
out["RP_ID"] = Column(scol("RP_ID", 16), description="Reid & Parker identifier")
out["Name"] = Column(scol("Name", 40), description="common name")
out["hash_id"] = Column(scol("hash_id", 16), description="HASH V/163 identifier")
out["RA"] = Column(col("RA"), unit="deg", format="%.6f",
                   description="optical right ascension, J2000")
out["Dec"] = Column(col("Dec"), unit="deg", format="%.6f",
                    description="optical declination, J2000")
out["reid_class"] = Column(scol("reid_classification", 12),
                           description="Reid & Parker optical classification")
out["hash_pnstat"] = Column(scol("hash_pnstat", 4),
                            description="HASH status: T true, P probable")
out["sep_meerkat_arcsec"] = Column(sep_mkt, unit="arcsec", format="%.2f",
                                   description="optical to MeerKAT offset")
out["sep_askap_arcsec"] = Column(sep_ask, unit="arcsec", format="%.2f",
                                 description="optical to ASKAP offset")
out["S_meerkat_mJy"] = Column(S_mkt, unit="mJy", format="%.4f",
                              description="MeerKAT 1.295 GHz integrated flux")
out["S_askap_mJy"] = Column(S_ask, unit="mJy", format="%.4f",
                            description="ASKAP-EMU 888 MHz integrated flux")
out["alpha"] = Column(alpha, format="%.3f",
                      description="in-band spectral index, reliable fits only")
out["alpha_err"] = Column(alpha_err, format="%.3f",
                          description="1-sigma uncertainty on alpha")
out["sp_class"] = Column(sp_class, description="thermal / uncertain / steep")
out["mir_radio_ratio"] = Column(mir, format="%.2f",
                                description="adopted 8 micron to radio flux ratio")
out["below_flux_ceiling"] = Column(below_ceiling,
                                   description="True if below the 2.2 mJy ceiling")
out["radio_class"] = Column(radio_class,
                            description="classification from the radio criteria alone")

# Sorted by right ascension, the usual ordering for a positional catalogue.
out = out[np.argsort(np.array(out["RA"], dtype=float))]

out.write(OUT_VOT, format="votable", overwrite=True)

# =================================================================== report
print(f"\ncolumns: {len(out.colnames)}   rows: {len(out)}")
print("\npopulated cells per column:")
for c in out.colnames:
    v = out[c]
    if v.dtype.kind in "fc":
        k = int(np.isfinite(np.array(v, dtype=float)).sum())
    else:
        k = int(np.sum([str(x).strip() != "" for x in v]))
    print(f"    {c:<22} {k:>4} / {len(out)}")

print("\nradio_class:")
for v in ("Radio_High", "Radio_Possible", "Radio_Weak", "Radio_Rejected"):
    print(f"    {v:<18} {int((out['radio_class'] == v).sum()):>4}")

# ============================================================ paper excerpt
# The excerpt carries identification, position, the optical class and our new
# columns.  The remaining Reid photometry stays in the full table.
cols = ["RP_ID", "RA", "Dec", "reid_class", "sep_meerkat_arcsec",
        "sep_askap_arcsec", "alpha", "alpha_err", "mir_radio_ratio",
        "below_flux_ceiling", "radio_class"]
head = ["ID", "RA", "Dec", "R14", r"$\rho_{\rm M}$", r"$\rho_{\rm A}$",
        r"$\alpha$", r"$\Delta\alpha$", "$R$", "Ceil.", "Radio class"]


def fmt(v, spec):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if not np.isfinite(f) else spec % f


with open(OUT_TEX, "w") as fh:
    fh.write("%% written by step7_final_catalogue.py -- do not edit by hand\n")
    fh.write("\\begin{tabular}{lrrlrrrrrcl}\n\\toprule\n")
    fh.write(" & ".join(head) + " \\\\\n")
    fh.write("\\midrule\n")
    # Rows sampled evenly through the RA-sorted catalogue rather than taken
    # from the top, so the excerpt shows the real spread of the columns --
    # objects with and without a spectral index, an ASKAP match, a ratio.
    pick = np.linspace(0, len(out) - 1, N_EXCERPT).astype(int)
    for r in out[pick]:
        cells = [
            str(r["RP_ID"]) if str(r["RP_ID"]).strip() else "--",
            fmt(r["RA"], "%.5f"), fmt(r["Dec"], "%.5f"),
            str(r["reid_class"]) if str(r["reid_class"]).strip() else "--",
            fmt(r["sep_meerkat_arcsec"], "%.2f"),
            fmt(r["sep_askap_arcsec"], "%.2f"),
            fmt(r["alpha"], "%.2f"), fmt(r["alpha_err"], "%.2f"),
            fmt(r["mir_radio_ratio"], "%.1f"),
            str(r["below_flux_ceiling"]) if str(r["below_flux_ceiling"]).strip() else "--",
            str(r["radio_class"]).replace("_", "\\_"),
        ]
        fh.write(" & ".join(cells) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

print(f"\n  -> {OUT_VOT}")
print(f"  -> {OUT_TEX}  ({N_EXCERPT} rows sampled through the table)")
print("\nstep7 complete")
