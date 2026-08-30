"""
step1_parent_catalogue.py
=========================
Build the single unified LMC PN catalogue used by all downstream scripts.

The parent sample now comes from the HASH PN database (Parker, Bojicic & Frew
2016), release V/163, instead of being assembled by hand from Warren's file.
Taking it from HASH means the selection is reproducible from a public
catalogue and the classifications are the ones the community curates.

Sources
-------
    HASH_V163_full.vot      — 11,760 entries; the LMC domain holds 845, of
                              which 707 are classified as a PN (PNstat T or P)
    Warren_PNE_673.vot      — 673 PNe; supplies the Reid & Parker running
                              number (RPRef) and the optical classification
                              (True/Known/Likely/Possible)
    reid2014_photometry.vot — 605 PNe; SIMBAD positions (_RA, _DE) +
                              multiwavelength photometry (optical/NIR/MIR)

Build strategy
--------------
1. The parent sample is every HASH LMC entry with PNstat in {T, P}: 707.
   Entries carrying any other status (star clusters, H II regions, supernova
   remnants and so on) are written out separately so the paper can tabulate
   what was set aside and why.
2. Each parent source is matched to Warren at <= 4.5 arcsec, which attaches
   the RP number and the Reid & Parker class where one exists.
3. Each parent source is matched to reid2014_photometry at <= 4.5 arcsec,
   which attaches the photometry columns.
4. Sources with no Warren counterpart keep a HASH-derived identifier, and
   their class is mapped from PNstat: T -> True, P -> Possible.

No source is added or removed by steps 2 to 4; they only attach columns.
The deduplication that the previous version needed is gone with it — HASH
entries are already unique.

Output columns (unchanged from the previous version, so that every
downstream script continues to work without modification)
--------------
    RA, Dec         — HASH positions (deg, J2000)
    RA_simbad       — SIMBAD RA from reid2014 (NaN if no match)
    Dec_simbad      — SIMBAD Dec from reid2014 (NaN if no match)
    RP_ID           — Reid & Parker reference ID; HASH<id> where there is none
    Name            — Common name (reid2014, else HASH)
    reid_class      — True / Known / Likely / Possible / Unknown
    has_phot        — 1 if reid2014 photometry available, 0 otherwise
    [phot columns]  — All reid2014 magnitude columns with uncertainties

Added columns (new, ignored by the older scripts)
--------------
    hash_id, hash_pnstat, hash_name, hash_catalogue, opt_majdiam

Output
------
    03_Outputs/step1_parent_catalogue.vot   the parent sample
    03_Outputs/step1_lmc_nonpn.vot     LMC entries rejected by PNstat

Run before Step 02a and Step 02b.

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os, warnings
import numpy as np
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

BASE        = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA        = os.path.join(BASE, "01_Data")
OUTS        = os.path.join(BASE, "03_Outputs")
HASH_FILE   = os.path.join(DATA, "HASH_V163_full.vot")
WARREN_FILE = os.path.join(DATA, "Warren_PNE_673.vot")
REID_FILE   = os.path.join(DATA, "reid2014_photometry.vot")
OUT_FILE    = os.path.join(OUTS, "step1_parent_catalogue.vot")
NONPN_FILE  = os.path.join(OUTS, "step1_lmc_nonpn.vot")
SAMPLE_CSV  = os.path.join(OUTS, "step1_catalogue_excerpt.csv")
SAMPLE_TEX  = os.path.join(OUTS, "step1_catalogue_excerpt.tex")
N_EXCERPT   = 10   # rows shown in the published excerpt

PN_STATUS   = ("T", "P")   # HASH: true PN, probable PN
MATCH_RAD   = 4.5          # arcsec — same acceptance radius as the radio matching

# PNstat -> reid_class, for the HASH sources that Warren never listed.
STATUS_TO_CLASS = {"T": "True", "P": "Possible"}

# Human-readable expansions of the HASH PNstat codes we reject.
STATUS_LABELS = {
    "T": "true PN", "P": "probable PN", "L": "likely PN",
    "Cl*": "star cluster", "HII": "H II region", "SNR": "supernova remnant",
    "CV*": "cataclysmic variable", "Sy*": "symbiotic star",
    "Sy?": "symbiotic candidate", "SR?": "supernova remnant candidate",
    "Y*O": "young stellar object", "star": "star", "s": "star",
    "c": "compact object", "a": "artefact", "cir": "circumstellar shell",
    "ooun": "unknown",
}

print("=" * 56)
print("  step1_build_reid.py")
print("=" * 56)


def clean(col):
    """HASH string columns arrive padded and sometimes masked."""
    return np.array([str(x).strip() for x in np.asarray(col)])


# ── Load ─────────────────────────────────────────────────────────────────────
print("\n[1] Loading catalogues...")
hash_all = Table.read(HASH_FILE,   format="votable")
warren   = Table.read(WARREN_FILE, format="votable")
reid14   = Table.read(REID_FILE,   format="votable")
print(f"    HASH V/163          : {len(hash_all)} entries")
print(f"    Warren_PNE_673      : {len(warren)} sources")
print(f"    reid2014_photometry : {len(reid14)} sources")

# ── Select the parent sample ─────────────────────────────────────────────────
print("\n[2] Selecting the LMC PN parent sample from HASH...")
lmc    = hash_all[clean(hash_all["Domain"]) == "LMC"]
pnstat = clean(lmc["PNstat"])
is_pn  = np.isin(pnstat, PN_STATUS)
parent = lmc[is_pn]
nonpn  = lmc[~is_pn]

print(f"    LMC domain              : {len(lmc)}")
print(f"    PNstat T (true PN)      : {int(np.sum(pnstat == 'T'))}")
print(f"    PNstat P (probable PN)  : {int(np.sum(pnstat == 'P'))}")
print(f"    Parent sample           : {len(parent)}")
print(f"    Set aside (not a PN)    : {len(nonpn)}")

print("\n    LMC entries excluded from the parent sample:")
codes, counts = np.unique(clean(nonpn["PNstat"]), return_counts=True)
for code, n in sorted(zip(codes, counts), key=lambda t: -t[1]):
    print(f"      {code:<6} {STATUS_LABELS.get(code, 'other'):<26} {n:>4}")

n_parent = len(parent)
ra_par  = np.asarray(parent["RAJ2000"], dtype=float)
dec_par = np.asarray(parent["DEJ2000"], dtype=float)
coords_par = SkyCoord(ra=ra_par * u.deg, dec=dec_par * u.deg)

# ── Match to Warren for the RP number and the optical class ──────────────────
print(f"\n[3] Matching parent -> Warren at <= {MATCH_RAD}\"...")
ra_war  = np.asarray(warren["RAx"],  dtype=float)
dec_war = np.asarray(warren["DECx"], dtype=float)
valid_war = np.isfinite(ra_war) & np.isfinite(dec_war)
coords_war = SkyCoord(ra=ra_war[valid_war] * u.deg,
                      dec=dec_war[valid_war] * u.deg)
war_rows = np.where(valid_war)[0]

idx_w, sep_w, _ = coords_par.match_to_catalog_sky(coords_war)
hit_w = sep_w.arcsec <= MATCH_RAD

rp_id = np.full(n_parent, "", dtype="U20")
cls   = np.full(n_parent, "", dtype="U10")
rp_id[hit_w] = clean(warren["RPRef"])[war_rows[idx_w[hit_w]]]
cls[hit_w]   = clean(warren["OBJECT_PROBABILITY"])[war_rows[idx_w[hit_w]]]

# HASH sources Warren never listed: give them a HASH-derived identifier and
# map the class from PNstat so that the downstream class-based summaries still
# have something meaningful to report.
hash_ids = np.asarray(parent["HASH"], dtype=int)
for i in np.where(~hit_w)[0]:
    rp_id[i] = f"HASH{hash_ids[i]}"
    cls[i]   = STATUS_TO_CLASS.get(pnstat[is_pn][i], "Unknown")

print(f"    With a Reid & Parker RP number : {int(hit_w.sum())}")
print(f"    HASH-only (class from PNstat)  : {int((~hit_w).sum())}")
assert len(set(rp_id)) == n_parent, "RP_ID is not unique"

# ── Match to reid2014 for the photometry ─────────────────────────────────────
print(f"\n[4] Matching parent -> reid2014 at <= {MATCH_RAD}\"...")
ra_r14  = np.asarray(reid14["_RA"], dtype=float)
dec_r14 = np.asarray(reid14["_DE"], dtype=float)
valid_r14 = np.isfinite(ra_r14) & np.isfinite(dec_r14)
coords_r14 = SkyCoord(ra=ra_r14[valid_r14] * u.deg,
                      dec=dec_r14[valid_r14] * u.deg)
r14_rows = np.where(valid_r14)[0]

idx_r, sep_r, _ = coords_par.match_to_catalog_sky(coords_r14)
hit_r = sep_r.arcsec <= MATCH_RAD
src_r = np.full(n_parent, -1, dtype=int)
src_r[hit_r] = r14_rows[idx_r[hit_r]]
print(f"    With reid2014 photometry : {int(hit_r.sum())}")

SKIP = {"recno", "Name", "_RA", "_DE", "SimbadName"}
PHOT_COLS = [c for c in reid14.colnames if c not in SKIP]


def phot_column(name):
    """Pull one reid2014 column onto the parent rows, NaN/blank where absent."""
    src = reid14[name]
    if src.dtype.kind in ("f", "i"):
        out = np.full(n_parent, np.nan)
        for i in np.where(hit_r)[0]:
            try:
                out[i] = float(src[src_r[i]])
            except Exception:
                pass
        return Column(out, name=name)
    out = np.full(n_parent, "", dtype="U10")
    for i in np.where(hit_r)[0]:
        try:
            out[i] = str(src[src_r[i]])
        except Exception:
            pass
    return Column(out, name=name)


# ── Assemble the table ───────────────────────────────────────────────────────
print("\n[5] Building unified table...")
ra_s  = np.full(n_parent, np.nan)
dec_s = np.full(n_parent, np.nan)
ra_s[hit_r]  = ra_r14[src_r[hit_r]]
dec_s[hit_r] = dec_r14[src_r[hit_r]]

name = clean(parent["Name"]).astype("U30")
r14_name = clean(reid14["Name"])
for i in np.where(hit_r)[0]:
    if r14_name[src_r[i]]:
        name[i] = r14_name[src_r[i]]

out = Table()
out.add_column(Column(ra_par,  name="RA",         unit="deg"))
out.add_column(Column(dec_par, name="Dec",        unit="deg"))
out.add_column(Column(ra_s,    name="RA_simbad",  unit="deg"))
out.add_column(Column(dec_s,   name="Dec_simbad", unit="deg"))
out.add_column(Column(rp_id,   name="RP_ID"))
out.add_column(Column(name,    name="Name"))
out.add_column(Column(cls,     name="reid_class"))
out.add_column(Column(hit_r.astype(np.int16), name="has_phot"))

for c in PHOT_COLS:
    out.add_column(phot_column(c))

# Provenance columns.  New, and ignored by every script written before them.
maj = np.asarray(parent["MajDiam"], dtype=float)
maj[~np.isfinite(maj)] = np.nan
out.add_column(Column(hash_ids.astype(np.int32),                name="hash_id"))
out.add_column(Column(pnstat[is_pn].astype("U4"),               name="hash_pnstat"))
out.add_column(Column(clean(parent["Name"]).astype("U32"),      name="hash_name"))
out.add_column(Column(clean(parent["Catalogue"]).astype("U24"), name="hash_catalogue"))
out.add_column(Column(maj, name="opt_majdiam", unit="arcsec"))

# ── Classification summary ───────────────────────────────────────────────────
print("\n    Classification breakdown:")
for c in ["True", "Known", "Likely", "Possible", "Unknown"]:
    print(f"      {c:<10}: {int(np.sum(np.asarray(out['reid_class']) == c))}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTS, exist_ok=True)
out.write(OUT_FILE, format="votable", overwrite=True)

rejected = Table()
rejected.add_column(Column(np.asarray(nonpn["HASH"], dtype=np.int32), name="hash_id"))
rejected.add_column(Column(np.asarray(nonpn["RAJ2000"], dtype=float), name="RA",  unit="deg"))
rejected.add_column(Column(np.asarray(nonpn["DEJ2000"], dtype=float), name="Dec", unit="deg"))
rejected.add_column(Column(clean(nonpn["PNstat"]).astype("U4"),       name="hash_pnstat"))
rejected.add_column(Column(np.array([STATUS_LABELS.get(c, "other")
                                     for c in clean(nonpn["PNstat"])], dtype="U28"),
                           name="object_type"))
rejected.add_column(Column(clean(nonpn["Name"]).astype("U32"),        name="hash_name"))
rejected.add_column(Column(clean(nonpn["Catalogue"]).astype("U24"),   name="hash_catalogue"))
rejected.write(NONPN_FILE, format="votable", overwrite=True)

# ── Published excerpt ────────────────────────────────────────────────────────
# The full catalogue goes to CDS; the paper carries a short excerpt so the
# reader can see the columns and their format without downloading anything.
# These are the columns every later step actually uses.
print("\n[6] Writing the catalogue excerpt for the paper...")
EXCERPT_COLS = [
    ("RP_ID",       "ID",            "Reid & Parker number, or HASH running number"),
    ("hash_pnstat", "PNstat",        "HASH classification: T true, P probable"),
    ("Name",        "Name",          "Common name"),
    ("RA",          "RAJ2000",       "deg"),
    ("Dec",         "DEJ2000",       "deg"),
    ("reid_class",  "Class",         "Reid & Parker optical class"),
    ("opt_majdiam", "Diam",          "Optical major diameter, arcsec"),
    ("__8.0_",      "[8.0]",         "IRAC 8.0 um magnitude"),
    ("has_phot",    "Phot",          "1 if Reid 2014 photometry available"),
]
present = [(c, h, d) for c, h, d in EXCERPT_COLS if c in out.colnames]

excerpt = out[[c for c, _, _ in present]][:N_EXCERPT].copy()
for c, h, _ in present:
    excerpt.rename_column(c, h)
excerpt.write(SAMPLE_CSV, format="ascii.csv", overwrite=True)

def cell(value):
    """Format one table cell for LaTeX, blanking anything unmeasured."""
    if isinstance(value, (float, np.floating)):
        return "--" if not np.isfinite(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    text = str(value).strip()
    return text if text else "--"

with open(SAMPLE_TEX, "w") as fh:
    fh.write("% Excerpt of the parent catalogue; full table available at CDS.\n")
    fh.write("\\begin{table*}\n\\centering\n")
    fh.write("\\caption{Excerpt from the LMC planetary nebula parent catalogue. "
             f"The full table of {n_parent} entries is available electronically.}}\n")
    fh.write("\\label{tab:parent_catalogue}\n")
    fh.write("\\begin{tabular}{" + "l" * len(present) + "}\n\\hline\n")
    fh.write(" & ".join(h for _, h, _ in present) + " \\\\\n\\hline\n")
    for row in excerpt:
        fh.write(" & ".join(cell(row[h]) for _, h, _ in present) + " \\\\\n")
    fh.write("\\hline\n\\end{tabular}\n\\end{table*}\n")

print(f"    columns: {', '.join(h for _, h, _ in present)}")
print(f"    -> {os.path.basename(SAMPLE_CSV)}")
print(f"    -> {os.path.basename(SAMPLE_TEX)}  ({N_EXCERPT} rows)")

print(f"\n{'='*56}")
print(f"  STEP 1 COMPLETE")
print(f"{'='*56}")
print(f"  Output : {os.path.basename(OUT_FILE)}")
print(f"  Sources: {n_parent}")
print(f"  Also   : {os.path.basename(NONPN_FILE)}  ({len(rejected)} non-PN)")
print(f"  Path   : {OUT_FILE}")
print(f"{'='*56}")
