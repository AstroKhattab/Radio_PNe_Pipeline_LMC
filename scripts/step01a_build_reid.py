"""
step01a_build_reid.py
=====================
Build the single unified Reid LMC PN catalogue used by all downstream scripts.

Merges:
    Warren_PNE_673.vot      — 673 PNe; positions (RAx, DECx) +
                              classification (True/Known/Likely/Possible)
    reid2014_photometry.vot — 605 PNe; SIMBAD positions (_RA, _DE) +
                              multiwavelength photometry (optical/NIR/MIR)

Merge strategy
--------------
1. Warren is the primary base (673 sources — has all classified PNe).
2. Each Warren source is matched to reid2014_photometry at <= 5 arcsec.
   Matched sources get photometry columns appended.
3. Unmatched Warren sources keep positions/classification; phot = NaN.
4. reid2014 sources with no Warren counterpart are appended (class=Unknown).
5. Near-duplicate positions (< 1 arcsec) are deduplicated (keep Warren).

Output columns
--------------
    RA, Dec         — Warren positions (deg, J2000); SIMBAD fallback for extras
    RA_simbad       — SIMBAD RA from reid2014 (NaN if no match)
    Dec_simbad      — SIMBAD Dec from reid2014 (NaN if no match)
    RP_ID           — Reid & Parker reference ID (RPRef from Warren)
    Name            — Common name from reid2014 (blank if no match)
    reid_class      — True / Known / Likely / Possible / Unknown
    has_phot        — 1 if reid2014 photometry available, 0 otherwise
    [phot columns]  — All reid2014 magnitude columns with uncertainties

Output
------
    03_Outputs/step01a_reid_master.vot

Run before Step 02a and Step 02b.

O. K. Khattab & M. D. Filipovic, Western Sydney University.
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
WARREN_FILE = os.path.join(DATA, "Warren_PNE_673.vot")
REID_FILE   = os.path.join(DATA, "reid2014_photometry.vot")
OUT_FILE    = os.path.join(OUTS, "step01a_reid_master.vot")
MERGE_RAD   = 5.0  # arcsec
DEDUP_RAD   = 1.0  # arcsec

print("step01a: building the unified Reid PN catalogue")

# Load
print("\n[1] Loading catalogues...")
warren = Table.read(WARREN_FILE, format="votable")
reid14 = Table.read(REID_FILE,   format="votable")
print(f"    Warren_PNE_673      : {len(warren)} sources")
print(f"    reid2014_photometry : {len(reid14)} sources")

ra_war  = np.array(warren["RAx"],  dtype=float)
dec_war = np.array(warren["DECx"], dtype=float)
ra_r14  = np.array(reid14["_RA"],  dtype=float)
dec_r14 = np.array(reid14["_DE"],  dtype=float)

valid_war = np.isfinite(ra_war)  & np.isfinite(dec_war)
valid_r14 = np.isfinite(ra_r14)  & np.isfinite(dec_r14)

coords_war = SkyCoord(ra=ra_war[valid_war]  * u.deg,
                      dec=dec_war[valid_war] * u.deg)
coords_r14 = SkyCoord(ra=ra_r14[valid_r14]  * u.deg,
                      dec=dec_r14[valid_r14] * u.deg)

# Match reid2014 → Warren
print(f"\n[2] Matching reid2014 → Warren at <= {MERGE_RAD}\"...")
idx_r, sep_r, _ = coords_r14.match_to_catalog_sky(coords_war)
sep_arcsec       = sep_r.to(u.arcsec).value

war_valid_idx = np.where(valid_war)[0]
r14_valid_idx = np.where(valid_r14)[0]

# war_row → (r14_row, separation)
war_to_r14 = {}
for j, (widx, sep) in enumerate(zip(idx_r, sep_arcsec)):
    if sep <= MERGE_RAD:
        wrow = war_valid_idx[widx]
        rrow = r14_valid_idx[j]
        if wrow not in war_to_r14 or sep < war_to_r14[wrow][1]:
            war_to_r14[wrow] = (rrow, sep)

print(f"    Warren with phot match : {len(war_to_r14)}")
print(f"    Warren without match   : {len(warren) - len(war_to_r14)}")

matched_r14 = {v[0] for v in war_to_r14.values()}
extra_r14   = [ri for ri in r14_valid_idx if ri not in matched_r14]
print(f"    reid2014 extras (→ Unknown) : {len(extra_r14)}")

# Photometry columns
SKIP  = {"recno", "Name", "_RA", "_DE", "SimbadName"}
PHOT_COLS = [c for c in reid14.colnames if c not in SKIP]

def get_phot(ri, col):
    try:
        val = reid14[col][ri]
        if reid14[col].dtype.kind in ("f", "i"):
            return float(val)
        return str(val)
    except Exception:
        return np.nan

def nan_phot(col):
    if reid14[col].dtype.kind in ("f", "i"):
        return np.nan
    return ""

# Build rows
print("\n[3] Building unified table...")

RA=[]; Dec=[]; RA_s=[]; Dec_s=[]
RPID=[]; NAME=[]; CLS=[]; HAS_PHOT=[]
phot = {c: [] for c in PHOT_COLS}

# Part A — all Warren sources
for wi in range(len(warren)):
    RA.append(float(ra_war[wi])  if valid_war[wi] else np.nan)
    Dec.append(float(dec_war[wi]) if valid_war[wi] else np.nan)
    RPID.append(str(warren["RPRef"][wi]))
    CLS.append(str(warren["OBJECT_PROBABILITY"][wi]))
    if wi in war_to_r14:
        ri, _ = war_to_r14[wi]
        RA_s.append(float(ra_r14[ri])  if valid_r14[ri] else np.nan)
        Dec_s.append(float(dec_r14[ri]) if valid_r14[ri] else np.nan)
        NAME.append(str(reid14["Name"][ri]))
        HAS_PHOT.append(1)
        for c in PHOT_COLS:
            phot[c].append(get_phot(ri, c))
    else:
        RA_s.append(np.nan); Dec_s.append(np.nan)
        NAME.append(""); HAS_PHOT.append(0)
        for c in PHOT_COLS:
            phot[c].append(nan_phot(c))

# Part B — reid2014 sources not in Warren
for ri in extra_r14:
    RA.append(float(ra_r14[ri]));  Dec.append(float(dec_r14[ri]))
    RA_s.append(float(ra_r14[ri])); Dec_s.append(float(dec_r14[ri]))
    RPID.append(""); NAME.append(str(reid14["Name"][ri]))
    CLS.append("Unknown"); HAS_PHOT.append(1)
    for c in PHOT_COLS:
        phot[c].append(get_phot(ri, c))

n_raw = len(RA)

# Deduplicate
print(f"[4] Deduplicating (< {DEDUP_RAD}\")...")
valid_out = np.isfinite(RA) & np.isfinite(Dec)
coords_all = SkyCoord(ra=np.array(RA)[valid_out]  * u.deg,
                      dec=np.array(Dec)[valid_out] * u.deg)
all_valid_idx = np.where(valid_out)[0]
keep = np.ones(n_raw, dtype=bool)
for i in range(len(coords_all)):
    if not keep[all_valid_idx[i]]:
        continue
    seps = coords_all[i].separation(coords_all).to(u.arcsec).value
    dups = np.where((seps < DEDUP_RAD) & (np.arange(len(coords_all)) != i))[0]
    for d in dups:
        keep[all_valid_idx[d]] = False

keep_idx = np.where(keep)[0]
n_final  = int(keep.sum())
print(f"    Before dedup : {n_raw}")
print(f"    After  dedup : {n_final}")

# Assemble final table
out = Table()
def arr(lst, dtype=float): return np.array(lst, dtype=dtype)[keep_idx]

out.add_column(Column(arr(RA),       name="RA",        unit="deg"))
out.add_column(Column(arr(Dec),      name="Dec",       unit="deg"))
out.add_column(Column(arr(RA_s),     name="RA_simbad", unit="deg"))
out.add_column(Column(arr(Dec_s),    name="Dec_simbad",unit="deg"))
out.add_column(Column(arr(RPID,   "U20"), name="RP_ID"))
out.add_column(Column(arr(NAME,   "U30"), name="Name"))
out.add_column(Column(arr(CLS,    "U10"), name="reid_class"))
out.add_column(Column(arr(HAS_PHOT, np.int16), name="has_phot"))

for c in PHOT_COLS:
    col_dtype = reid14[c].dtype
    if col_dtype.kind in ("f", "i"):
        out.add_column(Column(arr(phot[c]), name=c))
    else:
        out.add_column(Column(arr(phot[c], "U10"), name=c))

# Classification summary
print("\n    Classification breakdown:")
for cls in ["True", "Known", "Likely", "Possible", "Unknown"]:
    n = int(np.sum(np.array(out["reid_class"]) == cls))
    print(f"      {cls:<10}: {n}")

# Save
os.makedirs(OUTS, exist_ok=True)
out.write(OUT_FILE, format="votable", overwrite=True)

print("\nstep01a complete")
print(f"  Output : {os.path.basename(OUT_FILE)}")
print(f"  Sources: {n_final}")
print(f"  Path   : {OUT_FILE}")
