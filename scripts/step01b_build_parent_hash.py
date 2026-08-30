"""
step01b_build_parent_hash.py
============================
Step 1 - Parent sample.

Defines the optical parent sample of LMC planetary nebulae that every later
step works from.  The sample is taken directly from the HASH PN database
(Parker, Bojicic & Frew 2016), release V/163, rather than assembled by hand
from individual literature lists.  Two things follow from that choice:

  * the selection is reproducible - anyone can pull the same rows from CDS;
  * the object classifications are the ones HASH curates, so we inherit the
    community's judgement on what is and is not a PN instead of inventing
    our own.

Selection
---------
    Domain == 'LMC'                   845 entries
    PNstat in {'T', 'P'}              -> parent sample
        T  true PN
        P  probable PN
Everything else in the LMC domain (compact star clusters, H II regions,
supernova remnants, symbiotics and so on) is written to a separate file and
tabulated in the paper, so the reader can see what was set aside and why.

Ancillary data
--------------
Two older catalogues are cross-matched in purely to carry extra columns; they
do not add or remove sources:

    Warren_PNE_673.vot      Reid & Parker optical classification
                            (True / Known / Likely / Possible) and the RP
                            running number used throughout the literature.
    reid2014_photometry.vot Reid & Parker (2014) multiwavelength photometry;
                            we only need the IRAC 8.0 um magnitude, which
                            feeds the mid-infrared to radio ratio in Step 4b.

Outputs
-------
    03_Outputs/step01b_parent_sample.vot     the parent PNe
    03_Outputs/step01b_lmc_nonpn.vot         LMC entries rejected by PNstat
    03_Outputs/step01b_parent_summary.csv    counts behind the paper table
    04_Figures/step01b_parent_sample.pdf     sky map + provenance breakdown

Authors : Omar K. Khattab, M. D. Filipovic
Group   : Filipovic Group, Western Sydney University
"""

import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore")

BASE = os.path.expanduser("~/Desktop/Research/PN LMC Paper")
DATA = os.path.join(BASE, "01_Data")
OUTS = os.path.join(BASE, "03_Outputs")
FIGS = os.path.join(BASE, "04_Figures")

HASH_FILE   = os.path.join(DATA, "HASH_V163_full.vot")
WARREN_FILE = os.path.join(DATA, "Warren_PNE_673.vot")
REID_FILE   = os.path.join(DATA, "reid2014_photometry.vot")

OUT_PARENT  = os.path.join(OUTS, "step01b_parent_sample.vot")
OUT_NONPN   = os.path.join(OUTS, "step01b_lmc_nonpn.vot")
OUT_SUMMARY = os.path.join(OUTS, "step01b_parent_summary.csv")
OUT_FIG     = os.path.join(FIGS, "step01b_parent_sample.pdf")

PN_STATUS  = ("T", "P")   # true and probable PN
MATCH_RAD  = 4.5          # arcsec, same acceptance radius as the radio matching

# Human-readable expansions of the HASH PNstat codes we reject.  Anything not
# listed here is printed as the raw code.
STATUS_LABELS = {
    "T": "true PN", "P": "probable PN", "L": "likely PN",
    "Cl*": "star cluster", "HII": "H II region", "SNR": "supernova remnant",
    "CV*": "cataclysmic variable", "Sy*": "symbiotic star",
    "Sy?": "symbiotic candidate", "SR?": "supernova remnant candidate",
    "Y*O": "young stellar object", "star": "star", "s": "star",
    "c": "compact object", "a": "artefact", "cir": "circumstellar shell",
    "ooun": "unknown", "L": "likely PN",
}


def clean(col):
    """HASH string columns arrive padded and sometimes masked."""
    return np.array([str(x).strip() for x in np.asarray(col)])


print("=" * 60)
print("  Step 1  -  parent sample from HASH V/163")
print("=" * 60)

# ---------------------------------------------------------------- load HASH
hash_all = Table.read(HASH_FILE, format="votable")
print(f"\nHASH V/163 total entries : {len(hash_all)}")

domain = clean(hash_all["Domain"])
lmc = hash_all[domain == "LMC"]
print(f"LMC domain               : {len(lmc)}")

pnstat = clean(lmc["PNstat"])
is_pn = np.isin(pnstat, PN_STATUS)

parent = lmc[is_pn]
nonpn = lmc[~is_pn]
print(f"parent  (PNstat T or P)  : {len(parent)}")
print(f"set aside                : {len(nonpn)}")

# ------------------------------------------------- what exactly was set aside
print("\nLMC entries excluded from the parent sample:")
codes, counts = np.unique(clean(nonpn["PNstat"]), return_counts=True)
order = np.argsort(-counts)
for code, n in zip(codes[order], counts[order]):
    print(f"    {code:<6} {STATUS_LABELS.get(code, 'other'):<26} {n:>4}")

# ------------------------------------------------------------- build the table
out = Table()
out["HASH_ID"] = Column(np.asarray(parent["HASH"], dtype=np.int32))
out["PNG"] = Column(clean(parent["PNG"]).astype("U12"))
out["RA"] = Column(np.asarray(parent["RAJ2000"], dtype=float), unit="deg")
out["Dec"] = Column(np.asarray(parent["DEJ2000"], dtype=float), unit="deg")
out["PNstat"] = Column(clean(parent["PNstat"]).astype("U4"))
out["hash_name"] = Column(clean(parent["Name"]).astype("U32"))
out["simbad_id"] = Column(clean(parent["SimbadID"]).astype("U36"))
out["morph"] = Column(clean(parent["MorphMainCl"]).astype("U4"))
out["hash_catalogue"] = Column(clean(parent["Catalogue"]).astype("U24"))

maj = np.asarray(parent["MajDiam"], dtype=float)
maj[~np.isfinite(maj)] = np.nan
out["opt_majdiam"] = Column(maj, unit="arcsec")

coords_parent = SkyCoord(ra=np.asarray(out["RA"]) * u.deg,
                         dec=np.asarray(out["Dec"]) * u.deg)

# ------------------------------------------------------ Warren: RP id + class
warren = Table.read(WARREN_FILE, format="votable")
ra_w = np.asarray(warren["RAx"], dtype=float)
dec_w = np.asarray(warren["DECx"], dtype=float)
good_w = np.isfinite(ra_w) & np.isfinite(dec_w)
coords_w = SkyCoord(ra=ra_w[good_w] * u.deg, dec=dec_w[good_w] * u.deg)
w_rows = np.where(good_w)[0]

idx, sep, _ = coords_parent.match_to_catalog_sky(coords_w)
hit = sep.arcsec <= MATCH_RAD

rp_id = np.full(len(out), "", dtype="U20")
reid_class = np.full(len(out), "not in RP", dtype="U12")
rp_id[hit] = clean(warren["RPRef"])[w_rows[idx[hit]]]
reid_class[hit] = clean(warren["OBJECT_PROBABILITY"])[w_rows[idx[hit]]]

out["RP_ID"] = Column(rp_id)
out["reid_class"] = Column(reid_class)
out["reid_sep"] = Column(np.where(hit, sep.arcsec, np.nan), unit="arcsec")
print(f"\nWarren (Reid & Parker) match within {MATCH_RAD}\" : {int(hit.sum())}")

# ------------------------------------------- reid 2014: IRAC 8 um photometry
reid14 = Table.read(REID_FILE, format="votable")
ra_r = np.asarray(reid14["_RA"], dtype=float)
dec_r = np.asarray(reid14["_DE"], dtype=float)
good_r = np.isfinite(ra_r) & np.isfinite(dec_r)
coords_r = SkyCoord(ra=ra_r[good_r] * u.deg, dec=dec_r[good_r] * u.deg)
r_rows = np.where(good_r)[0]

idx2, sep2, _ = coords_parent.match_to_catalog_sky(coords_r)
hit2 = sep2.arcsec <= MATCH_RAD

mag8 = np.full(len(out), np.nan)
emag8 = np.full(len(out), np.nan)
mag8[hit2] = np.asarray(reid14["__8.0_"], dtype=float)[r_rows[idx2[hit2]]]
emag8[hit2] = np.asarray(reid14["e__8.0_"], dtype=float)[r_rows[idx2[hit2]]]

out["irac8_mag"] = Column(mag8, unit="mag")
out["e_irac8_mag"] = Column(emag8, unit="mag")
out["has_phot"] = Column(np.where(np.isfinite(mag8), 1, 0).astype(np.int16))
print(f"reid2014 photometry match within {MATCH_RAD}\" : {int(hit2.sum())}"
      f"  (finite 8 um: {int(np.isfinite(mag8).sum())})")

# A stable primary key for the whole pipeline.  RP numbers are what the
# literature quotes, so use them where they exist and fall back to the HASH
# running number otherwise.
pn_id = np.array([r if r else f"HASH{h}" for r, h in
                  zip(rp_id, np.asarray(out["HASH_ID"]))], dtype="U20")
out["PN_ID"] = Column(pn_id)
assert len(set(pn_id)) == len(pn_id), "PN_ID is not unique"

# --------------------------------------------------------------- write outputs
os.makedirs(OUTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
out.write(OUT_PARENT, format="votable", overwrite=True)

nonpn_out = Table()
nonpn_out["HASH_ID"] = Column(np.asarray(nonpn["HASH"], dtype=np.int32))
nonpn_out["RA"] = Column(np.asarray(nonpn["RAJ2000"], dtype=float), unit="deg")
nonpn_out["Dec"] = Column(np.asarray(nonpn["DEJ2000"], dtype=float), unit="deg")
nonpn_out["PNstat"] = Column(clean(nonpn["PNstat"]).astype("U4"))
nonpn_out["object_type"] = Column(
    np.array([STATUS_LABELS.get(c, "other") for c in clean(nonpn["PNstat"])],
             dtype="U28"))
nonpn_out["hash_name"] = Column(clean(nonpn["Name"]).astype("U32"))
nonpn_out["hash_catalogue"] = Column(clean(nonpn["Catalogue"]).astype("U24"))
nonpn_out.write(OUT_NONPN, format="votable", overwrite=True)

# ------------------------------------------------------ summary table (paper)
rows = [("HASH V/163, all domains", len(hash_all)),
        ("LMC domain", len(lmc)),
        ("parent sample (PNstat T or P)", len(parent)),
        ("    true PN (T)", int(np.sum(clean(parent["PNstat"]) == "T"))),
        ("    probable PN (P)", int(np.sum(clean(parent["PNstat"]) == "P"))),
        ("excluded (other PNstat)", len(nonpn))]
for code, n in zip(codes[order], counts[order]):
    rows.append((f"    {STATUS_LABELS.get(code, code)} ({code})", int(n)))
rows += [("with Reid & Parker RP number", int(hit.sum())),
         ("with IRAC 8.0 um photometry", int(np.isfinite(mag8).sum()))]

summary = Table(rows=rows, names=("selection", "N"))
summary.write(OUT_SUMMARY, format="ascii.csv", overwrite=True)

print("\n" + "-" * 46)
print(f"  {'selection':<36}{'N':>6}")
print("-" * 46)
for label, n in rows:
    print(f"  {label:<36}{n:>6}")
print("-" * 46)

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(11.0, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
ax.scatter(np.asarray(nonpn_out["RA"]), np.asarray(nonpn_out["Dec"]), s=10, c="0.72",
           marker="x", lw=0.7, label=f"excluded ({len(nonpn)})")
prob = clean(parent["PNstat"]) == "P"
ax.scatter(np.asarray(out["RA"])[prob], np.asarray(out["Dec"])[prob], s=9, c="#c44e52",
           edgecolors="none", alpha=0.85,
           label=f"probable PN ({int(prob.sum())})")
ax.scatter(np.asarray(out["RA"])[~prob], np.asarray(out["Dec"])[~prob], s=9, c="#22539b",
           edgecolors="none", alpha=0.85,
           label=f"true PN ({int((~prob).sum())})")
ax.invert_xaxis()
ax.set_xlabel("RA (J2000, deg)")
ax.set_ylabel("Dec (J2000, deg)")
ax.set_title("LMC entries in HASH V/163", fontsize=10)
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9,
          handletextpad=0.4, borderpad=0.4)
ax.tick_params(labelsize=8)

# Provenance: which literature source HASH credits for each parent PN.
ax2 = fig.add_subplot(gs[0, 1])
prov = clean(parent["Catalogue"])
prov[prov == ""] = "unattributed"
labels, n_prov = np.unique(prov, return_counts=True)
srt = np.argsort(n_prov)[::-1]
top = srt[:6]
shown_labels = [labels[i] for i in top]
shown_counts = [n_prov[i] for i in top]
if len(srt) > 6:
    shown_labels.append(f"other ({len(srt) - 6} sources)")
    shown_counts.append(int(n_prov[srt[6:]].sum()))
ypos = np.arange(len(shown_counts))[::-1]
ax2.barh(ypos, shown_counts, color="#22539b", alpha=0.85, height=0.66)
for y, n in zip(ypos, shown_counts):
    ax2.text(n + max(shown_counts) * 0.015, y, str(n), va="center", fontsize=8)
ax2.set_yticks(ypos)
ax2.set_yticklabels(shown_labels, fontsize=7.5)
ax2.set_xlabel("number of PNe")
ax2.set_xlim(0, max(shown_counts) * 1.18)
ax2.set_title(f"Provenance of the {len(parent)} parent PNe", fontsize=10)
ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_PARENT)}   ({len(out)} rows)")
print(f"wrote {os.path.basename(OUT_NONPN)}   ({len(nonpn_out)} rows)")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
