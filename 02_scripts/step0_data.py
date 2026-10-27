"""Step 0: build the parent sample.

Selects the HASH planetary nebulae of the galaxy named in config.yaml, attaches
the Reid & Parker (R14) running number and optical class, and the SAGE 8.0 um
photometry, by position.  Writes a fresh 04_catalogues/catalogue.fits that every later
step adds columns to, and the input-data table.

Outputs: 04_catalogues/catalogue.fits, 05_tables/t1_data.tex, numbers block [step0].
"""

import numpy as np
from astropy.table import Table

import common as c

cfg = c.CFG
col = cfg["columns"]
radius = cfg["parent"]["ancillary_arcsec"]

hash_all = c.read_input("hash")
reid = c.read_input("reid_positions")
phot = c.read_input("reid_photometry")

# Parent sample: the galaxy's PNe with the chosen HASH status.
h = col["hash"]
in_galaxy = c.text(hash_all, h["domain"]) == cfg["parent"]["domain"]
status = c.text(hash_all, h["status"])
parent = hash_all[in_galaxy & np.isin(status, cfg["parent"]["status"])]
n = len(parent)
ra = c.numeric(parent, h["ra"])
dec = c.numeric(parent, h["dec"])
pnstat = c.text(parent, h["status"])
hash_id = c.text(parent, h["id"])

# R14 running number and class; sources R14 never listed keep a HASH name.
r = col["reid_positions"]
idx, _ = c.nearest_match(ra, dec, c.numeric(reid, r["ra"]), c.numeric(reid, r["dec"]), radius)
has_rp = idx >= 0
ident = np.array([f"HASH{x}" for x in hash_id], dtype="U20")
ident[has_rp] = c.text(reid, r["id"])[idx[has_rp]]
reid_class = np.full(n, "none", dtype="U10")   # "none": not in the R14 list
reid_class[has_rp] = c.text(reid, r["cls"])[idx[has_rp]]
assert len(set(ident)) == n, "identifier is not unique"

# SAGE 8.0 um magnitude from the R14 photometry.
p = col["reid_photometry"]
idx_p, _ = c.nearest_match(ra, dec, c.numeric(phot, p["ra"]), c.numeric(phot, p["dec"]), radius)
has_phot = idx_p >= 0
mag8 = np.full(n, np.nan)
err8 = np.full(n, np.nan)
mag8[has_phot] = c.numeric(phot, p["mag8"])[idx_p[has_phot]]
err8[has_phot] = c.numeric(phot, p["err8"])[idx_p[has_phot]]

cat = Table()
cat["RP_ID"] = ident
cat["Name"] = c.text(parent, h["name"]).astype("U30")
cat["RA"] = ra
cat["Dec"] = dec
cat["hash_id"] = hash_id.astype("U10")
cat["hash_pnstat"] = pnstat.astype("U2")
cat["reid_class"] = reid_class
cat["opt_diam_arcsec"] = c.numeric(parent, h["diameter"])
cat["mag_8um"] = mag8
cat["err_mag_8um"] = err8
c.write_catalogue(cat)

# Input-data table.
mk3 = c.read_input("meerkat_broadband")
mk5 = c.read_input("meerkat_subbands")
askap = c.read_input("askap")
mk, ak, opt = cfg["meerkat"], cfg["askap"], cfg["optical"]
n_8um = int(np.isfinite(mag8).sum())
rows = [
    [mk["label"], f"{mk['frequency_ghz']}\\,GHz ({mk['band']})", mk["resolution"],
     mk["reference"], f"{len(mk3):,} / {len(mk5):,}".replace(",", "{,}")],
    [ak["label"], f"{ak['frequency_ghz'] * 1000:.0f}\\,MHz", ak["resolution"],
     ak["reference"], f"{len(askap):,}".replace(",", "{,}")],
    ["Reid \\& Parker optical PNe", "H$\\alpha$ / spectroscopy", "$\\sim1\\arcsec$",
     opt["reid_reference"], f"{len(reid)}"],
    ["\\textit{Spitzer} SAGE", "IRAC 8.0\\,\\ensuremath{\\mu}m", "$\\sim2\\arcsec$",
     opt["sage_reference"], f"{n_8um}"],
    ["HASH PN database (V/163)", "parent catalogue", "---", opt["hash_reference"], f"{n}"],
]
channels = ", ".join(f"{v:g}" for v in mk["channels"].values())
c.write_tex("t1_data.tex", ["Dataset", "Band / frequency", "Resolution", "Reference", "$N$"],
            rows, align="lllll",
            note=f"MeerKAT sub-band centres used in the spectral fit (MHz): {channels}.")

c.write_numbers("step0", [
    ("HASH entries in file", len(hash_all)),
    (f"HASH {cfg['parent']['domain']} entries", int(in_galaxy.sum())),
    ("Parent sample (T+P)", n),
    ("  HASH T", int((pnstat == "T").sum())),
    ("  HASH P", int((pnstat == "P").sum())),
    ("Set aside (other PNstat)", int(in_galaxy.sum()) - n),
    ("With R14 running number", int(has_rp.sum())),
    ("HASH-only identifier", int((~has_rp).sum())),
    ("With R14 photometry row", int(has_phot.sum())),
    ("With 8.0 um magnitude", n_8um),
    ("MeerKAT broadband catalogue rows", len(mk3)),
    ("MeerKAT sub-band catalogue rows", len(mk5)),
    ("ASKAP catalogue rows", len(askap)),
    ("R14 position list rows", len(reid)),
])
print(f"step0: {n} parent PNe")
