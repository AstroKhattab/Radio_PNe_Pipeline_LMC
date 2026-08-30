"""
step02c_crossmatch_surveys.py
=============================
Step 2 - Radio cross-matching.

Takes the HASH parent sample from Step 1 and looks for a radio counterpart in
both surveys at once:

    MeerKAT   1.295 GHz  (Cotton et al. 2026)   3 sigma Aegean catalogue
    ASKAP-EMU  888 MHz   (Pennock et al. 2021)  published source list

Doing the two together, rather than in separate scripts against separate
copies of the parent list, means the detection bookkeeping is unambiguous: a
PN is either detected in MeerKAT, in ASKAP, in both, or in neither, and those
four numbers add up to the parent sample.

Acceptance radius
-----------------
4.5 arcsec throughout, which is three MeerKAT pixels and roughly half the
1.295 GHz synthesised beam.  Nothing wider is accepted automatically; ASKAP
matches between 4.5 and 10 arcsec are recorded with a 'review' flag so they
can be inspected but they do not count as detections.

Chance coincidence
------------------
For each survey we quote the number of matches expected if the radio sources
were sprinkled at random over the survey area, N_parent * n_radio * pi r^2 / A.
This is the number the reader needs in order to judge whether the match rate
is meaningful, so it is written into the summary table rather than printed
and forgotten.

Spectral-index columns are looked up from the MeerKAT 5 sigma multi-band
catalogue where one exists; the actual fitting happens in Step 4a.

Outputs
-------
    03_Outputs/step02c_radio_crossmatch.vot    parent + all radio columns
    03_Outputs/step02c_crossmatch_summary.csv  counts behind the paper table
    04_Figures/step02c_crossmatch.pdf          offsets + detection summary

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

PARENT_FILE = os.path.join(OUTS, "step01b_parent_sample.vot")
MKT3_FILE = os.path.join(DATA, "MeerKAT_3sigma.vot")
MKT5_FILE = os.path.join(DATA, "MeerKAT_Final_5sigma.vot")
ASKAP_FILE = os.path.join(DATA, "ASKAP_LMC_888MHz_catalogue_full.vot")

OUT_TABLE = os.path.join(OUTS, "step02c_radio_crossmatch.vot")
OUT_SUMMARY = os.path.join(OUTS, "step02c_crossmatch_summary.csv")
OUT_FIG = os.path.join(FIGS, "step02c_crossmatch.pdf")

ACCEPT_RAD = 4.5      # arcsec
ASKAP_REVIEW_RAD = 10.0
MKT_AREA_DEG2 = 25.0   # MeerKAT LMC mosaic
ASKAP_AREA_DEG2 = 120.0
ASKAP_CAL_FRAC = 0.08  # Pennock et al. 2021 absolute calibration term


def nearest(cat_coords, ref_coords):
    idx, sep, _ = cat_coords.match_to_catalog_sky(ref_coords)
    return idx, sep.arcsec


print("=" * 60)
print("  Step 2  -  MeerKAT and ASKAP cross-match")
print("=" * 60)

parent = Table.read(PARENT_FILE, format="votable")
n_par = len(parent)
ra = np.asarray(parent["RA"], dtype=float)
dec = np.asarray(parent["Dec"], dtype=float)
coords_par = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
print(f"\nparent sample : {n_par}")

out = parent.copy()

# ============================================================ MeerKAT 1.295 GHz
mkt3 = Table.read(MKT3_FILE, format="votable")
print(f"MeerKAT 3 sigma catalogue : {len(mkt3)}")

c_mkt = SkyCoord(ra=np.asarray(mkt3["ra"], dtype=float) * u.deg,
                 dec=np.asarray(mkt3["dec"], dtype=float) * u.deg)
i_mkt, s_mkt = nearest(coords_par, c_mkt)
mkt_hit = s_mkt <= ACCEPT_RAD


def mkt_col(name, scale=1.0):
    src = np.asarray(mkt3[name], dtype=float)
    val = np.full(n_par, np.nan)
    val[mkt_hit] = src[i_mkt[mkt_hit]] * scale
    return val


out["mkt_sep"] = Column(np.where(mkt_hit, s_mkt, np.nan).astype("f4"),
                        unit="arcsec")
out["mkt_detected"] = Column(mkt_hit.astype(np.int16))
out["mkt_ra"] = Column(mkt_col("ra"), unit="deg")
out["mkt_dec"] = Column(mkt_col("dec"), unit="deg")
out["mkt_peak_mJy_beam"] = Column(mkt_col("peak_flux", 1e3), unit="mJy/beam")
out["mkt_int_mJy"] = Column(mkt_col("int_flux", 1e3), unit="mJy")
out["mkt_e_int_mJy"] = Column(mkt_col("err_int_flux", 1e3), unit="mJy")
out["mkt_rms_mJy_beam"] = Column(mkt_col("local_rms", 1e3), unit="mJy/beam")
out["mkt_a_arcsec"] = Column(mkt_col("a"), unit="arcsec")
out["mkt_b_arcsec"] = Column(mkt_col("b"), unit="arcsec")

with np.errstate(invalid="ignore", divide="ignore"):
    out["mkt_snr"] = Column(
        (np.asarray(out["mkt_peak_mJy_beam"]) /
         np.asarray(out["mkt_rms_mJy_beam"])).astype("f4"))

print(f"  matched within {ACCEPT_RAD}\" : {int(mkt_hit.sum())}")

# Spectral-index columns from the multi-band 5 sigma catalogue.  Not fitted
# here; Step 4a redoes the fit from the sub-band fluxes.
mkt5 = Table.read(MKT5_FILE, format="votable")


def pick(tbl, options):
    for c in options:
        if c in tbl.colnames:
            return c
    raise KeyError(f"none of {options} in table")


ra5 = pick(mkt5, ["RAJ2000_deg", "dRAJ2000", "ra"])
dec5 = pick(mkt5, ["DECJ2000_deg", "dDECJ2000", "dec"])
c_mkt5 = SkyCoord(ra=np.asarray(mkt5[ra5], dtype=float) * u.deg,
                  dec=np.asarray(mkt5[dec5], dtype=float) * u.deg)
i5, s5 = nearest(coords_par, c_mkt5)
hit5 = mkt_hit & (s5 <= ACCEPT_RAD)

out["mkt5_row"] = Column(np.where(hit5, i5, -1).astype(np.int32))
out["mkt5_matched"] = Column(hit5.astype(np.int16))
print(f"  also in the 5 sigma multi-band list : {int(hit5.sum())}")

# ================================================================ ASKAP 888 MHz
askap = Table.read(ASKAP_FILE, format="votable")
print(f"ASKAP-EMU catalogue : {len(askap)}")

c_ask = SkyCoord(ra=np.asarray(askap["RAJ2000"], dtype=float) * u.deg,
                 dec=np.asarray(askap["DEJ2000"], dtype=float) * u.deg)
i_ask, s_ask = nearest(coords_par, c_ask)
ask_hit = s_ask <= ACCEPT_RAD
ask_review = (s_ask > ACCEPT_RAD) & (s_ask <= ASKAP_REVIEW_RAD)


def ask_col(name, mask):
    src = np.asarray(askap[name], dtype=float)
    val = np.full(n_par, np.nan)
    val[mask] = src[i_ask[mask]]
    return val


keep = ask_hit | ask_review
emu = np.full(n_par, "", dtype="U34")
emu[keep] = np.asarray(askap["EMU-ID"], dtype=str)[i_ask[keep]]
flags = np.full(n_par, "", dtype="U6")
flags[keep] = np.asarray(askap["Flags"], dtype=str)[i_ask[keep]]

status = np.full(n_par, "unmatched", dtype="U10")
status[ask_review] = "review"
status[ask_hit] = "accepted"

e_int = ask_col("e_IntFlux", keep)
e_int[e_int < 0] = np.nan          # VizieR sentinel for "not fitted"
int_flux = ask_col("IntFlux", keep)

out["askap_sep"] = Column(np.where(keep, s_ask, np.nan).astype("f4"),
                          unit="arcsec")
out["askap_status"] = Column(status)
out["askap_detected"] = Column(ask_hit.astype(np.int16))
out["askap_id"] = Column(emu)
out["askap_flags"] = Column(flags)
out["askap_ra"] = Column(ask_col("RAJ2000", keep), unit="deg")
out["askap_dec"] = Column(ask_col("DEJ2000", keep), unit="deg")
out["askap_peak_mJy_beam"] = Column(ask_col("PeakFlux", keep).astype("f4"),
                                    unit="mJy/beam")
out["askap_int_mJy"] = Column(int_flux.astype("f4"), unit="mJy")
out["askap_e_int_fit_mJy"] = Column(e_int.astype("f4"), unit="mJy")
out["askap_e_int_mJy"] = Column(
    np.sqrt(e_int ** 2 + (ASKAP_CAL_FRAC * int_flux) ** 2).astype("f4"),
    unit="mJy")
out["askap_rms_mJy_beam"] = Column(ask_col("LocalRMS", keep).astype("f4"),
                                   unit="mJy/beam")
out["askap_a_arcsec"] = Column(ask_col("a", keep).astype("f4"), unit="arcsec")
out["askap_b_arcsec"] = Column(ask_col("b", keep).astype("f4"), unit="arcsec")
with np.errstate(invalid="ignore", divide="ignore"):
    out["askap_snr"] = Column(
        (np.asarray(out["askap_peak_mJy_beam"]) /
         np.asarray(out["askap_rms_mJy_beam"])).astype("f4"))

print(f"  matched within {ACCEPT_RAD}\" : {int(ask_hit.sum())}")
print(f"  {ACCEPT_RAD}-{ASKAP_REVIEW_RAD}\" (flagged, not counted) : "
      f"{int(ask_review.sum())}")

# ================================================================== bookkeeping
both = mkt_hit & ask_hit
mkt_only = mkt_hit & ~ask_hit
ask_only = ask_hit & ~mkt_hit
neither = ~mkt_hit & ~ask_hit
detected = mkt_hit | ask_hit

detmode = np.full(n_par, "none", dtype="U10")
detmode[mkt_only] = "MeerKAT"
detmode[ask_only] = "ASKAP"
detmode[both] = "both"
out["detection"] = Column(detmode)
out["catalogue_detected"] = Column(detected.astype(np.int16))

assert int(both.sum() + mkt_only.sum() + ask_only.sum() + neither.sum()) == n_par

# Chance coincidence, measured rather than assumed.  Displacing every parent
# position by a fixed 60 arcsec in a random direction preserves the clustering
# of the optical sample and the real non-uniformity of the radio source density,
# which a plain N * n * pi r^2 / A calculation does not.  Averaging over many
# realisations gives the number of 4.5 arcsec matches expected by accident.
rng = np.random.default_rng(20260830)
N_TRIALS = 200
SHIFT = 60.0 * u.arcsec

trial_mkt = np.empty(N_TRIALS)
trial_ask = np.empty(N_TRIALS)
for t in range(N_TRIALS):
    pa = rng.uniform(0.0, 360.0, n_par) * u.deg
    shifted = coords_par.directional_offset_by(pa, SHIFT)
    trial_mkt[t] = np.sum(nearest(shifted, c_mkt)[1] <= ACCEPT_RAD)
    trial_ask[t] = np.sum(nearest(shifted, c_ask)[1] <= ACCEPT_RAD)

chance_mkt = float(trial_mkt.mean())
chance_ask = float(trial_ask.mean())
chance_mkt_sd = float(trial_mkt.std())
chance_ask_sd = float(trial_ask.std())
print(f"\nchance coincidence from {N_TRIALS} offset realisations:")
print(f"  MeerKAT : {chance_mkt:.1f} +/- {chance_mkt_sd:.1f}  "
      f"({chance_mkt / max(mkt_hit.sum(), 1) * 100:.1f}% of detections)")
print(f"  ASKAP   : {chance_ask:.1f} +/- {chance_ask_sd:.1f}  "
      f"({chance_ask / max(ask_hit.sum(), 1) * 100:.1f}% of detections)")

print("\n" + "-" * 52)
print(f"  {'':<34}{'N':>6}")
print("-" * 52)
rows = [
    ("parent sample", n_par),
    ("MeerKAT 1.295 GHz detections", int(mkt_hit.sum())),
    ("ASKAP-EMU 888 MHz detections", int(ask_hit.sum())),
    ("detected in both", int(both.sum())),
    ("MeerKAT only", int(mkt_only.sum())),
    ("ASKAP only", int(ask_only.sum())),
    ("catalogue detections (union)", int(detected.sum())),
    ("no catalogue counterpart", int(neither.sum())),
    ("ASKAP 4.5-10 arcsec, flagged", int(ask_review.sum())),
]
for label, n in rows:
    print(f"  {label:<34}{n:>6}")
print(f"  {'chance matches, MeerKAT':<34}{chance_mkt:>6.1f}")
print(f"  {'chance matches, ASKAP':<34}{chance_ask:>6.1f}")
print("-" * 52)

summary = Table(
    rows=rows + [("expected by chance, MeerKAT", round(chance_mkt, 1)),
                 ("chance scatter, MeerKAT", round(chance_mkt_sd, 1)),
                 ("expected by chance, ASKAP", round(chance_ask, 1)),
                 ("chance scatter, ASKAP", round(chance_ask_sd, 1))],
    names=("quantity", "value"))
summary.write(OUT_SUMMARY, format="ascii.csv", overwrite=True)

os.makedirs(OUTS, exist_ok=True)
out.write(OUT_TABLE, format="votable", overwrite=True)

# ====================================================================== figure
fig = plt.figure(figsize=(12.0, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32)


def offset_panel(axis, mask, radio_ra, radio_dec, title, colour):
    cr = SkyCoord(ra=radio_ra[mask] * u.deg, dec=radio_dec[mask] * u.deg)
    dra, ddec = coords_par[mask].spherical_offsets_to(cr)
    dx = dra.arcsec
    dy = ddec.arcsec
    axis.axhline(0, color="0.55", lw=0.8, zorder=0)
    axis.axvline(0, color="0.55", lw=0.8, zorder=0)
    circle = plt.Circle((0, 0), ACCEPT_RAD, fill=False, ls="--", lw=1.0,
                        ec="0.35", zorder=1)
    axis.add_patch(circle)
    axis.scatter(dx, dy, s=11, c=colour, alpha=0.65, edgecolors="none", zorder=2)
    axis.set_xlim(-5.2, 5.2)
    axis.set_ylim(-5.2, 5.2)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$\Delta\alpha\,\cos\delta$ (arcsec)")
    axis.set_ylabel(r"$\Delta\delta$ (arcsec)")
    axis.set_title(title, fontsize=10)
    axis.text(0.03, 0.97,
              f"N = {len(dx)}\nmedian = {np.nanmedian(np.hypot(dx, dy)):.2f}\"",
              transform=axis.transAxes, va="top", fontsize=8,
              bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.3"))
    axis.tick_params(labelsize=8)


offset_panel(fig.add_subplot(gs[0, 0]), mkt_hit,
             np.asarray(out["mkt_ra"]), np.asarray(out["mkt_dec"]),
             "MeerKAT 1.295 GHz", "#22539b")
offset_panel(fig.add_subplot(gs[0, 1]), ask_hit,
             np.asarray(out["askap_ra"]), np.asarray(out["askap_dec"]),
             "ASKAP-EMU 888 MHz", "#c44e52")

ax3 = fig.add_subplot(gs[0, 2])
cats = ["both", "MeerKAT\nonly", "ASKAP\nonly", "no radio\ncounterpart"]
vals = [int(both.sum()), int(mkt_only.sum()), int(ask_only.sum()),
        int(neither.sum())]
cols = ["#4c72b0", "#22539b", "#c44e52", "0.75"]
bars = ax3.bar(cats, vals, color=cols, width=0.68)
for b, v in zip(bars, vals):
    ax3.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
             ha="center", fontsize=9)
ax3.set_ylabel("number of PNe")
ax3.set_ylim(0, max(vals) * 1.16)
ax3.set_title(f"Parent sample of {n_par} PNe", fontsize=10)
ax3.tick_params(labelsize=8)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)

fig.savefig(OUT_FIG, bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"\nwrote {os.path.basename(OUT_TABLE)}   ({len(out)} rows)")
print(f"wrote {os.path.basename(OUT_SUMMARY)}")
print(f"wrote {os.path.basename(OUT_FIG)}")
