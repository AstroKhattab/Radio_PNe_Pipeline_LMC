"""Step 1: cross-match the parent sample with MeerKAT and ASKAP.

Each PN takes the nearest radio source within the radius set in config.yaml.
The Step 2 manual flux densities (01_data/manual_fluxes.csv) are then added to the
MeerKAT detections.  Survey footprints are read from the finite pixels of the
two images.

Outputs: catalogue columns mkt_* / askap_* / in_* / detection,
         03_figures/f1_sky_map.pdf, 03_figures/fa1_offsets.pdf,
         05_tables/t2_detections.tex, numbers block [step1].
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

import common as c

cfg = c.CFG
plt = c.plot_style()
radius = cfg["crossmatch"]["radius_arcsec"]

cat = c.read_catalogue()
n = len(cat)
ra = np.asarray(cat["RA"], float)
dec = np.asarray(cat["Dec"], float)
ids = np.asarray(cat["RP_ID"], str)

# --- MeerKAT ------------------------------------------------------------------
mk = c.read_input("meerkat_broadband")
m = cfg["columns"]["meerkat_broadband"]
mk_ra, mk_dec = c.numeric(mk, m["ra"]), c.numeric(mk, m["dec"])
idx_m, sep_m = c.nearest_match(ra, dec, mk_ra, mk_dec, radius)
hit_m = idx_m >= 0
assert len(set(idx_m[hit_m])) == hit_m.sum(), "one MeerKAT source matched to two PNe"


def take(values, index, hit, scale=1.0):
    out = np.full(n, np.nan)
    out[hit] = values[index[hit]] * scale
    return out


mkt_ra = take(mk_ra, idx_m, hit_m)
mkt_dec = take(mk_dec, idx_m, hit_m)
mkt_flux = take(c.numeric(mk, m["int"]), idx_m, hit_m, 1e3)        # mJy
mkt_err = take(c.numeric(mk, m["err_int"]), idx_m, hit_m, 1e3)
mkt_rms = take(c.numeric(mk, m["rms"]), idx_m, hit_m, 1e3)
mkt_snr = take(c.numeric(mk, m["peak"]), idx_m, hit_m, 1e3) / mkt_rms
mkt_sep = np.where(hit_m, sep_m, np.nan)
method = np.where(hit_m, "catalogue", "").astype("U10")

# Step 2: flux densities measured by hand on the image.
manual = Table.read(c.data("manual_fluxes"), format="ascii.csv", comment="#")
key = cfg["manual"]["key"]
row_of = {k: i for i, k in enumerate(np.asarray(cat[key], str))}
n_manual = 0
for r in manual:
    i = row_of.get(str(r[key]).strip())
    if i is None:
        raise SystemExit(f"manual source {r[key]} is not in the parent sample")
    if method[i]:
        raise SystemExit(f"manual source {r[key]} already has a catalogue match")
    mkt_ra[i], mkt_dec[i] = r["ra_deg"], r["dec_deg"]
    mkt_flux[i], mkt_err[i], mkt_rms[i] = r["flux_mJy"], r["err_flux_mJy"], r["rms_mJy"]
    mkt_snr[i], mkt_sep[i] = r["snr"], r["offset_arcsec"]
    method[i] = "manual"
    n_manual += 1
det_m = method != ""

# --- ASKAP --------------------------------------------------------------------
ak = c.read_input("askap")
a = cfg["columns"]["askap"]
ak_ra, ak_dec = c.numeric(ak, a["ra"]), c.numeric(ak, a["dec"])
idx_a, sep_a = c.nearest_match(ra, dec, ak_ra, ak_dec, radius)
det_a = idx_a >= 0
assert len(set(idx_a[det_a])) == det_a.sum(), "one ASKAP source matched to two PNe"
askap_flux = take(c.numeric(ak, a["int"]), idx_a, det_a)              # mJy
fit_err = take(c.numeric(ak, a["err_int"]), idx_a, det_a)
fit_err[fit_err < 0] = np.nan                                         # VizieR sentinel
askap_err = np.sqrt(np.nan_to_num(fit_err) ** 2 + (cfg["askap"]["flux_scale"] * askap_flux) ** 2)
askap_snr = take(c.numeric(ak, a["peak"]), idx_a, det_a) / take(c.numeric(ak, a["rms"]), idx_a, det_a)
askap_ra = take(ak_ra, idx_a, det_a)
askap_dec = take(ak_dec, idx_a, det_a)
askap_sep = np.where(det_a, sep_a, np.nan)


# --- footprints -----------------------------------------------------------------
def image_mask(path):
    hdul = fits.open(path, memmap=True)
    return np.squeeze(hdul[0].data), WCS(hdul[0].header).celestial


def inside(path, ra_deg, dec_deg):
    image, wcs = image_mask(path)
    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < image.shape[1]) & (iy < image.shape[0])
    out = np.zeros(len(ra_deg), bool)
    out[ok] = np.isfinite(image[iy[ok], ix[ok]])
    return out


def outline(path, stride=64):
    image, wcs = image_mask(path)
    valid = np.isfinite(np.asarray(image[::stride, ::stride])).astype(float)
    fig, ax = plt.subplots()
    segs = ax.contour(valid, levels=[0.5]).allsegs[0]
    plt.close(fig)
    edge = max(segs, key=len)
    o_ra, o_dec = wcs.pixel_to_world_values(edge[:, 0] * stride, edge[:, 1] * stride)
    return np.append(o_ra, o_ra[0]), np.append(o_dec, o_dec[0])


in_mk = inside(c.data("meerkat_image"), ra, dec)
in_ak = inside(c.data("askap_image"), ra, dec)

detection = np.full(n, "none", dtype="U8")
detection[det_m & det_a] = "both"
detection[det_m & ~det_a] = "meerkat"
detection[~det_m & det_a] = "askap"

for name, values in [
    ("mkt_ra", mkt_ra), ("mkt_dec", mkt_dec), ("mkt_flux_mJy", mkt_flux),
    ("mkt_err_mJy", mkt_err), ("mkt_rms_mJy", mkt_rms), ("mkt_snr", mkt_snr),
    ("mkt_sep_arcsec", mkt_sep), ("mkt_method", method),
    ("askap_ra", askap_ra), ("askap_dec", askap_dec), ("askap_flux_mJy", askap_flux),
    ("askap_err_mJy", askap_err), ("askap_snr", askap_snr), ("askap_sep_arcsec", askap_sep),
    ("in_meerkat", in_mk), ("in_askap", in_ak), ("detection", detection),
]:
    c.set_column(cat, name, values)
c.write_catalogue(cat)

# --- numbers ----------------------------------------------------------------------
cat_m = method == "catalogue"
chance_m = np.pi * (radius / 3600) ** 2 * len(mk) / cfg["meerkat"]["area_deg2"] * n
chance_a = np.pi * (radius / 3600) ** 2 * len(ak) / cfg["askap"]["area_deg2"] * n


def offsets(r1, d1, r2, d2):
    dra, ddec = SkyCoord(r1 * u.deg, d1 * u.deg).spherical_offsets_to(SkyCoord(r2 * u.deg, d2 * u.deg))
    return dra.to_value(u.arcsec), ddec.to_value(u.arcsec)


xm, ym = offsets(ra[cat_m], dec[cat_m], mkt_ra[cat_m], mkt_dec[cat_m])
xa, ya = offsets(ra[det_a], dec[det_a], askap_ra[det_a], askap_dec[det_a])
pn = np.asarray(cat["hash_pnstat"], str)


def by_status(mask):
    return f"{int(mask.sum())} ({int((mask & (pn == 'T')).sum())} T, {int((mask & (pn == 'P')).sum())} P)"


none = detection == "none"
c.write_numbers("step1", [
    ("Match radius (arcsec)", radius),
    ("Inside MeerKAT footprint", int(in_mk.sum())),
    ("Inside ASKAP footprint", int(in_ak.sum())),
    ("MeerKAT catalogue matches", int(cat_m.sum())),
    ("  of which S/N >= 5", int((cat_m & (mkt_snr >= 5)).sum())),
    ("  of which S/N < 5", int((cat_m & (mkt_snr < 5)).sum())),
    ("MeerKAT manual (Step 2)", n_manual),
    ("MeerKAT detections total", by_status(det_m)),
    ("ASKAP detections", by_status(det_a)),
    ("Both surveys", by_status(detection == "both")),
    ("MeerKAT only", by_status(detection == "meerkat")),
    ("ASKAP only", by_status(detection == "askap")),
    ("Radio union", by_status(detection != "none")),
    ("No detection", by_status(none)),
    ("  no detection, inside MeerKAT footprint", int((none & in_mk).sum())),
    ("  no detection, outside MeerKAT footprint", int((none & ~in_mk).sum())),
    ("MeerKAT median offset, catalogue (arcsec)", f"{np.median(mkt_sep[cat_m]):.2f}"),
    ("ASKAP median offset (arcsec)", f"{np.median(askap_sep[det_a]):.2f}"),
    ("MeerKAT mean dRA, dDec (arcsec)", f"{xm.mean():+.2f}, {ym.mean():+.2f}"),
    ("MeerKAT std dRA, dDec (arcsec)", f"{xm.std():.2f}, {ym.std():.2f}"),
    ("ASKAP mean dRA, dDec (arcsec)", f"{xa.mean():+.2f}, {ya.mean():+.2f}"),
    ("ASKAP std dRA, dDec (arcsec)", f"{xa.std():.2f}, {ya.std():.2f}"),
    ("Expected chance matches MeerKAT", f"{chance_m:.1f}"),
    ("Expected chance matches ASKAP", f"{chance_a:.1f}"),
])

# --- t2_detections.tex: detections by HASH status ---------------------------------------------------
groups = [("MeerKAT", det_m), ("ASKAP", det_a), ("Both", detection == "both"),
          ("MeerKAT only", detection == "meerkat"), ("ASKAP only", detection == "askap"),
          ("None", none)]
rows = []
for label, sel in [("HASH T", pn == "T"), ("HASH P", pn == "P"), ("Total", np.ones(n, bool))]:
    if label == "Total":
        rows.append("midrule")
    rows.append([label, int(sel.sum())] + [int((sel & g).sum()) for _, g in groups])
c.write_tex("t2_detections.tex", ["Sample", "$N$"] + [g for g, _ in groups], rows, align="lrrrrrrr")

# --- f1_sky_map.pdf ----------------------------------------------------------------
colours = {"meerkat": "#0072B2", "askap": "#8E5AA9", "both": "#6F2DBD",
           "askap_only": "#E69F00", "none": "#B7BEC8"}
mk_out, ak_out = outline(c.data("meerkat_image")), outline(c.data("askap_image"))

wcs = WCS(naxis=2)
wcs.wcs.crpix = [500.0, 400.0]
wcs.wcs.cdelt = [-0.025, 0.025]
wcs.wcs.crval = [np.median(ra), np.median(dec)]
wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]

fig = plt.figure(figsize=(10.2, 8.2))
ax = fig.add_subplot(111, projection=wcs)
world = ax.get_transform("world")
ax.fill(*ak_out, color=colours["askap"], alpha=0.035, transform=world)
ax.plot(*ak_out, color=colours["askap"], lw=1.3, ls=(0, (5, 2, 1.3, 2)), transform=world)
ax.fill(*mk_out, color=colours["meerkat"], alpha=0.045, transform=world)
ax.plot(*mk_out, color=colours["meerkat"], lw=1.25, ls="--", transform=world)

sel = {k: detection == k for k in ("none", "meerkat", "askap", "both")}
ax.scatter(ra[sel["none"]], dec[sel["none"]], s=11, color=colours["none"], alpha=0.55,
           lw=0, transform=world, rasterized=True)
ax.scatter(ra[sel["meerkat"]], dec[sel["meerkat"]], s=38, facecolors="white",
           edgecolors=colours["meerkat"], lw=0.9, transform=world)
ax.scatter(ra[sel["askap"]], dec[sel["askap"]], s=74, marker="D",
           facecolors=colours["askap_only"], edgecolors="white", lw=0.8, transform=world)
ax.scatter(ra[sel["both"]], dec[sel["both"]], s=72, marker="P",
           facecolors=colours["both"], edgecolors="white", lw=0.75, transform=world)

ax.coords[0].set_axislabel("Right Ascension (J2000)")
ax.coords[1].set_axislabel("Declination (J2000)")
ax.coords[0].set_major_formatter("hh:mm")
ax.coords.grid(True, color="#DDE2E8", lw=0.55, alpha=0.65)
px, py = wcs.world_to_pixel_values(np.concatenate([mk_out[0], ak_out[0]]),
                                   np.concatenate([mk_out[1], ak_out[1]]))
sx, sy = np.ptp(px), np.ptp(py)
ax.set_xlim(px.min() - 0.07 * sx, px.max() + 0.07 * sx)
ax.set_ylim(py.min() - 0.08 * sy, py.max() + 0.08 * sy)

handles = [
    Line2D([], [], marker=".", ls="", color=colours["none"], ms=7,
           label=f"No radio detection ($N$={sel['none'].sum()})"),
    Line2D([], [], marker="o", ls="", mfc="white", mec=colours["meerkat"], ms=7,
           label=f"MeerKAT only ($N$={sel['meerkat'].sum()})"),
    Line2D([], [], marker="P", ls="", mfc=colours["both"], mec="white", ms=8,
           label=f"MeerKAT + ASKAP ($N$={sel['both'].sum()})"),
    Line2D([], [], marker="D", ls="", mfc=colours["askap_only"], mec="white", ms=7,
           label=f"ASKAP only ($N$={sel['askap'].sum()})"),
]
leg = ax.legend(handles=handles, loc="upper left", fontsize=8.8, framealpha=0.96)
ax.add_artist(leg)
ax.legend(handles=[
    Line2D([], [], color=colours["meerkat"], ls="--", label="MeerKAT 1.295 GHz footprint"),
    Line2D([], [], color=colours["askap"], ls=(0, (5, 2, 1.3, 2)), label="ASKAP 888 MHz footprint"),
], loc="lower left", fontsize=8.2, framealpha=0.96)
fig.savefig(c.figure("f1_sky_map.pdf"))
plt.close(fig)

# --- fa1_offsets.pdf: positional offsets, both surveys ------------------------------------------
with plt.style.context("ggplot"):
    fig = plt.figure(figsize=(15, 7))
    outer = fig.add_gridspec(1, 2, wspace=0.34)
    for k, (x, y, label) in enumerate([(xm, ym, "(a) MeerKAT 1.295 GHz"), (xa, ya, "(b) ASKAP 888 MHz")]):
        g = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[k], width_ratios=[6, 1, 0.35],
                                    height_ratios=[1, 6], wspace=0.10, hspace=0.10)
        top, main = fig.add_subplot(g[0, 0]), fig.add_subplot(g[1, 0])
        side, cax = fig.add_subplot(g[1, 1], sharey=main), fig.add_subplot(g[1, 2])
        hb = main.hexbin(x, y, gridsize=30, cmap="plasma", mincnt=1, rasterized=True)  # vector hexbins render wrongly in some PDF viewers
        main.set_xlabel("\u0394RA \u00b7 cos(\u03b4)  (arcsec)", fontsize=13)
        main.set_ylabel("\u0394\u03b4  (arcsec)", fontsize=13)
        main.set_xlim(-5, 5)
        main.set_ylim(-5, 5)
        main.axhline(0, color="black", lw=1.5)
        main.axvline(0, color="black", lw=1.5)
        main.grid(axis="both", linestyle="--", alpha=0.3)
        hist = dict(bins="auto", color="#D32F2F", alpha=0.55)
        top.hist(x, **hist)
        top.axvline(np.mean(x), color="black", ls="--", lw=1.2)
        top.set_ylabel("Counts", fontsize=11)
        top.set_xlim(-5, 5)
        top.set_title(label, fontsize=13)
        plt.setp(top.get_xticklabels(), visible=False)
        side.hist(y, **hist, orientation="horizontal")
        side.axhline(np.mean(y), color="black", ls="--", lw=1.2)
        side.set_xlabel("Counts", fontsize=11)
        side.set_ylim(-5, 5)
        plt.setp(side.get_yticklabels(), visible=False)
        cbar = fig.colorbar(hb, cax=cax)
        cbar.set_label("Counts per bin", fontsize=11)
        cbar.ax.tick_params(labelsize=11)
        at = AnchoredText("\n".join([
            rf"$\langle\Delta\mathrm{{RA}}\rangle = {np.mean(x):+.2f}^{{\prime\prime}}$",
            rf"$\sigma(\Delta\mathrm{{RA}}) = {np.std(x):.2f}^{{\prime\prime}}$",
            rf"$\langle\Delta\delta\rangle = {np.mean(y):+.2f}^{{\prime\prime}}$",
            rf"$\sigma(\Delta\delta) = {np.std(y):.2f}^{{\prime\prime}}$",
            f"$N = {len(x)}$"]), prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.3")
        at.patch.set_alpha(0.9)
        main.add_artist(at)
    fig.savefig(c.figure("fa1_offsets.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

print(f"step1: MeerKAT {det_m.sum()}, ASKAP {det_a.sum()}, both {(detection == 'both').sum()}")
