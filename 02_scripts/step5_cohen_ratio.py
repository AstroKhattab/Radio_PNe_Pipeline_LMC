"""Step 5: Cohen ratio.

The ratio of the SAGE 8.0 um flux density to the MeerKAT 1.295 GHz flux density
(Cohen & Green 2001; Cohen et al. 2011).  A ratio at or above the Galactic PN
median of Cohen et al. (2011) earns the Step 5 point.  No 8 um photometry or no
MeerKAT flux gives no point.

Outputs: catalogue columns flux_8um_Jy, cohen_ratio*, 03_figures/f4_cohen_ratio.pdf,
         numbers block [step5].
"""

import numpy as np
from matplotlib.ticker import FuncFormatter

import common as c

cfg = c.CFG
cr = cfg["cohen_ratio"]
plt = c.plot_style()

cat = c.read_catalogue()
mag = np.asarray(cat["mag_8um"], float)
mag_err = np.asarray(cat["err_mag_8um"], float)
f8 = cr["irac8_zero_point_jy"] * 10 ** (-mag / 2.5)
f8_err = f8 * np.log(10) / 2.5 * mag_err
s = np.asarray(cat["mkt_flux_mJy"], float) * 1e-3
s_err = np.asarray(cat["mkt_err_mJy"], float) * 1e-3
ratio = np.where(np.isfinite(f8) & (s > 0), f8 / s, np.nan)
ratio_err = ratio * np.hypot(f8_err / f8, s_err / s)
points = np.where(np.isfinite(ratio) & (ratio >= cr["threshold"]), cr["points"], 0)

for name, values in [("flux_8um_Jy", f8), ("cohen_ratio", ratio),
                     ("cohen_ratio_err", ratio_err), ("cohen_points", points)]:
    c.set_column(cat, name, values)
c.write_catalogue(cat)

gate = np.asarray(cat["gate"], str)
live = np.isin(gate, ["passed", "not_measured"])
cls = np.asarray(cat["alpha_class"], str)
has = live & np.isfinite(ratio)
lines = [("Threshold", cr["threshold"]),
         ("Gate survivors with a ratio", int(has.sum())),
         ("Gate survivors without (no 8 um or no MeerKAT)", int((live & ~np.isfinite(ratio)).sum())),
         ("  with R >= threshold", int((has & (ratio >= cr["threshold"])).sum())),
         ("Median R, all with ratio", f"{np.median(ratio[has]):.1f}")]
for k in ("thermal", "uncertain", "steep", "none"):
    sel = has & (cls == k)
    if sel.any():
        lines.append((f"  {k}: N, median R, N with R >= threshold",
                      f"{sel.sum()}, {np.median(ratio[sel]):.1f}, {(sel & (ratio >= cr['threshold'])).sum()}"))
c.write_numbers("step5", lines)

# f4_cohen_ratio.pdf: thermal against steep, outline histograms, threshold marked.
fig, ax = plt.subplots(figsize=(6.2, 4.2))
bins = np.logspace(-1.0, 3.6, 24)   # log spacing: R spans four decades
for k, label in (("thermal", "Thermal"), ("steep", "Steep")):
    v = ratio[has & (cls == k)]
    ax.hist(v, bins=bins, histtype="step", lw=2, color=c.COLOURS[k],
            label=f"{label} ($N$={len(v)}, median {np.median(v):.1f})")
    ax.axvline(np.median(v), color=c.COLOURS[k], ls=":", lw=1.3)
ax.axvline(cr["threshold"], color="k", lw=1.4,
           label=f"Cohen et al. 2011 Galactic PN median ({cr['threshold']})")
ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel(r"Cohen ratio  $R = S_{8\,\mu{\rm m}} / S_{1.295\,{\rm GHz}}$")
ax.set_ylabel("Number of sources")
ax.legend(fontsize=8, loc="upper right", framealpha=1, edgecolor="none")
fig.savefig(c.figure("f4_cohen_ratio.pdf"))
plt.close(fig)
print(f"step5: {has.sum()} ratios")
