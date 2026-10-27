"""Step 4: spectral index.

Fits S ~ nu^alpha to the twelve MeerKAT sub-bands plus the ASKAP 888 MHz point,
classifies the index (thermal / uncertain / steep), sets how well it is
constrained (tier high / medium / low) and gives the Step 4 points of the
score.  It then fits the MeerKAT 5 sigma catalogue the same way and compares
the median index of the PNe with that of the radio field in flux density bins.
All thresholds are in config.yaml.

Outputs: catalogue columns alpha*, 03_figures/f3_spectral_index.pdf,
         numbers blocks [step4] and [step4_field].
"""

import numpy as np

import common as c

cfg = c.CFG
si = cfg["spectral_index"]
plt = c.plot_style()

cat = c.read_catalogue()
fit = c.spectral_indices(cat)
alpha, err, npts = fit["alpha"], fit["alpha_err"], fit["n_points"]
cls = c.spectral_class(alpha)

tier = np.full(len(cat), "none", dtype="U6")
tier[np.isfinite(alpha)] = "low"
for name in ("medium", "high"):
    rule = si["tiers"][name]
    tier[np.isfinite(alpha) & (npts >= rule["min_points"]) & (err < rule["max_error"])] = name

pts = si["points"]
points = np.zeros(len(cat), dtype=int)
for t in ("high", "medium", "low"):
    points[(cls == "thermal") & (tier == t)] = pts["thermal"][t]
points[cls == "uncertain"] = pts["uncertain"]
points[cls == "steep"] = pts["steep"]

for name, values in [("alpha", alpha), ("alpha_err", err), ("alpha_chi2nu", fit["chi2nu"]),
                     ("alpha_n_points", npts), ("alpha_n_meerkat", fit["n_meerkat"]),
                     ("alpha_class", cls), ("alpha_tier", tier), ("alpha_points", points)]:
    c.set_column(cat, name, values)
c.write_catalogue(cat)

# Numbers: the gate survivors are the sample that is scored.
gate = np.asarray(cat["gate"], str)
live = np.isin(gate, ["passed", "not_measured"])
lines = [("Gate survivors", int(live.sum())),
         ("  with a fit (>= 2 points)", int((live & np.isfinite(alpha)).sum())),
         ("  no fit", int((live & ~np.isfinite(alpha)).sum()))]
for k in ("thermal", "uncertain", "steep", "inverted"):
    sel = live & (cls == k)
    lines.append((f"  {k}", f"{sel.sum()}  (high {(sel & (tier == 'high')).sum()}, "
                            f"medium {(sel & (tier == 'medium')).sum()}, low {(sel & (tier == 'low')).sum()})"))
for t in ("high", "medium", "low"):
    lines.append((f"  tier {t}", int((live & (tier == t)).sum())))
for p in sorted(set(points[live])):
    lines.append((f"  Step 4 points = {p}", int((live & (points == p)).sum())))
fin = live & np.isfinite(alpha)
lines += [("Median alpha, survivors with fit", f"{np.median(alpha[fin]):+.2f}"),
          ("Median alpha, thermal", f"{np.median(alpha[live & (cls == 'thermal')]):+.2f}"),
          ("Median alpha, steep", f"{np.median(alpha[live & (cls == 'steep')]):+.2f}"),
          ("Fits with ASKAP point", int((fin & np.isfinite(fit["flux"][:, -1])).sum())),
          ("All detections with a fit (incl. rejected)", int(np.isfinite(alpha).sum()))]
wild = np.where(np.isfinite(alpha) & ((alpha < -1.5) | (alpha > 2.0)))[0]
lines.append(("alpha outside -1.5..2 (ID alpha+-err n gate)",
              "; ".join(f"{cat['RP_ID'][i]} {alpha[i]:+.2f}+-{err[i]:.2f} n={npts[i]} {gate[i]}"
                        for i in wild[np.argsort(alpha[wild])])))
c.write_numbers("step4", lines)

# Field comparison: the same fit, on the MeerKAT sub-bands alone, for every
# source in the 5 sigma catalogue (overwhelmingly background galaxies).  The
# PNe are the gate survivors with a fit, binned by their MeerKAT flux density.
mk = cfg["meerkat"]
sub = c.read_input("meerkat_subbands")
chans = list(mk["channels"].items())
ffreq = np.full((len(sub), len(chans)), np.nan)
fflux = np.full_like(ffreq, np.nan)
ferr = np.full_like(ffreq, np.nan)
for k, (ch, mhz) in enumerate(chans):
    fname = next(n for n in (f"{ch}_int_flux_1", f"{ch}_int_flux") if n in sub.colnames)
    s = c.numeric(sub, fname) * 1e3
    e = c.numeric(sub, f"{ch}_err_int_flux") * 1e3
    e = np.where(np.isfinite(e) & (e > 0), e, mk["default_error_fraction"] * s)
    ffreq[:, k] = mhz / 1e3
    fflux[:, k] = np.where(s > 0, s, np.nan)
    ferr[:, k] = np.hypot(e, mk["flux_scale"] * s)
field_alpha = c.spectral_fit(ffreq, fflux, ferr)[0]
field_s = c.numeric(sub, "ch0_int_flux_1") * 1e3
pn_s = np.asarray(cat["mkt_flux_mJy"], float)
field_lines = [("Field sources with a fit", int(np.isfinite(field_alpha).sum()))]
bins = si["field_bins"]
for lo_s, hi_s in zip(bins[:-1], bins[1:]):
    in_pn = fin & (pn_s >= lo_s) & (pn_s < hi_s)
    in_f = np.isfinite(field_alpha) & (field_s >= lo_s) & (field_s < hi_s)
    m_pn, m_f = np.median(alpha[in_pn]), np.median(field_alpha[in_f])
    field_lines.append((f"  {lo_s}-{hi_s} mJy: PNe N, median; field N, median; offset",
                        f"{in_pn.sum()} {m_pn:+.2f}; {in_f.sum()} {m_f:+.2f}; {m_pn - m_f:+.2f}"))
c.write_numbers("step4_field", field_lines)

# f3_spectral_index.pdf: histogram of alpha by class, gate survivors.
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(-1.5, 2.05, 0.10)
names = {"steep": "Steep", "uncertain": "Uncertain", "thermal": "Thermal"}
for k in ("steep", "uncertain", "thermal"):
    v = alpha[live & (cls == k)]
    ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, color=c.COLOURS[k], edgecolor="white",
            lw=0.6, label=f"{names[k]} ($N$={len(v)})")
lo = si["thermal"][0]
for x in (si["steep_below"], lo):
    ax.axvline(x, color="#333333", lw=1.4, ls="--")
n_out = int((fin & ((alpha < bins[0]) | (alpha > bins[-1]))).sum())
ax.text(0.98, 0.70, f"No fit: $N$={(live & ~np.isfinite(alpha)).sum()}\n"
        f"Outside axis (shown in end bins): $N$={n_out}", transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb"))
ax.set_xlabel(r"Spectral index $\alpha$ ($S_\nu \propto \nu^\alpha$)")
ax.set_ylabel("Number of sources")
ax.legend(fontsize=9)
fig.savefig(c.figure("f3_spectral_index.pdf"))
plt.close(fig)
print(f"step4: {fin.sum()} fits among {live.sum()} gate survivors")
