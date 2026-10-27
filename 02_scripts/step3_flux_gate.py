"""Step 3: flux gate.

A PN at the LMC distance cannot be brighter than NGC 7027 placed there, so any
MeerKAT 1.295 GHz flux density above the ceiling in config.yaml is rejected and
read no further.  Sources with no MeerKAT flux (ASKAP only) are not gated.

Outputs: catalogue column gate, 03_figures/f2_flux_gate.pdf,
         numbers block [step3].
"""

import numpy as np

import common as c

cfg = c.CFG
plt = c.plot_style()
ceiling = cfg["flux_gate"]["ceiling_mjy"]

cat = c.read_catalogue()
flux = np.asarray(cat["mkt_flux_mJy"], float)
measured = np.isfinite(flux) & (flux > 0)
gate = np.full(len(cat), "", dtype="U12")
gate[np.asarray(cat["detection"], str) != "none"] = "not_measured"
gate[measured] = "passed"
gate[measured & (flux > ceiling)] = "rejected"
c.set_column(cat, "gate", gate)
c.write_catalogue(cat)

rej = gate == "rejected"
pn = np.asarray(cat["hash_pnstat"], str)
order = np.where(rej)[0][np.argsort(-flux[rej])]
c.write_numbers("step3", [
    ("Ceiling (mJy)", ceiling),
    ("Ceiling as M_radio", f"{c.magnitude(ceiling * 1e-3):.2f}"),
    ("Passed", int((gate == "passed").sum())),
    ("Rejected", f"{rej.sum()} ({(rej & (pn == 'T')).sum()} T, {(rej & (pn == 'P')).sum()} P)"),
    ("Not measured at 1.295 GHz (ASKAP only)", int((gate == "not_measured").sum())),
    ("Rejected sources (ID, mJy)", ", ".join(f"{cat['RP_ID'][i]} {flux[i]:.2f}" for i in order)),
])

# f2_flux_gate.pdf: spectral index against flux density, with the ceiling.
fit = c.spectral_indices(cat)
cls = c.spectral_class(fit["alpha"])
fig, ax = plt.subplots(figsize=(5.6, 4.0))
x_top = max(5.0, np.nanmax(flux) * 1.3)
y_lo, y_hi = -3.5, 2.5
alpha = fit["alpha"]
for k in ("thermal", "uncertain", "steep"):
    sel = measured & (cls == k)
    ax.scatter(flux[sel], np.clip(alpha[sel], y_lo, y_hi), s=14, color=c.COLOURS[k],
               edgecolor="white", lw=0.3, label=f"{k} ($N$={sel.sum()})")
# Indices beyond the axis are drawn at its edge as open triangles.
for off, y, mark in ((alpha < y_lo, y_lo + 0.1, "v"), (alpha > y_hi, y_hi - 0.1, "^")):
    off &= measured
    ax.scatter(flux[off], np.full(off.sum(), y), marker=mark, s=22, facecolors="none", edgecolors="0.3")
ax.set_ylim(y_lo, y_hi)
ax.axvline(ceiling, color="0.15", lw=1.1)
ax.axvspan(ceiling, x_top, color="#C23B33", alpha=0.09)
ax.text(ceiling * 1.08, 0.97, f"{ceiling} mJy", transform=ax.get_xaxis_transform(),
        va="top", fontsize=8, color="#8E2820")
lo, hi = cfg["spectral_index"]["thermal"]
for y in (cfg["spectral_index"]["steep_below"], lo):
    ax.axhline(y, color="0.55", ls="--", lw=0.9)
ax.set_xscale("log")
ax.set_xlim(0.008, x_top)
ax.set_xlabel(r"$S_{1.295\,{\rm GHz}}$ (mJy)")
ax.set_ylabel(r"Spectral index $\alpha$")
ax.legend(fontsize=7.5, loc="lower left", frameon=False)
fig.savefig(c.figure("f2_flux_gate.pdf"))
plt.close(fig)
print(f"step3: {rej.sum()} rejected above {ceiling} mJy")
