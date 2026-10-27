"""Step 6: score and confidence level.

Score = Step 4 points + Step 5 points (0-4) for every gate survivor.  The level
cuts are in config.yaml.  Sources above the flux gate are Rejected; PNe with
no radio detection get no level.

Outputs: catalogue columns score, level; 05_tables/t3_grading.tex,
         t4_levels.tex (levels against HASH status and R14 class),
         t5_catalogue_excerpt.tex; numbers block [step6].
"""

import numpy as np

import common as c

cfg = c.CFG
lv = cfg["levels"]
si = cfg["spectral_index"]
cr = cfg["cohen_ratio"]

cat = c.read_catalogue()
gate = np.asarray(cat["gate"], str)
live = np.isin(gate, ["passed", "not_measured"])
score = np.where(live, np.asarray(cat["alpha_points"], int) + np.asarray(cat["cohen_points"], int), -1)
level = np.full(len(cat), "", dtype="U9")
level[live] = "Weak"
level[live & (score >= lv["possible_min"])] = "Possible"
level[live & (score >= lv["high_min"])] = "High"
level[gate == "rejected"] = "Rejected"
c.set_column(cat, "score", score)
c.set_column(cat, "level", level)
c.write_catalogue(cat)

pn = np.asarray(cat["hash_pnstat"], str)
levels = ["High", "Possible", "Weak", "Rejected"]
lines = [(f"{name}", f"{(level == name).sum()} ({((level == name) & (pn == 'T')).sum()} T, "
                     f"{((level == name) & (pn == 'P')).sum()} P)") for name in levels]
for s in range(0, 5):
    lines.append((f"  score = {s}", int((score == s).sum())))
cls = np.asarray(cat["alpha_class"], str)
for name in levels[:3]:
    sel = level == name
    lines.append((f"  {name}: thermal/uncertain/steep/no class",
                  "/".join(str((sel & np.isin(cls, k)).sum())
                           for k in (["thermal"], ["uncertain"], ["steep"], ["none", "inverted"]))))
R14_CLASSES = ["Known", "True", "Likely", "Possible"]
r14 = np.asarray(cat["reid_class"], str)
for name in levels:
    sel = level == name
    lines.append((f"  {name}: R14 Known/True/Likely/Possible/not in R14",
                  "/".join(str((sel & (r14 == k)).sum()) for k in R14_CLASSES + ["none"])))
c.write_numbers("step6", lines)

# t3_grading.tex: the grading scheme, written from config.yaml.
p = si["points"]
lo, hi = si["thermal"]
th = si["tiers"]
ceil = cfg["flux_gate"]["ceiling_mjy"]
rows = [
    ["3. Flux cut", "Physical ceiling", f"$S_{{1.295}} > {ceil}$\\,mJy", "Rejected"],
    "midrule",
    ["4. Spectral index", "Thermal, high tier", f"${lo} \\le \\alpha \\le +{hi}$, $n\\ge{th['high']['min_points']}$, "
     f"$\\sigma_\\alpha<{th['high']['max_error']}$", f"+{p['thermal']['high']}"],
    ["", "Thermal, medium tier", f"$n\\ge{th['medium']['min_points']}$, $\\sigma_\\alpha<{th['medium']['max_error']}$",
     f"+{p['thermal']['medium']}"],
    ["", "Thermal, low tier", "any other fit", f"+{p['thermal']['low']}"],
    ["", "Uncertain", f"${si['steep_below']} \\le \\alpha < {lo}$", f"+{p['uncertain']}"],
    ["", "Steep or no fit", f"$\\alpha < {si['steep_below']}$", f"{p['steep']}"],
    "midrule",
    ["5. MIR/radio ratio", "Cohen ratio", f"$R \\ge {cr['threshold']}$", f"+{cr['points']}"],
    ["", "", f"$R < {cr['threshold']}$ or no data", "0"],
    "midrule",
    ["6. Level", "High", f"score $\\ge {lv['high_min']}$", ""],
    ["", "Possible", f"score $= {lv['possible_min']}$" if lv["high_min"] - lv["possible_min"] == 1
     else f"${lv['possible_min']}\\le$ score $<{lv['high_min']}$", ""],
    ["", "Weak", f"score $< {lv['possible_min']}$", ""],
]
c.write_tex("t3_grading.tex", ["Step", "Test", "Condition", "Points"], rows, align="llll")

# t4_levels.tex: levels against the HASH status and the R14 optical class.
r14 = np.asarray(cat["reid_class"], str)
groups = [("HASH T", pn == "T"), ("HASH P", pn == "P"), "midrule"]
groups += [(f"R14 {k}", r14 == k) for k in R14_CLASSES] + [("Not in R14", r14 == "none"), "midrule"]
groups += [("Total", np.ones(len(cat), bool))]
rows = []
for g in groups:
    if g == "midrule":
        rows.append(g)
        continue
    label, sel = g
    rows.append([label] + [int((sel & (level == k)).sum()) for k in levels]
                + [int((sel & (level == "")).sum()), int(sel.sum())])
c.write_tex("t4_levels.tex", ["Optical class"] + levels + ["Not detected", "Total"], rows, align="lrrrrrr",
            note="HASH: T true, P probable. R14: optical class of Reid \\& Parker (2014).")

# t5_catalogue_excerpt.tex: catalogue excerpt, rows spread evenly through the RA-ordered detections.
det = np.where(level != "")[0]
det = det[np.argsort(np.asarray(cat["RA"])[det])]
pick = det[np.linspace(0, len(det) - 1, cfg["excerpt_rows"]).astype(int)]
rows = []
for i in pick:
    r = cat[i]
    rows.append([c.tex_escape(r["RP_ID"]), r["hash_pnstat"], c.fmt(r["mkt_flux_mJy"], 3),
                 c.fmt(r["askap_flux_mJy"], 2),
                 f"${r['alpha']:+.2f}\\pm{r['alpha_err']:.2f}$" if np.isfinite(r["alpha"]) else "---",
                 r["alpha_tier"] if r["alpha_tier"] != "none" else "---",
                 c.fmt(r["cohen_ratio"], 1), r["score"] if r["score"] >= 0 else "---", r["level"]])
c.write_tex("t5_catalogue_excerpt.tex",
            ["ID", "HASH", "$S_{1.3}$ (mJy)", "$S_{0.9}$ (mJy)", "$\\alpha$", "Tier", "$R$", "Score", "Level"],
            rows, align="llrrllrrl")
print("step6: " + ", ".join(f"{k} {(level == k).sum()}" for k in levels))
