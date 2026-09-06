"""Write results/summary.md: every number the paper quotes, as this run produced it.

Each row carries the value from this run, the value in the manuscript
(manual/manuscript_values.yaml), and a mark where the two differ.  Nothing is
rounded towards the manuscript and nothing is reconciled with it; a difference
is a result of the audit, not a problem to be tidied away.

Input   the step tables in the outputs directory
Output  results/summary.md

O. K. Khattab & M. D. Filipovic, Western Sydney University.
"""

import os
import sys
import warnings
from datetime import date

import numpy as np
import yaml
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _support import config

warnings.simplefilter("ignore", AstropyWarning)
cfg = config.load()

with open(os.path.join(config.ROOT, "manual", "manuscript_values.yaml")) as fh:
    manuscript = yaml.safe_load(fh)
PAPER = manuscript["values"]

rows = []          # (section, quantity, value, key or None)
notes = []


def vot(name):
    return Table.read(cfg.out(name), format="votable")


def csv(name):
    return Table.read(cfg.out(name), format="ascii.csv")


def text(table, key):
    return np.array([str(x).strip() for x in table[key]])


def add(section, label, value, key=None, fmt="{}"):
    rows.append((section, label, fmt.format(value), key, value))


def lookup(table, key_column, value_column):
    return {str(k).strip(): v for k, v in zip(table[key_column], table[value_column])}


# ---------------------------------------------------------------- step 1
parent = vot("step1_parent_catalogue.vot")
pnstat = text(parent, "hash_pnstat")
add("4.1 Parent sample", "Parent sample, HASH LMC with PNstat T or P", len(parent), "parent_total")
add("4.1 Parent sample", "PNstat T (true)", int((pnstat == "T").sum()), "parent_true")
add("4.1 Parent sample", "PNstat P (probable)", int((pnstat == "P").sum()), "parent_probable")
add("4.1 Parent sample", "With a Reid & Parker RP number",
    int(np.sum([not r.startswith("HASH") for r in text(parent, "RP_ID")])), "parent_with_rp")
add("4.1 Parent sample", "With a usable 8 um magnitude",
    int(np.isfinite(np.ma.asarray(parent["__8.0_"], dtype=float).filled(np.nan)).sum()),
    "parent_with_8um")
add("4.1 Parent sample", "LMC entries set aside by the PNstat cut",
    len(vot("step1_lmc_nonpn.vot")), "parent_set_aside")

# ---------------------------------------------------------------- step 2
mkt = lookup(csv("step2a_meerkat_match_stats.csv"), "quantity", "value")
ask = lookup(csv("step2b_askap_match_stats.csv"), "quantity", "value")
add("4.2 Radio counterparts", "MeerKAT counterparts within 4.5 arcsec",
    int(mkt["n_matched"]), "meerkat_matched")
add("4.2 Radio counterparts", "ASKAP counterparts within 4.5 arcsec",
    int(ask["n_accepted"]), "askap_matched")
add("4.2 Radio counterparts", "ASKAP pairs flagged, 4.5-6 arcsec",
    int(ask["n_borderline_4p5_to_6"]), "askap_review_4p5_to_6")
add("4.2 Radio counterparts", "ASKAP pairs flagged, 6-10 arcsec",
    int(ask["n_wide_review_6_to_10"]), "askap_review_6_to_10")
add("4.2 Radio counterparts", "Median offset, MeerKAT (arcsec)",
    mkt["median_offset_arcsec"], "median_offset_arcsec", "{:.2f}")
add("4.2 Radio counterparts", "Median offset, ASKAP (arcsec)",
    ask["median_offset_arcsec"], None, "{:.2f}")
add("4.2 Radio counterparts", "Mean dRA cos(dec), MeerKAT (arcsec)",
    mkt["mean_dra_cosdec_arcsec"], "mean_dra_meerkat", "{:+.2f}")
add("4.2 Radio counterparts", "Mean dDec, MeerKAT (arcsec)",
    mkt["mean_ddec_arcsec"], None, "{:+.2f}")
add("4.2 Radio counterparts", "Mean dRA cos(dec), ASKAP (arcsec)",
    ask["mean_dra_cosdec_arcsec"], "mean_dra_askap", "{:+.2f}")
add("4.2 Radio counterparts", "Mean dDec, ASKAP (arcsec)",
    ask["mean_ddec_arcsec"], None, "{:+.2f}")
add("4.2 Radio counterparts", "Expected chance matches, MeerKAT",
    mkt["expected_chance_matches"], None, "{:.2f}")
add("4.2 Radio counterparts", "Expected chance matches, ASKAP",
    ask["expected_chance_matches"], "chance_matches_askap", "{:.2f}")

# ---------------------------------------------------------------- step 3
visual = vot("step3b_meerkat_visual.vot")
union = lookup(csv("step3d_detection_summary.csv"), "category", "N")
add("4.3 Sub-threshold", "Inspected inside the mosaic",
    len(vot("step3a_meerkat_to_check.vot")), "inspected_inside_mosaic")
add("4.3 Sub-threshold", "Outside the mosaic",
    len(vot("step3a_meerkat_outside.vot")), "outside_mosaic")
add("4.3 Sub-threshold", "Recovered by inspection", len(visual), "visual_recovered")
add("4.3 Sub-threshold", "Median integrated flux of the recoveries (mJy)",
    float(np.nanmedian(np.asarray(visual["mkt_int_flux_mJy"], dtype=float))),
    "visual_median_flux_mjy", "{:.4f}")
add("4.3 Sub-threshold", "Median peak S/N of the recoveries",
    float(np.nanmedian(np.asarray(visual["mkt_snr"], dtype=float))),
    "visual_median_peak_snr", "{:.1f}")
add("4.3 Sub-threshold", "Enclosed-flux correction applied",
    cfg.enclosed_fraction(), None, "{:.3f}")
add("4.3 Sub-threshold", "MeerKAT detections in total",
    int(union["meerkat_and_askap"]) + int(union["meerkat_only"]), "meerkat_total")
add("4.3 Sub-threshold", "Detected by both surveys", int(union["meerkat_and_askap"]), "both_surveys")
add("4.3 Sub-threshold", "MeerKAT only", int(union["meerkat_only"]), "meerkat_only")
add("4.3 Sub-threshold", "ASKAP only", int(union["askap_only"]), "askap_only")
add("4.3 Sub-threshold", "Radio union", int(union["radio_union"]), "radio_union")
add("4.3 Sub-threshold", "Neither survey", int(union["neither"]), "neither")

master = vot("step3d_multisurvey_master.vot")
cls_parent = text(master, "reid_classification")
det_mkt = np.asarray(master["det_meerkat"], dtype=int) == 1
det_union = det_mkt | (np.asarray(master["det_askap"], dtype=int) == 1)
detfrac = []
for name in ("True", "Known", "Likely", "Possible"):
    sel = cls_parent == name
    detfrac.append((name, int(sel.sum()), int((sel & det_mkt).sum()),
                    int((sel & det_union).sum())))

# ---------------------------------------------------------------- step 4a
tally = vot("step5d_criteria_tally.vot")
sp = text(tally, "sp_class")
n_det = len(tally)
n_thermal = int((sp == "thermal").sum())
n_uncertain = int((sp == "uncertain").sum())
n_steep = int((sp == "steep").sum())
n_none = int((sp == "no_reliable_alpha").sum())
add("4.4 Spectral indices", "Detections with a reliable index",
    n_thermal + n_uncertain + n_steep, "reliable_indices")
add("4.4 Spectral indices", "Thermal", n_thermal, "n_thermal")
add("4.4 Spectral indices", "Uncertain", n_uncertain, "n_uncertain")
add("4.4 Spectral indices", "Steep", n_steep, "n_steep")
add("4.4 Spectral indices", "No reliable index", n_none, "n_no_reliable")
add("4.4 Spectral indices", "Four categories sum to the detections",
    f"{n_thermal + n_uncertain + n_steep + n_none} of {n_det}", None)

flux = np.asarray(tally["mkt_int_flux_Jy"], dtype=float) * 1e3
steep = sp == "steep"
for limit in (0.3, 0.6, 1.0):
    add("6.2 Steep sources", f"Steep with S(1.295 GHz) > {limit} mJy",
        int(np.sum(steep & np.isfinite(flux) & (flux > limit))), None)

field = csv("step4a_field_comparison.csv")
field_keys = {0.0: "alpha_offset_below_0p3", 0.3: "alpha_offset_0p3_0p6",
              0.6: "alpha_offset_0p6_1p0", 1.0: "alpha_offset_above_1p0"}
for row in field:
    low, high = float(row["flux_low_mJy"]), float(row["flux_high_mJy"])
    label = f"{low:g}-{high:g}" if np.isfinite(high) else f"> {low:g}"
    add("6.2 Field comparison", f"Median alpha offset, PNe minus field, {label} mJy",
        float(row["alpha_offset"]), field_keys.get(low), "{:+.2f}")

# ---------------------------------------------------------------- step 4b
mir = lookup(csv("step4b_mir_summary.csv"), "subset", "N")
mir_med = lookup(csv("step4b_mir_summary.csv"), "subset", "median_ratio")
add("4.5 MIR/radio", "Detections with a ratio", int(mir["all"]), "mir_with_ratio")
add("4.5 MIR/radio", "Sample median ratio", float(mir_med["all"]), "mir_median_all", "{:.2f}")
for name, key in (("thermal", "mir_median_thermal"), ("uncertain", None), ("steep", "mir_median_steep")):
    add("4.5 MIR/radio", f"Median ratio, {name} (N = {int(mir[name])})",
        float(mir_med[name]), key, "{:.2f}")
add("4.5 MIR/radio", "With both a reliable index and a ratio",
    int(mir["reliable_index_and_ratio"]), "mir_reliable_index_and_ratio")

# ---------------------------------------------------------------- step 4c
criteria = csv("step5d_criteria_outcomes.csv")
crit = {str(r["criterion"]): (int(r["pass"]), int(r["fail"]), int(r["not_measurable"]))
        for r in criteria}
add("4.6 Flux ceiling", "Below the ceiling", crit["4c_flux_ceiling"][0], "ceiling_below")
add("4.6 Flux ceiling", "Above the ceiling", crit["4c_flux_ceiling"][1], "ceiling_above")
add("4.6 Flux ceiling", "No usable integrated flux", crit["4c_flux_ceiling"][2],
    "ceiling_not_measurable")
above = csv("step4c_above_ceiling.csv")

# ---------------------------------------------------------------- step 5
for name, label, keys in (
        ("4a_spectral_index", "4a spectral index", ("crit_4a_pass", "crit_4a_fail", "crit_4a_na")),
        ("4b_mir_radio_ratio", "4b MIR/radio ratio", ("crit_4b_pass", "crit_4b_fail", "crit_4b_na")),
        ("4c_flux_ceiling", "4c flux ceiling", ("crit_4c_pass", "crit_4c_fail", "crit_4c_na"))):
    for value, word, key in zip(crit[name], ("pass", "fail", "not measurable"), keys):
        add("4.7 Criteria", f"{label}: {word}", value, key)

score = csv("step5d_passed_vs_measurable.csv")
verdict = text(tally, "radio_only_class")
for name, key in (("Radio_High", "radio_high"), ("Radio_Possible", "radio_possible"),
                  ("Radio_Weak", "radio_weak"), ("Radio_Rejected", "radio_rejected")):
    add("4.7 Criteria", name, int((verdict == name).sum()), key)

hs = text(tally, "hash_status_label")
hash_true = hs == "Listed as True in HASH"
hash_prob = hs == "Listed as Possible in HASH"
add("4.8 Radio vs optical", "HASH true PNe detected", int(hash_true.sum()), "hash_true_total")
add("4.8 Radio vs optical", "HASH true reaching Radio_High",
    int((hash_true & (verdict == "Radio_High")).sum()), "hash_true_high")
add("4.8 Radio vs optical", "HASH true rejected",
    int((hash_true & (verdict == "Radio_Rejected")).sum()), "hash_true_rejected")
add("4.8 Radio vs optical", "HASH probable PNe detected", int(hash_prob.sum()), "hash_probable_total")
add("4.8 Radio vs optical", "HASH probable reaching Radio_High",
    int((hash_prob & (verdict == "Radio_High")).sum()), "hash_probable_high")
add("4.8 Radio vs optical", "HASH probable rejected",
    int((hash_prob & (verdict == "Radio_Rejected")).sum()), "hash_probable_rejected")
add("4.8 Radio vs optical", "Rate reaching Radio_High, HASH true (per cent)",
    100 * (hash_true & (verdict == "Radio_High")).sum() / max(hash_true.sum(), 1),
    "rate_true_percent", "{:.0f}")
add("4.8 Radio vs optical", "Rate reaching Radio_High, HASH probable (per cent)",
    100 * (hash_prob & (verdict == "Radio_High")).sum() / max(hash_prob.sum(), 1),
    "rate_probable_percent", "{:.0f}")
add("4.8 Radio vs optical", "Optically weak but Radio_High",
    len(csv("step5d_promoted.csv")), "promoted")

# ---------------------------------------------------------------- step 6
pnlf = vot("step6a_pnlf.vot")
models = csv("step6a_model_parameters.csv")
models_high = csv("step6g_model_parameters_high.csv")


def canonical_row(table):
    names = [str(m) for m in table["model"]]
    return table[names.index("Ciardullo (canonical)")]


full = canonical_row(models)
high = canonical_row(models_high)
add("4.10 PNLF", "Sources entering the fit",
    int(np.asarray(pnlf["pnlf_included"], dtype=int).sum()), "pnlf_fitted_sources")
add("4.10 PNLF", "Bins fitted", int(full["dof"]) + int(full["k"]), "pnlf_bins")
add("4.10 PNLF", "M* (canonical Ciardullo, mag)",
    float(str(full["parameters"]).split(",")[-1]), "pnlf_m_star", "{:.3f}")
add("4.10 PNLF", "log10 L* (erg/s/Hz)",
    np.log10(cfg["pnlf"]["reference_luminosity_cgs"]
             * 10 ** (-float(str(full["parameters"]).split(",")[-1]) / 2.5)),
    "pnlf_log_lstar", "{:.2f}")
add("4.10 PNLF", "chi2_nu (canonical Ciardullo)", float(full["redchi"]), "pnlf_redchi", "{:.2f}")
add("4.10 PNLF", "R2 (canonical Ciardullo)", float(full["r2"]), "pnlf_r2", "{:.2f}")
add("4.10 PNLF", "AIC rank of the canonical form", int(full["rank"]), "pnlf_ciardullo_rank")
add("4.10 PNLF", "delta AIC of the canonical form",
    float(full["delta_aic"]), "pnlf_ciardullo_delta_aic", "{:.2f}")

masks = lookup(csv("step6e_mask_comparison.csv"), "treatment", "chi2_nu")
add("4.10 PNLF", "chi2_nu with the over-ceiling objects left in",
    float(masks["no_ceiling_mask"]), "pnlf_redchi_ceiling_off", "{:.2f}")

pnlf_high = vot("step6g_pnlf_high.vot")
add("4.11 High-confidence PNLF", "Sample size", len(pnlf_high), "high_sample_size")
add("4.11 High-confidence PNLF", "Bins fitted", int(high["dof"]) + int(high["k"]), "high_bins")
add("4.11 High-confidence PNLF", "M* (canonical Ciardullo, mag)",
    float(str(high["parameters"]).split(",")[-1]), "high_m_star", "{:.3f}")
add("4.11 High-confidence PNLF", "chi2_nu (canonical Ciardullo)",
    float(high["redchi"]), "high_redchi", "{:.2f}")
add("4.11 High-confidence PNLF", "AIC rank of the canonical form",
    int(high["rank"]), "high_ciardullo_rank")
add("4.11 High-confidence PNLF", "delta AIC of the canonical form",
    float(high["delta_aic"]), "high_ciardullo_delta_aic", "{:.2f}")

# ---------------------------------------------------------------- HASH SMC
hash_all = Table.read(cfg.data("hash_catalogue"), format="votable")
domain = np.array([str(x).strip() for x in hash_all["Domain"]])
status = np.array([str(x).strip() for x in hash_all["PNstat"]])
smc = domain == "SMC"
add("6.7 SMC", "SMC-domain entries in the frozen V/163 release", int(smc.sum()), None)
add("6.7 SMC", "SMC entries with PNstat T or P",
    int((smc & np.isin(status, ["T", "P"])).sum()), None)
add("6.7 SMC", "SMC PNstat T", int((smc & (status == "T")).sum()), None)
add("6.7 SMC", "SMC PNstat P", int((smc & (status == "P")).sum()), None)


# ---------------------------------------------------------------- write it out
def differs(value, paper):
    if isinstance(paper, str) or isinstance(value, str):
        return str(value) != str(paper)
    return abs(float(value) - float(paper)) > 0.5 * 10 ** -_decimals(paper)


def _decimals(paper):
    text_form = str(paper)
    return len(text_form.split(".")[1]) if "." in text_form else 0


lines = [
    "# Radio PNe in the LMC: the numbers this run produced",
    "",
    f"Generated by `scripts/make_summary.py` on {date.today().isoformat()}. "
    "Every value is read from the tables in this run's output directory; none "
    "is typed in.",
    "",
    f"Compared against **{manuscript['source']}**. A row marked **differs** is a "
    "finding of the audit: the pipeline was not adjusted to reproduce the "
    "manuscript, and no threshold, cut or mask was changed to move a value "
    "towards it.",
    "",
    "| Section | Quantity | This run | Manuscript | |",
    "|---|---|---|---|---|",
]
n_differ = 0
for section, label, shown, key, raw in rows:
    paper = PAPER.get(key) if key else None
    if paper is None:
        lines.append(f"| {section} | {label} | {shown} | — | |")
        continue
    flag = differs(raw, paper)
    n_differ += bool(flag)
    lines.append(f"| {section} | {label} | {shown} | {paper} | "
                 f"{'**differs**' if flag else 'agrees'} |")

lines += ["", f"{n_differ} of the compared values differ from the manuscript.", ""]

lines += ["## Detection fraction by optical class", "",
          "| R14 class | Parent | MeerKAT | Radio union | MeerKAT fraction |",
          "|---|---|---|---|---|"]
for name, total, n_mkt_cls, n_union_cls in detfrac:
    lines.append(f"| {name} | {total} | {n_mkt_cls} | {n_union_cls} | "
                 f"{100 * n_mkt_cls / max(total, 1):.1f}% |")
lines.append(f"| All | {len(master)} | {int(det_mkt.sum())} | {int(det_union.sum())} | "
             f"{100 * det_mkt.sum() / len(master):.1f}% |")
lines.append("")

lines += ["## Criteria passed, out of the criteria measurable", "",
          "| Passed | Measurable | N |", "|---|---|---|"]
for row in score:
    if int(row["measurable"]) == 0:
        lines.append(f"| — | none measurable | {int(row['N'])} |")
    else:
        lines.append(f"| {int(row['passed'])} | {int(row['measurable'])} | {int(row['N'])} |")
lines.append("")

lines += ["## Detections above the flux ceiling", "",
          "| Identifier | S(1.295 GHz) / mJy | HASH | Spectral class | S / S(NGC 7027) |",
          "|---|---|---|---|---|"]
for row in above:
    lines.append(f"| {row['identifier']} | {float(row['S_1295_mJy']):.2f} | "
                 f"{row['hash_pnstat']} | {row['sp_class']} | {float(row['S_over_S_7027']):.1f} |")
lines.append("")

for title, table in (("Ten models, full sample", models),
                     ("Ten models, Radio_High sample", models_high)):
    lines += [f"## {title}", "",
              "| Rank | Model | k | R2 | AIC | dAIC | chi2_nu | Parameters |",
              "|---|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append(f"| {int(row['rank'])} | {row['model']} | {int(row['k'])} | "
                     f"{float(row['r2']):.3f} | {float(row['aic']):.2f} | "
                     f"{float(row['delta_aic']):.2f} | {float(row['redchi']):.3f} | "
                     f"{row['parameters']} |")
    lines.append("")

lines += ["## Contingency tables", ""]
for title, name in (("Against the HASH classification", "step5d_tally_vs_hash.csv"),
                    ("Against the R14 optical class", "step5d_tally_vs_reid.csv")):
    table = csv(name)
    lines += [f"### {title}", "",
              "| " + " | ".join(table.colnames) + " |",
              "|" + "---|" * len(table.colnames)]
    for row in table:
        lines.append("| " + " | ".join(str(row[c]) for c in table.colnames) + " |")
    lines.append("")

lines += ["## Fixed parameters this run used", "",
          "| Parameter | Value |", "|---|---|",
          f"| LMC distance | {cfg['distance']['lmc_kpc']} kpc |",
          f"| Acceptance radius | {cfg['crossmatch']['accept_arcsec']} arcsec |",
          f"| Flux ceiling | {cfg['flux_ceiling']['ceiling_mjy']} mJy |",
          f"| Flux ceiling as a magnitude | {cfg.ceiling_magnitude():.3f} (computed) |",
          f"| Completeness limit | {cfg['pnlf']['completeness_limit_mag']} mag "
          f"= {cfg.completeness_flux_ujy():.1f} uJy (computed) |",
          f"| Aperture enclosed fraction | {cfg.enclosed_fraction():.3f} (computed) |",
          f"| Spectral index cuts | n >= {cfg['spectral_index']['min_points_reliable']}, "
          f"delta alpha < {cfg['spectral_index']['max_alpha_error']}, "
          f"chi2_nu < {cfg['spectral_index']['max_reduced_chisq']} |",
          f"| Thermal / steep boundaries | {cfg['spectral_index']['thermal_range']}, "
          f"{cfg['spectral_index']['steep_below']} |",
          f"| IRAC 8 um zero point | {cfg['mir_radio']['irac8_zero_point_jy']} Jy |",
          f"| PNLF bin width | {cfg['pnlf']['bin_width_mag']} mag |",
          f"| Fit seed | {cfg['pnlf']['fit_seed']} |",
          ""]

out = cfg.result("summary.md")
with open(out, "w") as fh:
    fh.write("\n".join(lines))
print(f"wrote {out}")
print(f"{len(rows)} quantities, {n_differ} differing from the manuscript")
