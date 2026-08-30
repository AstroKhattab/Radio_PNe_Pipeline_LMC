# lmc-radio-pne

Analysis pipeline for a radio-continuum catalogue of planetary nebulae in the
Large Magellanic Cloud, and for the first radio planetary nebula luminosity
function (PNLF) measured for a galaxy other than our own.

The parent sample is taken from the HASH PN database (Parker, Bojicic & Frew
2016), release V/163: every LMC entry classified as a true or probable PN, 707
objects in all. Those positions are cross-matched against the MeerKAT
1.295 GHz LMC mosaic and the ASKAP-EMU 888 MHz survey; inspection of the
MeerKAT image recovers nebulae the source finder missed; three criteria that
never use the optical classification are applied to each detection; and the
resulting sample is fitted with ten luminosity-function models.

Every number, table and figure in the accompanying paper is produced by this
code. Running it from the raw catalogues reproduces the published results in
full.

## Citing this work

If you use this pipeline, or the catalogue it produces, please cite the paper:

> Khattab, O. K., Filipovic, M. D., et al., *A radio-continuum catalogue and
> the first radio planetary nebula luminosity function for the Large Magellanic
> Cloud* (in preparation).

`CITATION.cff` carries the machine-readable form. Please cite the paper rather
than the repository alone: the paper documents the calibration choices and the
selection function, and a result quoted without them is difficult to interpret.

## Pipeline

Run in order. `scripts/run_pipeline.py` enforces the order and stops at the
first failure; `--list` prints the steps and `--from <step>` resumes partway.
Each step writes a table, a short CSV summary, and a figure.

| Step | What it does | Script |
|---|---|---|
| 1 | Parent sample from HASH V/163 (707 PNe) | `step1_parent_catalogue.py` |
| 2a | Cross-match with MeerKAT 1.295 GHz | `step2a_crossmatch_meerkat.py` |
| 2b | Cross-match with ASKAP-EMU 888 MHz | `step2b_crossmatch_askap.py` |
| 3a | Inspection queue for the unmatched | `step3a_inspect_meerkat.py` |
| 3b | Photometry of the visual detections | `step3b_extract_visual.py` |
| 3b2 | Aperture-corrected visual photometry | `step3b2_reextract_visual.py` |
| 3c | Detection catalogue and sky maps | `step3c_detection_catalogue.py` |
| 3d | Multi-survey union | `step3d_multisurvey_union.py` |
| 4a | Criterion: in-band spectral index | `step4a_spectral_index.py` |
| 4b | Criterion: mid-infrared to radio ratio | `step4b_mir_radio_ratio.py` |
| 4c | Criterion: radio flux ceiling | `step4c_flux_ceiling.py` |
| 5a | Combined evidence grade | `step5a_classify_confidence.py` |
| 5b | Spectral reliability recut | `step5b_spectral_recut.py` |
| 5c | Detection and evidence summary | `step5c_detection_and_evidence.py` |
| 5d | Criteria tally, radio against optical | `step5d_criteria_tally.py` |
| 6a | PNLF: ten empirical models | `step6a_fit_pnlf_empirical.py` |
| 6b | PNLF: Ciardullo function | `step6b_fit_pnlf_ciardullo.py` |
| 6c | PNLF: paper figures | `step6c_pnlf_paper_figures.py` |
| 6d | PNLF: model grid | `step6d_pnlf_empirical_grid.py` |
| 6e | PNLF: masking comparison | `step6e_pnlf_mask_vs_clean.py` |
| 6g | PNLF on the high-confidence sample | `step6a_fit_pnlf_empirical.py --sample high` |
| 6h | Model grid, high-confidence sample | `step6d_pnlf_empirical_grid.py --sample high` |
| 7 | **The final catalogue** | `step7_final_catalogue.py` |

```bash
python scripts/run_pipeline.py --list        # show the steps
python scripts/run_pipeline.py               # run all of them
python scripts/run_pipeline.py --from 6a     # resume from the PNLF fits
```

Steps 6a and 6d take `--sample full` (default) or `--sample high`. Both samples
share one code path, so the two sets of fits differ only in which sources enter
them, never in how they are fitted or masked.

## The final catalogue

`step7_final_catalogue.py` writes `Final_catalogue.vot` (and a CSV copy): one
row per radio-detected PN, 254 in all, carrying

| Column | Meaning |
|---|---|
| `RP_ID`, `Name`, `hash_id` | identification |
| `RA`, `Dec` | optical position, degrees J2000 |
| `reid_class`, `hash_pnstat` | optical classifications |
| `sep_meerkat_arcsec`, `sep_askap_arcsec` | offset from each survey, separately |
| `S_meerkat_mJy`, `S_askap_mJy` | integrated flux densities |
| `alpha`, `alpha_err` | in-band spectral index, reliable fits only |
| `sp_class` | thermal / uncertain / steep |
| `mir_radio_ratio` | adopted 8 um to radio flux ratio |
| `below_flux_ceiling` | True below 2.2 mJy, False above |
| `radio_class` | classification from the radio criteria alone |

Empty cells are genuine absences — no ASKAP detection, no index that passed the
quality cuts, no 8 um photometry — and are left blank rather than filled with a
sentinel so they cannot be mistaken for measurements.

## Method notes

Three points that matter if you intend to reuse any of this:

**The radio classification never sees the optical class.** That is deliberate,
and it is what allows the radio verdict to be compared against HASH rather than
merely summarised alongside it. An earlier version of the grade did use the
optical class as an input; it was replaced for exactly this reason. If you
modify `step5d_criteria_tally.py`, keep that separation.

**Two masks are applied to the luminosity function, and only two:** bins
brighter than the physical flux ceiling, and bins fainter than the 5-sigma
completeness limit. Masked bins are plotted as open symbols rather than
removed, so the effect of the masking is visible in every figure.

**The mid-infrared criterion uses a Galactic calibration.** In this sample it
does not behave as intended: thermal-spectrum sources sit above the PN band and
steep-spectrum sources within it. This is reported rather than tuned away, and
is discussed at length in the paper. Treat that criterion with care.

`docs/METHOD.md` sets out the numerical choices and their justifications.

## Requirements

Python 3.10 or later, with `numpy`, `scipy`, `astropy` and `matplotlib`
(`requirements.txt`). No compiled extensions and no external services.

## Data

The input catalogues are not distributed here. The MeerKAT LMC data products
are available through the SARAO archive, the ASKAP-EMU catalogue through CASDA
and VizieR, the HASH database at `hashpn.space`, and the Reid & Parker optical
catalogue from VizieR (J/MNRAS/438/2642). Set `PNBASE` to the project root, or
accept the default layout, and place the catalogues under `01_Data/`.

## License

MIT (`LICENSE`). The catalogue and figures the pipeline produces are covered by
the paper's terms; please cite it.

---

O. K. Khattab and M. D. Filipovic, Western Sydney University.
