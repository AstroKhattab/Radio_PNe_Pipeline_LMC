# lmc-radio-pne

Analysis pipeline for a radio-continuum catalogue of planetary nebulae in the
Large Magellanic Cloud. Starting from the 679 optically catalogued PNe of Reid &
Parker, it cross-matches and photometers them against the MeerKAT 1.295 GHz LMC
mosaic and the ASKAP-EMU 888 MHz LMC survey, fits a combined radio spectral
index per source from the MeerKAT sub-bands plus the ASKAP point, adds a
mid-infrared/radio diagnostic from Spitzer SAGE 8 um, assigns each source a
transparent evidence grade, and builds from the resulting sample the first
radio planetary nebula luminosity function measured for an external galaxy.

## Pipeline stages

Run in this order. The runner (`scripts/step00_run_pipeline.py`) enforces it.

| # | Stage | Script | Principal input | Principal output |
|---|-------|--------|-----------------|------------------|
| 1 | Methodology schematic | `step00a_make_methodology_workflow.py` | — | `04_Figures/step00a_methodology_workflow.pdf` |
| 2 | Schematic, one-row | `step00b_methodology_workflow_horizontal.py` | — | `04_Figures/step00b_methodology_workflow_horizontal.pdf` |
| 3 | Schematic, two-row | `step00b_methodology_workflow_2row.py` | — | `04_Figures/step00b_methodology_workflow_2row.pdf` |
| 4 | Unified optical catalogue | `step01a_build_reid.py` | `Warren_PNE_673.vot`, `reid2014_photometry.vot` | `step01a_reid_master.vot` (679 rows) |
| 5 | Reid x MeerKAT cross-match | `step02a_crossmatch_meerkat.py` | `step01a_reid_master.vot`, `MeerKAT_3sigma.vot`, `MeerKAT_Final_5sigma.vot` | `step02a_meerkat_matched.vot` (188) |
| 6 | Reid x ASKAP cross-match | `step02b_crossmatch_askap.py` | `step01a_reid_master.vot`, `ASKAP_LMC_888MHz_catalogue_full.vot` | `step02b_askap_matched.vot` (61) |
| 7 | MeerKAT inspection queue | `inspect/step03a_inspect_meerkat.py` | `step02a_meerkat_*.vot`, MeerKAT mosaic | `step03a_meerkat_to_check.vot` (468) |
| 8 | Visual-detection photometry | `step03b_extract_meerkat_visual.py` | `step03a_meerkat_to_check.vot`, MeerKAT mosaic + RMS map | `step03b_meerkat_visual.vot` (32) |
| 9 | Aperture re-extraction | `step03b2_reextract_visual_photometry.py` | same as stage 8 | `step03b2_meerkat_visual_reextracted.vot` (32) |
| 10 | ASKAP inspection queue | `inspect/step03c_inspect_askap.py` | `step02b_askap_matched.vot`, ASKAP image, MCELS, SAGE | `step03c_askap_inspection_queue.vot` |
| 11 | MeerKAT catalogue and sky maps | `step04a_build_meerkat_catalogue_and_multisurvey_figures.py` | stages 5, 7, 8, survey mosaics | `step04a_meerkat_detected.vot` (220) |
| 12 | Multi-survey master and union | `step04b_build_multisurvey_catalogue.py` | stages 4, 5, 6, 8, 11 | `step04b_multisurvey_master.vot` (679), `..._detected_union.vot` (225) |
| 13 | HASH status cross-match | `step04c_crossmatch_hash.py` | `step04b_multisurvey_detected_union.vot`, `HASH_V163_full.vot` | `step04c_hash_status_all.vot` (225) |
| 14 | Combined spectral index | `step05a_fit_spectral_index.py` | `step04c_hash_status_all.vot`, `MeerKAT_Final_5sigma.vot` | `step05a_spectral_index.vot` (225) |
| 15 | Spectral consistency check | `inspect/step05b_inspect_spectral_consistency.py` | `step05a_spectral_index.vot` | `04_Figures/Inspect/step05b_spectral_consistency.pdf` |
| 16 | MIR/radio ratio | `step06a_mir_radio_ratio.py` | `step05a_spectral_index.vot` | `step06a_mir_radio_ratio.vot` (225) |
| 17 | Evidence grading | `step06b_classify_confidence.py` | `step06a_mir_radio_ratio.vot` | `step06b_confidence.vot` (225) |
| 18 | Spectral-class recut | `step05c_recut_spectral_classes.py` | `step05a_spectral_index.vot`, `step06b_confidence.vot` | `step05c_spectral_index_recut.vot` (225) |
| 19 | Empirical PNLF fits | `step07a_fit_pnlf_empirical.py` | `step06b_confidence.vot` | `step07a_pnlf.vot` (225), `step07a_pnlf_empirical_fits.pdf` |
| 20 | Ciardullo PNLF fits | `step07b_fit_pnlf_ciardullo.py` | `step07a_pnlf.vot` | `04_Figures/step07b_pnlf_ciardullo_fits.pdf` |
| 21 | PNLF paper figures | `step07c_make_pnlf_paper_figures.py` | `step07a_pnlf.vot` | `step07c_pnlf_model_overlay.pdf`, `step07c_pnlf_ciardullo_comparison.pdf` |
| 22 | Empirical-fit grid figure | `step07d_pnlf_empirical_grid.py` | `step07a_pnlf.vot` | `step07d_pnlf_empirical_grid_10up.pdf`, `..._4up.pdf` |
| 23 | PNLF consistency check | `inspect/step07d_inspect_pnlf_consistency.py` | `step07a_pnlf.vot` | `04_Figures/Inspect/step07d_pnlf_consistency.pdf` |
| 24 | Selection-aware radio PNLF | `step07e_fit_radio_pnlf_selection_aware.py` | `step04b_multisurvey_master.vot`, MeerKAT RMS map | `step07e_radio_pnlf_selection_catalogue.vot` (679), `..._model_comparison.vot`, `..._summary.vot` |
| 25 | Mask-versus-clean comparison | `step07f_pnlf_mask_vs_clean.py` | `step07a_pnlf.vot`, recut catalogue | `04_Figures/step07f_pnlf_mask_vs_clean.pdf` |
| 26 | PNLF grids by sample | `step07g_pnlf_grids_by_sample.py` | `step07a_pnlf.vot`, recut catalogue | `04_Figures/step07g_pnlf_grid_A4_S*.pdf` |
| 27 | Detection and evidence summary | `step08a_detection_and_evidence.py` | recut catalogue | `04_Figures/step08a_detection_and_evidence.pdf` |

## Headline results

* 679 optically catalogued PN candidates form the parent sample.
* 220 are detected at MeerKAT 1.295 GHz: 188 by catalogue cross-match at
  4.5 arcsec, plus 32 recovered by position-guided visual inspection.
* 61 are associated with an ASKAP-EMU 888 MHz source.
* The union detected by either survey is 225 sources; 56 are detected by both
  and 5 by ASKAP alone.
* 208 of the 225 have a counterpart in HASH V/163.
* 112 sources have a reliable combined spectral index. Of these, 48 are steep
  (alpha < -0.5), 33 thermal (-0.2 <= alpha <= +2.0) and 31 in the intermediate
  uncertain interval.
* Evidence grades over the 225: 28 high-confidence PN, 90 probable PN,
  97 possible PN, 10 questionable / for review.
* The MeerKAT 1.295 GHz sample yields the first radio planetary nebula
  luminosity function measured for an external galaxy.

## Running it

```
pip install -r requirements.txt
export PNBASE=/path/to/the/data/tree     # holds 01_Data/, 03_Outputs/, 04_Figures/
python scripts/step00_run_pipeline.py
```

`PNBASE` defaults to the parent of `scripts/`. The runner uses the interpreter
it was started with, stops at the first failing step, checks each step's
principal product (row count and required column for VOTables), and writes
`00_Context/Logs/pipeline_<timestamp>.log` plus `pipeline_latest.log`.

Options:

| Option | Effect |
|--------|--------|
| `--from-step LABEL` | start at that step instead of the first |
| `--to-step LABEL` | stop after that step |
| `--skip-inspection` | skip the visual-QA steps (`03a`, `03c`, `05b`, `07d-inspect`) |
| `--dry-run` | print the command for each step without running it |

Step labels are those in the pipeline table: `00a`, `00b1`, `00b2`, `01a`,
`02a`, `02b`, `03a`, `03b`, `03b2`, `03c`, `04a`, `04b`, `04c`, `05a`, `05b`,
`06a`, `06b`, `05c`, `07a`, `07b`, `07c`, `07d-grid`, `07d-inspect`, `07e`,
`07f`, `07g`, `08a`. Two scripts are numbered 07d in the original working tree;
the runner distinguishes them as `07d-grid` (the paper's grid figure) and
`07d-inspect` (the consistency check), and neither file is renamed because the
manuscript refers to the current figure names.

Stages 7, 8 and 10 involve human inspection of the cutout pages. The RP_IDs of
the visually confirmed detections are listed near the top of
`scripts/step03b_extract_meerkat_visual.py`; re-running the pipeline end to end
reproduces the published result with that list unchanged.

## Data

The input mosaics and catalogues are **not** in this repository. They total
close to 20 GB and are all publicly available from their original archives.
Place them in `$PNBASE/01_Data/` under the file names the scripts expect.

| Data set | File names used here | Where to get it |
|----------|---------------------|-----------------|
| MeerKAT 1.295 GHz LMC mosaic and source catalogues | `LMC_I_mosaic_ch0_beam.fits`, `LMC_I_mosaic_ch0_rms.fits`, `LMC_I_mosaic_ch0_bkg.fits`, `MeerKAT_3sigma.vot`, `MeerKAT_Final_5sigma.vot` | MeerKAT LMC survey, via the SARAO archive (https://archive.sarao.ac.za) |
| ASKAP-EMU 888 MHz LMC image and catalogue | `ASKAP_LMC_888MHz_image.fits`, `ASKAP_LMC_888MHz_catalogue_full.vot` | CASDA (https://data.csiro.au/domain/casdaObservation); catalogue also at VizieR J/MNRAS/506/3540 (Pennock et al. 2021) |
| Reid & Parker LMC PN catalogue and photometry | `Warren_PNE_673.vot`, `reid2014_photometry.vot`, `Reid_679.vot` | VizieR J/MNRAS/438/2642 (Reid & Parker 2014) |
| HASH PN database, CDS release | `HASH_V163_full.vot`, `HASH_V163_ReadMe.txt` | VizieR V/163; full database at https://hashpn.space |
| Spitzer SAGE IRAC 8.0 um LMC mosaic | `SAGE_LMC_IRAC8.0_2_mosaic.fits` | IRSA (https://irsa.ipac.caltech.edu), SAGE enhanced products |
| MCELS H-alpha, [O III], [S II] LMC mosaics | `LMC.ha.fits`, `LMC.oiii.fits`, `LMC.sii.fits` | Magellanic Cloud Emission Line Survey (MCELS), NOIRLab / CTIO |

`03_Outputs/` and `04_Figures/` are created by the pipeline and are not tracked.

`docs/METHOD.md` records the methodological decisions behind each stage,
including the cross-match radii, the spectral-index quality cuts and the
completeness treatment in the PNLF fits.

## Citation

If you use this code or the catalogue it produces, please cite the paper and
the software record:

```bibtex
@software{khattab_lmc_radio_pne,
  author  = {Khattab, Omar K. and Filipovi\'c, Miroslav D.},
  title   = {lmc-radio-pne: a radio-continuum catalogue and radio PNLF for
             planetary nebulae in the LMC},
  year    = {2026},
  license = {MIT},
  url     = {https://github.com/<account>/lmc-radio-pne},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

See `CITATION.cff` for machine-readable metadata. Replace the placeholder DOI
with the Zenodo DOI minted on release.

## Acknowledgements

This research has made use of the HASH PN database at hashpn.space.

This work uses observations from the MeerKAT telescope, operated by the South
African Radio Astronomy Observatory (SARAO), a facility of the National
Research Foundation, and from the Australian SKA Pathfinder (ASKAP), part of
the Australia Telescope National Facility managed by CSIRO. It also uses data
from the Spitzer Space Telescope, obtained through the NASA/IPAC Infrared
Science Archive, and from the Magellanic Cloud Emission Line Survey. The
VizieR catalogue access tool and the SIMBAD database, CDS Strasbourg, were
used throughout. The analysis relies on Astropy, NumPy, SciPy and Matplotlib.

Work carried out in the Filipovic group, Western Sydney University.
Supervisor: Prof. Miroslav Filipovic.

## Licence

MIT. See `LICENSE`.
