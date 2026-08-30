# lmc-radio-pne

Analysis pipeline for a radio-continuum catalogue of planetary nebulae in the
Large Magellanic Cloud, and for the first radio planetary nebula luminosity
function measured for a galaxy other than our own.

The parent sample is taken from the HASH PN database (Parker, Bojicic & Frew
2016), release V/163: every LMC entry classified as a true or probable PN.
Those positions are cross-matched against the MeerKAT 1.295 GHz LMC mosaic and
the ASKAP-EMU 888 MHz survey, forced photometry recovers sources the catalogues
missed, three independent criteria are applied to each detection, and the
surviving sample is fitted with a selection-aware luminosity function.

## Pipeline

Run in order; `scripts/run_pipeline.py` enforces it and stops at the first
failure. Each step writes a table, a small summary in CSV, and a figure.

| Step | What it does | Script | Output |
|---|---|---|---|
| 1 | Parent sample from HASH V/163 | `step01b_build_parent_hash.py` | `step01b_parent_sample.vot` (707) |
| 2 | MeerKAT and ASKAP cross-match | `step02c_crossmatch_surveys.py` | `step02c_radio_crossmatch.vot` (707) |
| 3a | Forced photometry on the misses | `step03d_forced_photometry.py` | `step03d_forced_photometry.vot` (477) |
| 3b | Vetting those candidates | `step03e_vet_candidates.py` | `step03e_visual_detections.vot` |
| 4a | One flux scale for everything | `step04a_detection_catalogue.py` | `step04a_radio_detections.vot` (244) |
| 4b | Criterion 1: thermal spectrum | `step04b_spectral_index.py` | `step04b_spectral_index.vot` |
| 4c | Criterion 2: MIR/radio ratio | `step04c_mir_radio_ratio.py` | `step04c_mir_radio_ratio.vot` |
| 4d | Criterion 3: flux ceiling | `step04d_flux_ceiling.py` | `step04d_flux_ceiling.vot` |
| 4e | The criteria combined | `step04e_classify.py` | `step04e_classified.vot` |
| 4f | Radio verdict against optical | `step04f_compare_optical.py` | `step04f_contingency.csv` |
| 5 | Luminosity function, two samples | `step05_pnlf.py` | `step05_pnlf_models.csv` |

`scripts/_support/pnlf_models.py` holds the luminosity-function models and the
selection-aware likelihood, so that both samples in Step 5 are fitted by
identical code.

## Results

**Sample.** 707 LMC PNe in HASH V/163 (545 true, 162 probable). A further 138
LMC entries carry another classification - 105 star clusters, 8 supernova
remnants, 7 H II regions and 18 others - and are excluded.

**Detections.** 225 have a MeerKAT 1.295 GHz counterpart within 4.5 arcsec and
82 an ASKAP-EMU counterpart; 77 are seen by both and 5 by ASKAP alone, giving
230 catalogue detections. Displacing every position by 60 arcsec and repeating
the match gives 8.8 +/- 3.0 chance coincidences for MeerKAT and 1.2 +/- 1.1 for
ASKAP, so under 4 per cent of the matches are accidental.

**Forced photometry.** Of the 477 undetected parent PNe, 459 lie inside the
mosaic. Measuring a flux at all of them and keeping those above 5 sigma in both
peak and aperture significance yields 25 candidates - against 11 at control
positions offset by 60 arcsec, so this selection is only about 56 per cent
pure. Rejecting candidates whose flux is still growing at a 24 arcsec aperture,
and those within 60 arcsec of a source brighter than 5 mJy/beam, leaves 14
tentative detections. **244 radio detections in total.**

**Criteria.** Thermal spectral index: 31 pass, 66 fail, 147 not measurable.
MIR/radio ratio: 92 pass, 34 fail, 118 not measurable. Flux ceiling: 237 pass,
7 fail. Scored out of what could be measured for each source: 78 confirmed,
98 probable, 61 possible, 7 rejected.

**Against the optical classification.** Of the 191 detections HASH calls true
PNe, 76 are confirmed by the radio criteria and 1 is rejected. Of the 53 it
calls probable, 2 are confirmed and 6 rejected. The radio and optical verdicts
were derived independently and agree.

**Luminosity function.** Fitted for the full radio sample (235 after masking)
and for the confirmed sample (77), each masked only at its own 5 sigma
completeness limit and for the sources above the physical flux ceiling. Seven
functional forms are compared by AICc, BIC and a probability integral
transform. The two samples do not choose the same model, which is itself the
result and is discussed in the paper.

## Running it

```
pip install -r requirements.txt
python scripts/run_pipeline.py
python scripts/run_pipeline.py --from 4b      # restart part way through
```

Paths are resolved relative to `~/Desktop/Research/PN LMC Paper`, which must
hold `01_Data/`, `03_Outputs/` and `04_Figures/`. The raw MeerKAT and ASKAP
images are not distributed here; they come from the surveys cited in the paper.

## Data

| File | Source |
|---|---|
| `HASH_V163_full.vot` | HASH PN database, VizieR V/163 |
| `MeerKAT_3sigma.vot`, `MeerKAT_Final_5sigma.vot` | MeerKAT 1.295 GHz LMC mosaic, Cotton et al. (2026) |
| `ASKAP_LMC_888MHz_catalogue_full.vot` | ASKAP-EMU, Pennock et al. (2021) |
| `LMC_I_mosaic_ch0_*.fits` | MeerKAT total intensity, RMS and background maps |
| `reid2014_photometry.vot` | Reid & Parker (2014) multiwavelength photometry |
| `Warren_PNE_673.vot` | Reid & Parker optical classifications |

## Citing

See `CITATION.cff`. The paper this supports is in preparation.

## Licence

MIT. See `LICENSE`.
