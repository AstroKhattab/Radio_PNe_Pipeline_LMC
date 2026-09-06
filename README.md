# Radio_PNe_Pipeline_LMC

Analysis pipeline for a radio-continuum catalogue of planetary nebulae in the
Large Magellanic Cloud, and for the first radio planetary nebula luminosity
function (PNLF) measured for a galaxy other than our own.

The parent sample is every LMC entry in the HASH PN database (Parker, Bojičić &
Frew 2016), release V/163, classified as a true or probable PN: 707 objects.
Those positions are cross-matched against the MeerKAT 1.295 GHz LMC mosaic and
the ASKAP–EMU 888 MHz survey; inspection of the MeerKAT image recovers nebulae
the source finder missed; three criteria that never use the optical
classification are applied to each detection; and the resulting sample is
binned and fitted with ten luminosity-function models. Every number, table and
figure in the accompanying paper is produced by this code.

Results of the current run, with each value set against the corresponding
number in the manuscript, are in [`results/summary.md`](results/summary.md).

## The six steps

| Step | What it does | Script | Principal outputs |
|---|---|---|---|
| 1 | Parent sample from HASH V/163 | `step1_parent_catalogue.py` | `step1_parent_catalogue.vot`, `tables/step1_catalogue_excerpt.tex` |
| 2a | Cross-match with MeerKAT 1.295 GHz | `step2a_crossmatch_meerkat.py` | `step2a_meerkat_matched.vot`, `figures/step2a_meerkat_offsets.pdf` |
| 2b | Cross-match with ASKAP–EMU 888 MHz | `step2b_crossmatch_askap.py` | `step2b_askap_matched.vot`, `figures/step2b_askap_offsets.pdf` |
| 3a | Split the unmatched, render inspection pages | `step3a_inspect_meerkat.py` | `step3a_meerkat_to_check.vot`, cutout PDFs |
| 3b | Aperture photometry of the visual detections | `step3b_extract_visual.py` | `step3b_meerkat_visual.vot` |
| 3b2 | Wide-aperture check on that photometry | `step3b2_reextract_visual.py` | `step3b2_meerkat_visual_reextracted.vot` |
| 3c | Detection catalogue and sky maps | `step3c_detection_catalogue.py` | `step3c_meerkat_detected.vot`, sky maps, S/N figure |
| 3d | Multi-survey union | `step3d_multisurvey_union.py` | `step3d_multisurvey_master.vot`, detection bookkeeping |
| 4a | Criterion 4a, the spectral index | `step4a_spectral_index.py` | `step4a_spectral_index.vot`, field comparison, SED atlas |
| 4b | Criterion 4b, the MIR/radio ratio | `step4b_mir_radio_ratio.py` | `step4b_mir_radio_ratio.vot` |
| 4c | Criterion 4c, the flux ceiling | `step4c_flux_ceiling.py` | `step4c_flux_ceiling.vot`, the over-ceiling list |
| 5d | Score the criteria; radio against optical | `step5d_criteria_tally.py` | `step5d_criteria_tally.vot`, contingency tables |
| 5c | Detection and evidence summary figure | `step5c_detection_and_evidence.py` | `figures/step5c_detection_and_evidence.pdf` |
| 6a–6h | The radio PNLF, ten models, two samples | `step6*.py` | `step6a_pnlf.vot`, `tables/step6a_model_parameters.tex`, PNLF figures |
| 7 | The published catalogue | `step7_final_catalogue.py` | `Final_catalogue.vot`, `tables/Final_catalogue_excerpt.tex` |

Step 5 runs `5d` before `5c` because the figure shows the classification the
score produces.

## Install

```bash
git clone https://github.com/AstroKhattab/Radio_PNe_Pipeline_LMC.git
cd Radio_PNe_Pipeline_LMC
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

or, with conda, `conda env create -f environment.yml && conda activate lmc-radio-pne`.

Python 3.10 is required: the versions in `requirements.txt` are the ones the
published run used, and numpy 1.24 has no wheels for 3.12 or later. The run in
`results/summary.md` was made on Python 3.10.20; the clean-environment check
before release was made on 3.10.21.

## Get the data

None of the input data is redistributed here. Put the files below in one
directory and point `paths.data` in `config.yaml` at it. Row counts are what
this pipeline expects; a different count means a different release, and the
numbers will not match.

| File | Source | Query | Expected |
|---|---|---|---|
| `HASH_V163_full.vot` | HASH PN database, VizieR **V/163** | `https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=V/163` — retrieve all rows, all columns, as VOTable. Needs `Domain`, `PNstat`, `RAJ2000`, `DEJ2000`, `HASH`, `Name`, `Catalogue`, `MajDiam`. | 11,760 rows (845 with `Domain = LMC`) |
| `Warren_PNE_673.vot` | Reid & Parker (2006, 2014) | VizieR **J/MNRAS/438/2642**, the positional table. Needs `RPRef`, `RAx`, `DECx`, `OBJECT_PROBABILITY`. | 673 rows |
| `reid2014_photometry.vot` | Reid & Parker (2014) photometry | VizieR **J/MNRAS/438/2642**, the photometric table. Needs `_RA`, `_DE`, `__8.0_`, `e__8.0_`. | 605 rows |
| `MeerKAT_3sigma.vot` | MeerKAT LMC mosaic, Aegean broadband 3σ catalogue (Rajabpour et al. 2026) | SARAO archive; see the paper's data-availability statement. | 386,520 rows, ~351 MB |
| `MeerKAT_Final_5sigma.vot` | MeerKAT LMC mosaic, Aegean multi-channel 5σ catalogue | SARAO archive, same source. Needs `ch2`–`ch15` `*_int_flux_1` and `*_err_int_flux`. | 339,128 rows, ~976 MB |
| `ASKAP_LMC_888MHz_catalogue_full.vot` | ASKAP–EMU LMC (Pennock et al. 2021) | VizieR **J/MNRAS/506/3540**, all reliability tiers, or CASDA. Needs `RAJ2000`, `DEJ2000`, `PeakFlux`, `IntFlux`, `e_IntFlux`, `LocalRMS`, `SourceList`, `EMU-ID`, `Flags`. | 54,612 rows |
| `LMC_I_mosaic_ch0_beam.fits` | MeerKAT beam-corrected mosaic, Jy/beam | SARAO archive | 24235 × 36281, ~3.5 GB |
| `LMC_I_mosaic_ch0_rms.fits` | Matching RMS map | SARAO archive | 24235 × 36281, ~3.5 GB |
| `ASKAP_LMC_888MHz_image.fits` | ASKAP–EMU LMC image | CASDA | 31236 × 31668, ~4.0 GB |
| `LMC.ha.fits`, `LMC.oiii.fits` | MCELS Hα and [O III] | NOIRLab / CTIO MCELS release | 16740 × 16740, ~1.1 GB each |

Only the two MeerKAT catalogues, the ASKAP catalogue and the three optical
catalogues are needed for the numbers. The FITS images are needed for the
sub-threshold recovery (step 3b), the survey footprints (step 3c) and the
inspection cutouts; without them, run `make fast` after supplying the mosaic
and RMS map alone.

Record the checksums of your own downloads once you have them:

```bash
shasum -a 256 /path/to/data/*.vot > data_checksums.sha256
```

## Run it

```bash
make all             # everything, then the bookkeeping checks
make fast            # the same without the ~700 inspection cutout pages
make list            # the steps and their keys
make step-4a         # one step
make fig-pnlf        # one figure
make check           # the bookkeeping identities, against existing outputs
make summary         # rewrite results/summary.md
```

`make all` takes about six minutes end to end on a laptop, roughly half of it
in step 3a rendering ~700 inspection cutouts from the 3.5 GB mosaics. `make
fast` skips those and takes about three. The mosaics are read memory-mapped, so
the peak resident set stays under a gigabyte.

Equivalently, without make:

```bash
python3 scripts/run_pipeline.py            # all
python3 scripts/run_pipeline.py --only 6a  # one step
python3 scripts/run_pipeline.py --from 4a  # resume
```

## Configuration

`config.yaml` holds every input path and every fixed parameter: the acceptance
radius, the flux ceiling, the spectral-index cuts, the class boundaries, the
IRAC zero point, the adopted distance, the bin width, the completeness limit
and the fit seed. Each carries a comment saying what it is. No threshold, cut,
mask or physical constant is written inside a script.

To run against a different data directory or a scratch output directory
without editing the repository's own file, copy `config.yaml`, edit the copy,
and point `PNLF_CONFIG` at it:

```bash
PNLF_CONFIG=/path/to/my_config.yaml make all
```

Quantities that *follow* from those parameters are computed, never typed: the
flux ceiling as a magnitude, the flux at the completeness limit, and the
fraction of a point source the photometric aperture encloses. They are derived
in `scripts/_support/config.py` and reported in `results/summary.md`.

The one manual input to the analysis is
[`manual/visual_detections.txt`](manual/visual_detections.txt): the RP numbers
read off the step 3a inspection pages as showing emission at the optical
position. It is a text file with comments rather than a list inside a script,
so the decision can be reviewed and changed without editing code.

## Output layout

```
outputs/          every table the pipeline writes (VOTable and CSV); not tracked
figures/          paper figures, at the file names the manuscript uses; tracked
tables/           machine-written LaTeX tables, likewise; tracked
results/summary.md   every number, against the manuscript's value; tracked
04_Inspect/       per-source SED and cutout pages, quality control; not tracked
```

`figures/` and `tables/` can be copied straight into the Overleaf project.

## Checks

`tests/check_bookkeeping.py` asserts the identities that must hold if the chain
is consistent — the four detection categories summing to the parent sample, each
step reading the number of rows the previous one wrote, the spectral categories
summing to the detections, the criteria and score tallies summing to the
detections, the PNLF sample being exactly what the ceiling and the frequency
leave, and the derived constants equalling what they are derived from. It also
rebuilds the radio classification from the three criterion columns alone and
checks it reproduces the published verdict, which is the test of the claim that
the score uses no optical information.

It exits non-zero on any failure and is run automatically by `make all`.

## Citing this work

> Khattab O. K., Filipović M. D., Smeaton Z. J., Crawford E. J., *A
> Radio-Continuum Catalogue of Planetary Nebulae in the Large Magellanic Cloud
> and the First Radio Planetary Nebula Luminosity Function* (submitted).

`CITATION.cff` carries the machine-readable form. Please cite the paper rather
than the repository alone: the paper documents the calibration choices and the
selection function, and a result quoted without them is hard to interpret.

## Licence

MIT, see [LICENSE](LICENSE). The input catalogues carry their own terms; cite
HASH, Reid & Parker, Pennock et al. and the MeerKAT LMC survey as appropriate.

## Changelog

See [CHANGELOG.md](CHANGELOG.md). The 1.0.0 release follows an audit that found
two errors affecting published numbers: the sub-threshold aperture correction
was documented but not applied in the chain that fed the results, and the
spectral classification existed in two places with different reliability cuts,
so criterion 4a was scored on indices the paper states are unusable. Both are
fixed; `results/summary.md` marks every number that moved.
