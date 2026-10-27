# Radio_PNe_Pipeline_LMC

Analysis pipeline for a radio-continuum catalogue of planetary nebulae (PNe) in the
Large Magellanic Cloud, built from MeerKAT 1.295 GHz and ASKAP-EMU 888 MHz
observations, and for the first radio planetary nebula luminosity function (PNLF)
of the LMC.

The pipeline reproduces every number, table and figure in the paper from public
catalogues.

## Steps

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `step0_data.py` | Parent sample: HASH true and probable PNe in the LMC (707), with the Reid & Parker (R14) number, optical class and SAGE 8.0 um photometry attached by position. |
| 1 | `step1_crossmatch.py` | Nearest MeerKAT and ASKAP source within 4.5 arcsec; adds the Step 2 measurements. |
| 2 | (no script) | Flux densities measured by hand on the MeerKAT image for PNe with 3 <= S/N < 5, stored in `01_data/manual_fluxes.csv`. |
| 3 | `step3_flux_gate.py` | Rejects detections brighter than 2.2 mJy, the flux of NGC 7027 at the LMC distance. |
| 4 | `step4_spectral_index.py` | Weighted least-squares spectral index over twelve MeerKAT sub-bands and the ASKAP point; thermal, uncertain or steep class, fit tier and points; comparison with the radio field. |
| 5 | `step5_cohen_ratio.py` | Mid-infrared to radio ratio of Cohen et al. (2011); one point for R >= 4.7. |
| 6 | `step6_score.py` | Score (Step 4 + Step 5 points) and level: High >= 3, Possible 2, Weak 0-1. |
| 7 | `step7_pnlf.py` | Radio absolute magnitudes, ten PNLF forms fitted by the Cash statistic and ranked by AIC. |

`02_scripts/common.py` holds the shared helpers. Every input file, column name,
threshold and constant is set in `02_scripts/config.yaml`; no script needs to be
edited to change them.

## Layout

```
01_data/        input catalogues (not distributed, see below) and manual_fluxes.csv
02_scripts/     the pipeline and config.yaml
03_figures/     PDF figures
04_catalogues/  catalogue.fits, one row per parent PN with every derived column
05_tables/      LaTeX tables
07_notes/       numbers.txt, every number quoted in the paper, by step
```

## Input data

Place these files in `01_data/` (names as in `config.yaml`):

| File | Source |
|------|--------|
| `HASH_V163_full.vot` | HASH PN database, VizieR V/163 (Parker et al. 2016) |
| `Warren_PNE_673.vot` | Reid & Parker (2014) LMC PN positions, VizieR J/MNRAS/438/2642 |
| `reid2014_photometry.vot` | Reid & Parker (2014) photometry, VizieR J/MNRAS/438/2642 |
| `MeerKAT_3sigma.vot`, `MeerKAT_Final_5sigma.vot` | MeerKAT LMC broadband and sub-band catalogues (SARAO archive) |
| `LMC_I_mosaic_ch0_beam.fits` | MeerKAT LMC mosaic, used for its footprint only |
| `ASKAP_LMC_888MHz_catalogue_full.vot` | ASKAP-EMU LMC catalogue, Pennock et al. (2021), VizieR J/MNRAS/506/3540 |
| `ASKAP_LMC_888MHz_image.fits` | ASKAP-EMU LMC image (CASDA), used for its footprint only |

## Running

```bash
pip install -r requirements.txt
cd 02_scripts
python run_all.py          # every step
python run_all.py 4 5 6    # only these steps
```

Each step can also be run on its own once the steps before it have run.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`.
