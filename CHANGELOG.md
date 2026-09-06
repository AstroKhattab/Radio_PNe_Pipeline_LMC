# Changelog

## 1.0.0 — 2026-09-07

The release accompanying the submitted manuscript, following a full audit of
the chain. Two of the fixes below change numbers the paper quotes;
`results/summary.md` sets every value against the manuscript and marks the ones
that moved.

### Errors found and fixed

**The sub-threshold aperture correction was not in the chain that produced the
results.** `step3b` measured the 24 visual recoveries in an aperture of one beam
half-width and divided by the full beam solid angle. A Gaussian beam encloses
exactly half its flux inside `r = FWHM/2`, so those integrated fluxes were half
their true value. `step3b2` had been written to correct it, but as a separate
16-arcsec-aperture re-extraction, and `step3d` reached into that file for a
comparison column to patch the flux back up. The correction now happens in
`step3b`, where the photometry is done, with the enclosed fraction computed from
the aperture and the beam rather than assumed; `step3b2` is what its name says,
an independent wide-aperture check that nothing downstream reads. The
uncertainty on the flux now belongs to the same measurement as the flux, which
it previously did not. *Affects:* the median flux of the recoveries
(0.020 → 0.0394 mJy). The fluxes themselves are unchanged from the previous run,
which had the same correction applied by a longer route.

**The spectral classification existed twice, with different cuts.** `step4a`
applied the reliability cuts the paper states (`n_pts ≥ 4`, `Δα < 0.5`,
`χ²ν < 10`) and classified only the fits that passed. `step5b` then re-derived
the classes with `Δα ≤ 0.7`, and assigned thermal / uncertain / steep to fits
that had failed the χ² and Δα cuts as well. Criterion 4a read the second one, so
the score, the four radio classes and both contingency tables were built on
indices the paper says are unusable. This is also why the manuscript's own
sentence does not add up: 128 is the count from `step4a`, and 45 + 35 + 88 = 168
are the counts from `step5b`. `step5b` has been removed, and criterion 4a now
reads `step4a` directly, so the cuts live in one place. *Affects:* the spectral
class counts (45/35/88/86 → 32/30/66/126), criterion 4a (45/88/121 →
32/66/156), the radio classes (71/105/70/8 → 76/103/67/8), both contingency
tables, and the high-confidence PNLF, whose sample grows from 71 to 76.

**Figure 1 had every count typed in as a string**, and had drifted from both the
pipeline and the manuscript (it read 72/104/70 and 34% vs 6% where the paper
read 71/105/70 and 33% vs 8%). It now reads every count out of the tables the
run wrote.

**The flux-ceiling magnitude was a typed constant** (`M_CEIL = -4.52`) repeated
in five scripts, when it follows from the 2.2 mJy ceiling and the adopted
distance (`-4.528`). When the distance was corrected from 49.97 to 49.59 kpc it
should have moved and did not. It is now computed. The bin count is unchanged.
The same applies to the flux at the completeness limit, which was back-derived
from the magnitude and quoted as though it were an input.

**`step5b` wrote its output without `overwrite=True`**, so a second run against
a non-empty output directory failed part way through.

**`step3c` silently dropped every flux column** from the detection catalogue.
The catalogue holds fluxes in Jy and the visual extraction in mJy under
different names, and stacking on the intersection of the two column sets left
the result with no flux at all. The visual rows are now put onto the
catalogue's names and units first. No number changed — nothing downstream was
reading that file's fluxes — but the published table was misleading.

**Panel (b) of the step 5c figure showed the wrong classification.** It plotted
the evidence grade, which used the optical class as an input, in a figure whose
section states that nothing in the classification uses it. It now shows the
radio classification.

**The chance-coincidence estimate was printed and discarded**, and existed only
for ASKAP. Both surveys' estimates, and the offset statistics the paper quotes,
are now written to CSV.

**The comparison of PN and field spectral indices was not in the pipeline at
all** — the numbers in the discussion had no code behind them. It is now
computed in `step4a`, using the same fit routine and the same reliability cuts
for both populations, which is what makes the comparison mean anything. The
values differ from those in the manuscript (+0.49/+0.69/+1.02/+1.17 against
+0.11/+0.45/+0.81/+0.95); the trend with flux is unchanged but the faintest bin
is no longer consistent with no separation.

### Also changed

- `config.yaml` now holds every input path and every fixed parameter, with a
  comment on each. Nothing that sets a threshold, cut, mask or physical
  constant remains inside a script.
- The visual detection list moved out of two hard-coded Python lists and into
  `manual/visual_detections.txt`, with the per-source search-radius overrides.
  Eight of the 32 inspected candidates are not in the HASH parent sample; they
  were previously dropped with a bare warning and are now reported.
- The spectral fit is one vectorised routine in `scripts/_support/spectral.py`,
  used for both the PNe and the 339,128-source field catalogue. Numerically
  identical to the per-source solver it replaces.
- `run_pipeline.py` gained `--only` and `--fast`; `step3a` gained
  `--no-cutouts`, so the tables can be reproduced without spending hours
  rendering ~700 inspection pages.
- Scripts resolve their imports from their own location, so any step can be run
  from any working directory.
- `warnings.filterwarnings("ignore")` narrowed to `AstropyWarning` throughout;
  the bare `except Exception: pass` in the step 1 photometry join replaced with
  masked-array handling.
- `step6a` now reads the criteria tally directly, so the `--sample high` path is
  the same code and the same table as the full-sample path.
- Added `tests/check_bookkeeping.py`, `Makefile`, `environment.yml`, pinned
  `requirements.txt`, and `scripts/make_summary.py`, which writes
  `results/summary.md`.
- Removed `step5a_classify_confidence.py` (the optical-informed evidence grade,
  which the paper does not use) and `scripts/_support/pnlf_models.py` (a
  selection-aware likelihood nothing imported and the paper does not describe).

### Not changed

No threshold, cut, mask or model choice was altered to move a result towards a
value in the manuscript or in an earlier run. Where the pipeline and the
manuscript disagree, `results/summary.md` records the disagreement.
