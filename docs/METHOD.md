# Method

Why the analysis is done the way it is. **No counts appear in this file.** The
numbers this pipeline produces are in [`results/summary.md`](../results/summary.md),
which is written by the code; an earlier version of this document restated them
by hand and had drifted from both the pipeline and the manuscript by the time
anyone noticed. Every threshold named below is set in
[`config.yaml`](../config.yaml).

## Step 1 — the parent sample

The parent sample is every HASH V/163 entry whose domain flag is `LMC` and whose
`PNstat` is `T` or `P`. Taking it from HASH rather than assembling it by hand
does two things: the selection becomes a single reproducible query against a
public catalogue, and the classifications are the ones the community curates
rather than ones we impose. Entries carrying any other status are written out
separately so the paper can tabulate what was set aside and why.

Two ancillary catalogues are matched in at the same radius as the radio
matching, purely to carry extra columns: the R14 list supplies the RP running
number and the optical class, and the R14 photometric compilation supplies the
IRAC 8 µm magnitude. Neither adds nor removes a source. Where HASH lists an
object R14 does not, the optical class is mapped from the HASH status.

## Step 2 — radio cross-matching

Both surveys are matched against the same parent list, so every PN is
unambiguously detected in MeerKAT, in ASKAP, in both, or in neither, and those
four numbers sum to the parent sample. The acceptance radius is the same for
both: three MeerKAT pixels, roughly half the synthesised beam. ASKAP
counterparts wider than that but inside the outer search radius are recorded
with a review flag and are not counted as detections.

Matching is nearest-neighbour within the acceptance radius, and the pipeline
asserts that no radio source is claimed by two PNe. Offsets are computed with
`spherical_offsets_to`, so ∆α carries the cos δ factor.

Positional offsets are used to check the matching, not to classify. A tight
distribution centred on zero shows that the two catalogues share an astrometric
frame; it says nothing about the nature of any individual object. The number of
matches expected by chance is computed from each catalogue's surface density
and written out alongside.

## Step 3 — inspection and quality assurance

A source finder works on a fixed island threshold, so a real but faint nebula
below that threshold never enters the catalogue and would be counted as a
non-detection. The parent PNe left unmatched by step 2 are split into those
inside and outside the MeerKAT mosaic, using the mosaic's own finite-pixel mask,
and each of those inside is examined on the image itself beside MCELS Hα and
[O III].

Where emission is present at the optical position, aperture photometry is
extracted. **The aperture radius is one beam half-width, and the flux is divided
by the fraction of an unresolved source that aperture encloses.** For a Gaussian
beam that fraction is `1 - exp(-4 ln2 r²/FWHM²)`, which at `r = FWHM/2` is
exactly one half; without the correction every one of these fluxes is half its
true value. The fraction is computed from the configured aperture and beam, not
assumed. The local noise comes from an annulus well outside the aperture.

`step3b2` repeats the measurement in a 2 × FWHM aperture with a local background
subtracted, where no enclosed-flux correction is needed at all. It is a check on
the correction, not a replacement for it: at these flux levels the wide aperture
is noise dominated, which is why the analysis uses the narrow corrected one.
Nothing downstream reads `step3b2`.

These recoveries are the least secure detections in the sample and are flagged
separately throughout.

## Step 4 — three independent criteria

Three questions are asked of every detection. Each returns pass, fail, or
nothing to say, and **none of them uses the optical classification.**

### 4a — the in-band spectral index

An optically thin ionised nebula has a nearly flat radio spectrum, rising
towards `α = +2` where it becomes optically thick. The dominant contaminant at
these flux levels is the background radio galaxy population, synchrotron
dominated near `α = -0.7`. The index is therefore the most direct test of
whether a radio source at the position of an optical PN is in fact the PN.

`log10 S = α log10 ν + c` is fitted by weighted least squares over the usable
MeerKAT sub-bands together with the ASKAP point **where the source is detected
by ASKAP**, with weights propagated from the flux uncertainties into log space
as `σ_logS = σ_S / (S ln10)`. The MeerKAT broadband point is drawn on the SED
pages but not fitted: it comes from the same data as the sub-bands and would
double-count the band.

An index is used only where `n_pts`, `Δα` and `χ²ν` all pass the configured cuts.
Appendix A of the paper sets out why each. **These cuts are applied in exactly
one place, in `step4a`.** They used to be applied a second time downstream with
a looser `Δα`, and with classes assigned to fits that had failed them, which is
how criterion 4a came to be scored on indices the paper states are unusable.

Surviving indices are classified thermal, uncertain or steep by the configured
boundaries. A fit that failed a cut is recorded with the reason and counts as
neither thermal nor steep; it enters step 5 as *not measurable* for criterion
4a. The four categories — the three classes plus no-reliable-index — are
mutually exclusive and sum to the number of detections, which
`tests/check_bookkeeping.py` asserts.

The same fit and the same cuts are applied to the whole MeerKAT 5σ catalogue,
which at these flux levels is overwhelmingly background galaxies, and the two
populations are compared bin by bin in flux density. Both populations go through
the same routine in `_support/spectral.py`; the comparison would mean nothing
otherwise.

### 4b — the mid-infrared to radio ratio

Dust in an ionised nebula reprocesses stellar radiation into the mid-infrared,
and the MIR and radio fluxes scale together. Cohen et al. (2011) turned this
into a diagnostic, combining MGPS-2 at 843 MHz with NVSS at 1.4 GHz and treating
the pair as one measurement near 1 GHz. They did **not** define it at 1.295 GHz.

The ratio is therefore computed at the frequencies actually measured, rather
than scaling a flux to a frequency at which there is no data: the ASKAP 888 MHz
flux where it exists, otherwise the MeerKAT broadband flux. 888 MHz is the
closer of the two to the effective frequency of the Galactic calibration and is
preferred for that reason. Every source records which frequency its ratio came
from. The 8 µm magnitude is converted with the IRAC zero point in `config.yaml`.

The screen is kept as published so the numbers stay comparable with the Galactic
literature. The calibration is Galactic, and measurements in the Magellanic
Clouds place the same ratio higher; the paper reports that the screen does not
behave as intended for this sample rather than adjusting it.

### 4c — the radio flux ceiling

The radio luminosity of a PN is set by the ionised mass it can hold, which has
an upper limit. Filipović et al. (2009) placed NGC 7027 at the distance of the
Magellanic Clouds and adopted a working ceiling allowing for objects somewhat
more luminous. Because PNe are optically thin at these frequencies and their
spectra are close to flat, a limit derived at 4.8 GHz carries to 1.295 GHz
essentially unchanged, which is the only reason one number can serve at both.

The ceiling is applied to the integrated flux density at 1.295 GHz. A source
with no usable integrated flux is *not measurable*, which is not the same as
passing.

## Step 5 — scoring the criteria

Each criterion returns pass, fail, or nothing to say, and the third answer is
not a failure. A nebula with no 8 µm photometry and a spectrum too faint to fit
is not a weaker candidate than one with three measurements; it is an object
about which less is known. Scoring out of a fixed total of three would penalise
it for the state of the ancillary data rather than for anything about the
source. Each detection is therefore scored out of the criteria that could
actually be evaluated for it, and both numbers are carried.

| Class | Definition |
|---|---|
| `Radio_Rejected` | fails a hard physical test: above the flux ceiling, or radio position further from the optical one than the acceptance radius |
| `Radio_High` | every measurable criterion passed, with at least two measurable |
| `Radio_Possible` | more than half of the measurable criteria passed |
| `Radio_Weak` | half or fewer passed, or nothing was measurable |

The class names deliberately avoid the vocabulary the optical catalogues use.
Rejections are only the two hard physical tests; a steep spectrum or an extreme
MIR ratio is a failed criterion, not a rejection.

Nothing in this score uses the optical classification. That is the entire point:
a score that took the optical class as an input could not then be used to test
it. `tests/check_bookkeeping.py` rebuilds the verdict from the three criterion
columns alone and checks it reproduces the published one, so the claim is tested
rather than asserted.

## Step 6 — the luminosity function

Integrated flux densities are converted to luminosities with `L = 4π d² S` at
the adopted distance and expressed as `M_radio = -2.5 log10(L / L_ref)`. Only
MeerKAT 1.295 GHz fluxes are used; the ASKAP-only detections have no measurement
at the reference frequency and are excluded, because mixing 888 MHz and
1.295 GHz fluxes would combine different frequencies, beams, sensitivities and
completeness functions.

Counts are binned at the configured width. **Two masks are applied and only
two:** bins brighter than the flux ceiling of step 4c, and bins fainter than the
completeness limit. The ceiling magnitude is computed from the ceiling flux and
the distance, so it cannot drift away from either. Masked bins are drawn as open
symbols so the masking is visible rather than implied.

An earlier version excluded a further range of bright bins on the grounds that
they looked anomalous. That exclusion has been removed: the ranking of the
fitted models depended on it, which is not a property a result should have.
`step6e` shows the three treatments side by side.

Ten functional forms are fitted by Poisson-weighted least squares, optimised by
differential evolution with a pinned seed and refined by Levenberg–Marquardt,
and compared by AIC computed as `χ² + 2k` with `k` the number of free
parameters — consistently for all ten, on the same bins. Among them is the
canonical Ciardullo (1989) form with its shape parameters held at their optical
values, and a version with both free.

The high-confidence run uses the same script, the same binning, the same masks
and the same ten models; only the sample differs, which is what makes the two
comparable.
