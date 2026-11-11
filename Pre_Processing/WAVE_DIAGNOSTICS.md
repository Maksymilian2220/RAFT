# Wave Diagnostics

Run the diagnostic exporter after preparing the cache, then regenerate plots:

```powershell
python Pre_Processing/wave_diagnostics.py --cache-file Pre_Processing/era5_processed/era5_point_2006_2025.parquet --results-dir Pre_Processing/wave_results_2006_2025
python Pre_Processing/plot_wave_results.py --results-dir Pre_Processing/wave_results_2006_2025
```

The exporter does not download data, reread ERA5 archives, or change the
serviceability filters. Existing base analysis tables are unchanged. It writes:

- `wave_height_comparison.csv`: timestamp-aligned combined, wind-wave and swell
  heights, the total-minus-wind height increment, energy-based swell proxy,
  partition energy residual and concurrent 10 m wind speed.
- `wave_height_cdf.csv`: empirical height CDFs and exceedance probabilities,
  separately for available samples and common three-height samples.
- `directional_wave_probability.csv`: occupied 0.25 m / 0.25 s / 30 degree bins.
  Probabilities sum to one per source over its finite paired observations.
- `wave_variable_coverage.csv`: monthly finite-hour coverage for each variable.
- `wave_quality_summary.json`: coverage windows and height consistency checks.

Plots include combined-versus-wind height comparison, cumulative/exceedance
curves, partition energy checks, wind-versus-wave density, monthly coverage and
a 3D Hs/Tp/direction scatter with joint probability as colour (four variables).
The main Hs/Tp plots share high-contrast turbo/logarithmic styling; the full and
retained panels share colour normalization and mask zero-probability cells.

## Interpretation

Significant heights approximately combine through spectral variance:
`swh^2 ~= shww^2 + shts^2`. Therefore `swh - shww` is NOT swell Hs.
`sqrt(max(swh^2 - shww^2, 0))` is an energy-based proxy, not a downloaded swell
measurement. Negative differences are counted before clipping the proxy.
Partition closure can be affected by archive precision, spatial interpolation
and processing; a small residual does not prove that timestamps, frames or the
selected location are correct. Verify those independently.

The current 2006-2025 cache has 175,320 paired combined/wind heights. Direct
swell height exists only for July-December 2025 (4,416 hours). Wave directions
have 42,408 samples with gaps. Do not compare unequal-window CDFs as a clean
partition effect: use the common-sample panel. These are height occurrence
distributions, not frequency-domain wave spectra. No literature distribution
has been calibrated or overlaid; matching location, season, height definition
and sampling is needed before quantitative comparison.

Directions are wave directions FROM north, not inferred from wind. The Tp axis
is ERA5 `pp1d`, the combined-sea peak period, including in the wind-wave panel;
it is not a measured wind-wave partition peak period. 0 and 360 degrees refer
to the same direction. Directional probabilities are conditional on available
direction data, not the complete 20-year record. The selected grid point is
59 N, 4 W, not the nominal farm coordinate.

## JONSWAP Gamma

Bulk Hs and Tp and ERA5 `wsp` do not identify JONSWAP gamma. Supply measured
surface-elevation variance spectra, not platform-response PSDs, as CSV:

```csv
spectrum_id,frequency_hz,psd_m2_per_hz
record_001,0.04,0.03
record_001,0.05,0.08
```

Each spectrum needs at least 12 unique positive frequency bins, nonnegative
finite density, an interior peak, and adequate coverage around the peak. Add
`--spectra-file path/to/spectra.csv` to the exporter command. It fits free
amplitude, peak frequency and gamma in [1,10], with standard sigma 0.07/0.09,
using log-density least squares around 0.5-3 times the observed peak frequency.
This is a numerical fit to the JONSWAP parametric form, not a validated site
spectral model. Results are `jonswap_fits.csv`, `jonswap_summary.json` and
`jonswap_spectral_curves.csv`. The plotting command then creates
`jonswap_fitting.png` with measured/fitted spectra and a gamma histogram.
Inspect log-PSD residuals, gamma boundary flags, fit-window dependence, noise,
frequency resolution and multimodal seas before interpreting the arithmetic
mean or median gamma. Fit stationary wind-sea spectra separately from swell
where possible; there is no universal defensible gamma for a mixed sea.
The fitted peak-frequency search is limited to +/-15% of the observed peak.
Synthetic recovery verifies the implementation, not real measurement quality.

Literature starting points:
- Hasselmann et al. (1973), Measurements of Wind-Wave Growth and Swell Decay
  during the Joint North Sea Wave Project (JONSWAP), Deutsche Hydrographische
  Zeitschrift, Supplement A 12.
- ECMWF ERA5 data documentation (ocean-wave parameters and conventions):
  https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

## Constructed Combined-Sea Spectra

When measured elevation spectra are unavailable, construct Hs/Tp-matched curves
from the existing cache, independently of the measured-fitting workflow:

```powershell
python Pre_Processing/jonswap_site_spectra.py --cache-file Pre_Processing/era5_processed/era5_point_2006_2025.parquet --start-year 2006 --end-year 2025 --results-dir Pre_Processing/wave_results_2006_2025
python Pre_Processing/plot_wave_results.py --results-dir Pre_Processing/wave_results_2006_2025
python Pre_Processing/jonswap_site_spectra.py --self-test
```

This writes `jonswap_constructed_states.csv`, `jonswap_constructed_spectra.csv`,
`jonswap_iec_gamma_distribution.csv` and `jonswap_constructed_metadata.json`.
Plotting optionally adds `plots/jonswap_constructed_spectra.png` and
`plots/jonswap_iec_gamma_distribution.png`; legacy plotting is unchanged when
the constructed tables are absent. No measured-fit outputs are overwritten.
Run with the preprocessing environment, which must also support importing
`raft.helpers.JONSWAP` and its dependencies. No solver calculation is run.

Inputs are combined `swh` and combined peak period `pp1d`, not the wind-wave
event raster. All finite nonnegative Hs / finite positive Tp pairs are included
without wind or serviceability filtering. Invalid pairs and calm hours are
counted in metadata. Occupied half-open 0.25 m / 0.25 s bins start at zero.
The mode is the most populated bin; the other selectors use empirical combined
Hs p50/p90/p99 targets and choose the nearest occupied Hs center, then largest
count, then ascending Hs/Tp bin order. Curves use the selected bin centers;
targets, actual centers, counts and valid-pair occurrence weights are exported.
Selectors can refer to the same bin. These four selectors are not a complete
sea-state probability distribution, and their occurrence weights need not sum
to one.

The default gamma follows RAFT's IEC 61400-3 / FAST v7 rule with
`r = Tp/sqrt(Hs)`: gamma is 5 for `r <= 3.6`, 1 for `r >= 5`, and
`exp(5.75 - 1.15*r)` otherwise. Its small boundary discontinuities are preserved,
including interior values slightly greater than 5 near 3.6. Sensitivity curves
assume gamma 1, 3.3 and 5. Gamma summary mean, median and histogram are computed
from actual hourly pairs with one weight per positive-Hs hour, not bin-center
gammas or an unweighted mean of bins. Zero-Hs states have zero spectrum and no
defined gamma; they are excluded from gamma statistics but retained in pair/bin
counts. Bin-center curves remain bin-center approximations even for bins
containing calm hours.

Frequencies are strictly positive Hz. The unmodified RAFT helper receives
`omega = 2*pi*f` in rad/s. Although its docstring labels the output per Hz,
its formula contains `1/(2*pi)`; for gamma 1 its raw integral over Hz is
`Hs^2/(16*2*pi)`. We do not silently apply an additional PSD conversion.
Instead every exported curve is numerically normalized in Hz so
`integral S(f) df = Hs^2/16`; the raw integral, normalization factor and
reconstructed Hs are recorded for every curve. An adaptive logarithmic grid
checks a conservative omitted-tail bound and refinement error below `1e-5`,
with local peak spacing at most `0.002*fp`. Plot views show up to `3*fp`, while
the CSV retains the full converged grid. For a separate PSD expressed per rad/s,
the density transformation would be `S_omega = S_f/(2*pi)`.

These are constructed, assumed-shape spectra, **not fitted site gamma** or
measured spectra. The 59 N, 4 W point remains an accepted regional proxy,
approximately 42 km from the farm. Combined wind/swell seas can be multimodal;
a unimodal JONSWAP approximation can miss that structure. No average spectrum
is claimed to be a realizable representative sea. Genuine gamma fitting still
requires the separate measured elevation-spectrum workflow above.

## Validation

```powershell
python Pre_Processing/test_wave_diagnostics.py
python Pre_Processing/jonswap_site_spectra.py --self-test
python Pre_Processing/plot_wave_results.py --self-test
```
