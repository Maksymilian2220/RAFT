# Pentland FOWT Serviceability Study

## Objective

Use a 20-year ERA5 hourly hindcast for the Pentland Floating Offshore Wind Farm to characterize wind-wave conditions that satisfy two serviceability constraints. Use the filtered sea-state bins as RAFT inputs, then compare selected RAFT cases with OrcaFlex time-domain simulations.

## Site and data source

- Site: Pentland Floating Offshore Wind Farm, approximately 58°37.776′N, 3°50.422′W (58.6296°N, 3.8404°W).
- Dataset: [ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download).
- Target duration: 20 complete calendar years. The proposed period is 2006–2025 inclusive; confirm the desired final year and ensure all requested months are available before analysis.
- Requested fields:
  - 10 m eastward wind component (`10m_u_component_of_wind`). Include the northward component (`10m_v_component_of_wind`) if the wind constraint is intended to mean total wind speed rather than the signed eastward component.
  - Significant height of wind waves (`significant_height_of_wind_waves`).
  - Peak wave period (`peak_wave_period`).
  - Mean wave period (`mean_wave_period`).
  - Mean zero-crossing wave period (`mean_zero_crossing_wave_period`).

The download script currently requests additional wind and wave fields as well. Keep extra fields only where they support analysis or model setup; document the final variable list in output metadata.

## CDS request

The monthly or grouped-month download request uses the following structure. `year`, `month`, `day`, and `time` are populated for each request package by the downloader.

```python
request = {
    'product_type': ['reanalysis'],
    'variable': VARIABLES,
    'year': [str(year)],
    'month': [f'{month:02d}' for month in months],
    'day': [f'{day:02d}' for day in range(1, days + 1)],
    'time': TIMES,
    'area': [59.0, -4.0, 58.75, -3.75],
    'data_format': 'netcdf',
    'download_format': 'unarchived',
}
```

The CDS `area` order is north, west, south, east. This is a bounding box, not a point coordinate. Verify the returned ERA5 latitude/longitude grid point(s) against the site coordinate and record the selected grid point in the analysis metadata. If a single point is intended, choose the nearest grid point explicitly; if a box is intended, define whether grid points receive equal weights or area weights.

## Analysis

- Analyze hourly records at the selected site grid point (or the explicitly agreed regional subset).
- Bin significant wind-wave height and the selected period measure at 0.25 m and 0.25 s, respectively.
- Use the joint Hs-period bins as the events raster for RAFT.
- Preserve the unfiltered valid-record count as the denominator for all discard percentages.
- Report the fraction discarded by the wind constraint, the fraction discarded by the wave-height constraint, their overlap, and the combined fraction discarded by either constraint. Also report the retained fraction. These categories are not additive when records violate both constraints.
- Produce full and retained distributions and counts, with retained probabilities normalized over the retained records. Include the joint distribution, marginals, metadata, and plots.

### Agreed Serviceability Filters

The requested wording is:

1. Discard records with `u10 >= 20 m/s`.
2. Discard records with `Hs >= 2.5 m`.

The agreed wind criterion uses total 10 m wind speed, `sqrt(u10**2 + v10**2)`, rather than the signed eastward component alone. Discard speed >= 20 m/s and wind-wave Hs >= 2.5 m. Retained records satisfy both strict inequalities. Missing wind values are excluded separately and reported independently from threshold violations. No 100 m wind constraint is applied.

## Turbine and simulation setup

- Wind turbine state: idle turbine, rotor speed `omega = 0`, and the specified turbine wind input `u = 0` for the simulation setup. Keep this turbine operating-state input distinct from the ERA5 environmental wind used to select/filter records.
- RAFT: calculate response for every Hs-period event bin remaining after applying both serviceability filters. Record the environmental conditions and response for each bin.
- OrcaFlex: run five cases:
  - The most probable retained sea-state bin.
  - The four most severe cases selected from the RAFT results. Define severity using the response quantity of interest (for example, acceleration) and document tie-breaking and any duplicate-bin handling.
- OrcaFlex duration: 10 minutes for ramp-up/convergence, followed by 10 minutes of data acquisition. Exclude the ramp-up interval from post-processing.
- Post-process OrcaFlex results using Welch power spectral density (PSD) and RMS values. State the output channels, sampling rate, Welch segment/window/overlap settings, and RMS interval in the run record.

## Workflow and deliverables

1. Confirm the 20-year date span, final available month, CDS variable names, filter interpretation, and selected grid point.
2. Download monthly ERA5 NetCDF data for the complete agreed period. Validate every archive/file and verify continuous hourly timestamps with no duplicates or gaps.
3. Run the distribution analysis and report record counts and discard/retention percentages for each constraint and their combined application.
4. Export the filtered event raster and a ranked list of retained bins for RAFT.
5. Run RAFT across all retained bins; rank cases by the agreed severity response.
6. Run the five selected OrcaFlex cases with the specified idle-turbine state and durations.
7. Compare RAFT and OrcaFlex results and publish PSD, RMS, selected sea states, assumptions, and reproducibility metadata.

## Current workspace status

- Downloader: `download_era5.py`; currently configured for a Pentland-area bounding box and monthly/grouped-month CDS retrieval. Verify its `START_YEAR`, `END_YEAR`, `VARIABLES`, and area against the agreed 20-year request before downloading.
- Analysis: `wave_state_distribution.py`; the initial default is deliberately 2006–2020 inclusive (15 years). Set `START_YEAR`, `END_YEAR`, paths, bins, and thresholds at the top, or override years and parameters on the command line. The longer-term target remains 2006–2025; 2024 and 2025 packages are currently missing.
- Run with `python wave_state_distribution.py`. For a short verification run, use `python wave_state_distribution.py --start-year 2006 --end-year 2006 --output-dir wave_results_validation_2006`.
- Pentland archive outputs include `discard_summary.csv`, `raft_event_raster.csv` (occupied retained bins), full/retained probability tables, supplementary period summaries, monthly summaries, plots, and `metadata.json`.
- The supplied download box does not contain the named farm coordinate: its southern edge is 58.75 N, whereas the farm is approximately 58.6296 N. The sampled archive wave point is 59 N, 4 W. Metadata records the selected point and this mismatch; confirm spatial representativeness before treating these results as farm-specific.
