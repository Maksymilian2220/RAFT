# Hs Difference vs Peak Period Probability Plot

## Goal

Add a joint-probability plot of hourly combined significant wave height minus wind-wave significant height (`swh - shww`) versus the combined-sea peak period (`pp1d`). This is a descriptive height increment conditioned on combined-sea peak period. Do not label or interpret it as physical swell significant height or swell peak period.

## Agreed choices

- Use the direct difference already saved in `swh_minus_shww_m`; do not substitute `shts` or the energy-based proxy `sqrt(max(swh^2 - shww^2, 0))`.
- Use the full saved CSV period, 2006-2023.
- Use symmetric 0.1 m difference bins centered around zero and 0.25 s peak-period bins.
- Include rows with finite difference and peak period; normalize bin counts by the total number of those paired rows.
- Label the plot explicitly: difference is `swh - shww`; period is combined-sea `pp1d`.
- Keep existing height comparison plots unchanged.
- Use the saved CSV only. No ERA5 downloads or distribution-analysis reruns are needed.

## Data caveat

`Pre_Processing/wave_swell_wind_comparison_results/wave_state_data.csv` currently contains 157,776 rows from 2006-01-01 through 2023-12-31, with finite `swh_minus_shww_m` and `peak_period_s` values. Its sibling `metadata.json` still reports 2006-2020 and 131,496 observations. For this plot, use the CSV timestamps/count rather than the stale metadata period. The current CSV's `shts_m` column is blank throughout; that does not prevent plotting the direct difference.

## Implementation steps

1. In `Pre_Processing/plot_wave_results.py`, add an optional plotting function that reads `wave_state_data.csv`, validates the required columns, and filters to finite `swh_minus_shww_m` and `peak_period_s` values.
2. Build a signed difference-by-period bin-count matrix with symmetric difference edges around zero and 0.25 s period bins. Normalize by all retained paired observations and mask empty cells in the plot.
3. Plot the joint bin probabilities with a probability color scale. Include the paired-hour count and date coverage when practical. Make the labels and title clear that the plotted difference is not physical swell Hs and that `pp1d` is the combined-sea period.
4. Save the image as `hs_difference_peak_period_joint_probability.png` under the selected results directory's `plots/` folder. Add the function to `plot_results()`; missing optional input should skip the plot consistently with other optional plots.
5. Add focused self-tests for negative, zero, and positive difference bins; matrix orientation; probabilities summing to one; and PNG creation. Add coverage for missing or invalid optional data behavior as appropriate.

## Relevant files

- `Pre_Processing/plot_wave_results.py` - plotting helpers, saved-results dispatch, embedded plot self-tests.
- `Pre_Processing/wave_swell_wind_comparison_results/wave_state_data.csv` - saved paired hourly values for 2006-2023.
- `Pre_Processing/wave_swell_wind_comparison_results/metadata.json` - stale period metadata; do not use it to truncate this plot's data.
- `Pre_Processing/PLAN.md` - study conventions, including the 0.25 s period-bin width.

## Validation

From the repository root in PowerShell, using the preprocessing virtual environment:

```powershell
& ".\Pre_Processing\.venv\Scripts\python.exe" Pre_Processing/plot_wave_results.py --self-test
& ".\Pre_Processing\.venv\Scripts\python.exe" Pre_Processing/plot_wave_results.py --results-dir Pre_Processing/wave_swell_wind_comparison_results
Get-Item Pre_Processing\wave_swell_wind_comparison_results\plots\hs_difference_peak_period_joint_probability.png | Select-Object FullName, Length, LastWriteTime
```

Confirm the image is non-empty, the difference axis spans both sides of zero, the finite paired count reflects the 2006-2023 CSV, and the plotted probabilities sum to one. Visually inspect the labels and probability color scale. Report the physical interpretation caveat with the result.
