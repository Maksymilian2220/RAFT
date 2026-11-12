# 16 s Wind / No-Wind OrcaFlex Comparison Against RAFT

Frequency: 0.062500 Hz (16 s).
Wind and no-wind records each contain 6001 samples from 0.0 to 600.0 s.
Summary window: trailing 10 wave cycles (160 s).

See `per_run_comparison_summary.csv` for each OrcaFlex case versus the RAFT response scaled to that run's fitted wave amplitude. `wind_effect_summary.csv` reports wind-minus-no-wind differences; `per_run_harmonic_fits.csv` has all windows.

## Interpretation limits

- The wind-present OrcaFlex run must not be called a matched RAFT case: RAFT is zero-wind/parked and does not model the OrcaFlex NPD wind loading in this setup.
- A separate RAFT singlef run uses 17.1 m/s input with the turbine parked, as requested; RAFT warns that parked mode neglects rotor aerodynamic loads, so this is not a full match to an OrcaFlex NPD wind case.
- Confirm active OrcaFlex wind-load switches, direction conventions, and Primary WF/Dynamic axes and reference point before interpreting differences or signed phase.
- Group replication is limited to one run in each condition; no between-seed variability estimate is available.
- High fit residuals or strong window sensitivity mean the 16 s harmonic is not isolated; inspect the per-run fits before treating differences as RAOs.
