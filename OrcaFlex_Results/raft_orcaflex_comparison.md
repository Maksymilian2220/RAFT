# RAFT and OrcaFlex Motion Comparison

## Case conditions

| Parameter | RAFT | OrcaFlex analysis |
|---|---:|---:|
| Wind speed at 10 m | 3.93 m/s | 3.93 m/s |
| Turbine status | Parked | Parked |
| Wave period | 9.875 s | 9.875 s |
| Wave height | 0.125 m | 0.125 m (treated as crest-to-trough height) |
| Wave model | JONSWAP | Regular-wave harmonic reference |

## Six-DOF statistics

Values are shown as reported by each tool. RAFT's `RMS` is not assumed to be equivalent to OrcaFlex's standard deviation.

| DOF | Unit | RAFT average | RAFT RMS | RAFT min to max | OrcaFlex mean | OrcaFlex std | OrcaFlex min to max |
|---|---|---:|---:|---:|---:|---:|---:|
| Surge | m | 1.72e-2 | 9.71e-3 | -1.19e-2 to 4.64e-2 | 2.11e-3 | 1.63e-2 | -2.20e-2 to 2.64e-2 |
| Sway | m | -6.30e-11 | 1.40e-16 | -6.30e-11 to -6.30e-11 | -1.09e-2 | 7.67e-3 | -2.41e-2 to 2.96e-3 |
| Heave | m | 1.81e-1 | 8.24e-3 | 1.56e-1 to 2.05e-1 | 2.60e-2 | 2.26e-2 | -1.80e-2 to 7.17e-2 |
| Roll | deg | -6.78e-13 | 6.75e-17 | -6.78e-13 to -6.78e-13 | 4.34e-3 | 2.48e-3 | 5.46e-4 to 8.18e-3 |
| Pitch | deg | 1.12e-3 | 3.65e-3 | -9.83e-3 to 1.21e-2 | -7.09e-4 | 1.32e-2 | -2.00e-2 to 1.85e-2 |
| Yaw | deg | -1.15e-14 | 3.90e-16 | -1.27e-14 to -1.04e-14 | -6.76e-3 | 2.87e-3 | -1.12e-2 to -2.39e-3 |

## RAFT equilibrium offsets

| DOF | Equilibrium offset |
|---|---:|
| Surge | 1.72e-2 m |
| Sway | -6.30e-11 m |
| Heave | 1.81e-1 m |
| Roll | -1.18e-14 deg |
| Pitch | 1.95e-5 deg |
| Yaw | -2.01e-16 deg |

## Interpretation

- RAFT's equilibrium heave offset is about 0.18 m. OrcaFlex's exported dynamic channels are labeled as local/dynamic axes, and their reference frame and offsets have not been verified. Their means should therefore not be interpreted as directly comparable absolute platform positions.
- The two wave descriptions differ: RAFT used JONSWAP, whereas the OrcaFlex analysis uses a regular-wave harmonic reference. This affects the response statistics, especially RMS and extrema.
- The RAFT and OrcaFlex statistics use different reported measures (`RMS` versus standard deviation). Treat the table as a compact overview, not a like-for-like validation.
- RAFT reports parked-turbine rotor fluid loads as neglected. Nacelle acceleration and tower bending are omitted because no corresponding OrcaFlex channels were included in this six-DOF export.

OrcaFlex statistics source: `../results/orcaflex_comparison/orcaflex_six_dof_statistics.csv`. OrcaFlex run metadata and input hashes: `../results/orcaflex_comparison/run_manifest.json`.
