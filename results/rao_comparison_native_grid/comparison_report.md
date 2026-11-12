# OrcaFlex and RAFT RAO comparison

Compared RAOS-folder OrcaFlex files at 29 distinct frequencies from 0.01 to 0.25 Hz.

## Motion RAOs

The RAOS displacement files contain amplitude only, with no phase. RAFT complex values are interpolated in real/imaginary coordinates and reduced to magnitude for comparison. Phase differences are therefore unavailable. Relative errors are blank when the RAFT amplitude is below 1e-8 in the DOF's RAO units.

The plot shows RAFT and OrcaFlex at their original frequency samples; the comparison table uses the interpolated RAFT values. The saved RAFT RAO CSV does not record wave amplitude, so it cannot identify frequencies where the wave component is zero or too small for a reliable ratio. A saved zero at such a frequency is not evidence of a zero physical RAO. RAFT's getRAO helper excludes wave amplitudes at or below 1e-6 m; the exporter instead divides by every nonzero wave component.

Only the three per-DOF OrcaFlex displacement CSVs in `OrcaFlex_Results/RAOS/` are used as OrcaFlex motion inputs. Their filenames specify 0 degrees. The RAFT generator uses heading 0 degrees; physical axis and direction conventions remain unverified.

RAFT generation settings visible in `examples/special_course_single_raos.py`: VolturnUS-S design, parked turbine, zero wind, JONSWAP, Tp=16 s, Hs=7 m, gamma=3.3, heading=0 degrees. These are source defaults, not embedded provenance in the RAO CSV.

At the compared sample nearest 9.875 s (0.10000000 Hz):

| DOF | Unit | OrcaFlex amplitude | RAFT interpolated amplitude | Absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|
| surge | m/m | 0.329856 | 0.326712 | 0.00314414 | 0.96% |
| heave | m/m | 0.350664 | 0.337392 | 0.0132724 | 3.93% |
| pitch | rad/m | 0.00497119 | 0.00272303 | 0.00224816 | 82.56% |

## Load RAO files

The three load files are plain-text CSVs with frequency and amplitude only, without phase. Their metadata does not identify a load definition, reference frame, or reference point. RAFT's current saved RAO table contains motion/velocity/acceleration RAOs, not force RAOs, so these load amplitudes are inventoried but not quantitatively compared.

| DOF | Amplitude units | Samples | Frequency range [Hz] | Phase available |
|---|---|---:|---:|---|
| surge | kN/m | 31 | 0.00166667 to 0.25 | no |
| heave | kN/m | 31 | 0.00166667 to 0.25 | no |
| pitch | kN.m/m | 31 | 0.00166667 to 0.25 | no |

## Caveats

- The per-DOF OrcaFlex files do not state a reference point; the RAFT platform reference point and axis mapping have not been independently verified.
- The OrcaFlex motion files specify 0-degree filenames, but physical heading alignment is not independently established.
- RAFT's JONSWAP case and OrcaFlex's exported RAO conditions are not proven identical by the available files.
- Interpolated values are not extrapolated beyond shared frequency coverage.

## Inputs

- OrcaFlex motion inputs (all in `OrcaFlex_Results/RAOS/`):
  - `IEA15MW_VolturnUS_Disp_RAO_surge_0deg.csv` (SHA-256 `caa74a892b9f2dfaa0cf42cffd948c8beb96c8c0cbac3d4d2a8c3ab2d8cb0d05`)
  - `IEA15MW_VolturnUS_Disp_RAO_heave_0deg.csv` (SHA-256 `48f76e12a399fbb56df561ed69e04dbac3a9c4b1923f4fe5ad890810c236116b`)
  - `IEA15MW_VolturnUS_Disp_RAO_pitch_0deg.csv` (SHA-256 `8a6d23087f205c7f06a79db0f17059e9659db343064c2fa6455e0bddc8703830`)
- RAFT motion RAOs: `C:\Users\Z00565RD\OneDrive - Siemens Energy\Documents\DTU Special Course 2026\RAFT\results\floater_raos.csv` (SHA-256 `4fa1854e7c1eba824a6a33e23af6026e737f7ccaab9160ecd1b785acb85f7ebb`)
- OrcaFlex load CSV hashes and metadata are recorded in `load_rao_inventory.csv` and `run_manifest.json`.
