# Parked OrcaFlex / RAFT comparison

Record: 6001 samples, 600.000 s, dt=0.100000 s.

Wave: H=0.125 m, T=9.875 s, f=0.10126582 Hz. Wind: 3.93 m/s at 10 m.

## Interpretation and limitations

- The complete exported record is post-startup; no samples were removed.
- Regular wave: wave_height=0.125 m is treated as crest-to-trough H, A=H/2. The source calls it Hs; confirm this convention in OrcaFlex.
- Motion/acceleration six-DOF time channels use native dynamic/local axes, not assumed global platform axes.
- Turbine attachment point and axis mapping are unverified. No quantitative hub error percentages are reported.
- Platform predictions come from origin RAOs, not measured platform time histories. No platform means/extrema can be recovered from these exports.
- RAFT parked status omits rotor aerodynamic loads. Steady parked wind loading may therefore differ from OrcaFlex.
- Wind is specified at 10 m; hub conversion uses an assumed power-law exponent, configurable with --shear-exponent.
- Head-on alignment is given, but world axes and from/to conventions are not verified; RAFT headings default provisionally to 0 degrees. OrcaFlex RAO direction is 180 degrees.
- PSD is one-sided density per Hz. RAFT density per rad/s must be multiplied by 2*pi, not divided.
- Regular-wave comparisons use line amplitude and variance; peak PSD height depends on record/window. No wave phase or coherence is inferred.
- Observed extrema, total RMS, fluctuating standard deviation, harmonic RMS and mean +/- 3std are distinct statistics.
