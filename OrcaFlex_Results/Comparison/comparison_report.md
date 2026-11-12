# Regular-wave OrcaFlex / RAFT comparison

Frequency: 0.062500 Hz (16.0 s).
RAFT selected component amplitude: 0.269736 m.
OrcaFlex measured wave amplitude from Sea surface Z: 0.269674 m.
Summary uses the trailing 10-cycle window; see `window_sensitivity.csv` before interpreting stability.

## Interpretation limits

- The exported OrcaFlex direction, motion frames/origins, and Primary WF filtering are not verified against RAFT.
- Dynamic and Primary WF channels are both retained as alternatives, not treated as interchangeable.
- The RAFT and OrcaFlex means are shown separately; compare wave-frequency amplitudes and wave-relative phases first.
- This is a single-frequency comparison, not a comparison of full JONSWAP and regular-wave statistics.
- Several motion fits have residual RMS larger than their 16 s harmonic amplitude; those harmonic estimates are not clean single-frequency responses. Check `window_sensitivity.csv` and the time-history plot before drawing quantitative conclusions.
