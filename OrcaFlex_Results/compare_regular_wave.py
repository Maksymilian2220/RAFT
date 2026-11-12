"""Compare a regular-wave OrcaFlex floater run with a matched RAFT case.

Run from the repository root or any directory:

    python OrcaFlex_Results/compare_regular_wave.py

The script writes its RAFT run products and comparison artifacts under
OrcaFlex_Results/Comparison. OrcaFlex channel frame/filter definitions remain
unverified, so raw Dynamic and Primary WF results are reported separately.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ORCAFLEX_DIR = ROOT / "OrcaFlex_Results" / "Dir0_Hs0.5395_Tp16"
OUTPUT_DIR = ROOT / "OrcaFlex_Results" / "Comparison"
FLOAT_CSV = ORCAFLEX_DIR / "IEA15MW_VolturnUS_regular_Hs0.5395_Tp16_floater_01.csv"
TURBINE_CSV = ORCAFLEX_DIR / "IEA15MW_VolturnUS_regular_Hs0.5395_Tp16_01.csv"
RAFT_SCRIPT = ROOT / "examples" / "special_course_singlef.py"
FREQUENCY_HZ = 1.0 / 16.0
PERIOD_S = 16.0
RAFT_HS_M = 7.0
RAFT_GAMMA = 30.0
NOMINAL_WAVE_HEIGHT_M = 0.5395
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
CHANNEL_SETS = {
    "Dynamic": tuple(f"Dynamic {axis}" for axis in ("x", "y", "z", "Rx", "Ry", "Rz")),
    "Primary WF": tuple(f"Primary WF {dof}" for dof in DOFS),
}
UNIT_BY_DOF = {dof: "m" if index < 3 else "deg" for index, dof in enumerate(DOFS)}
WINDOW_CYCLES = (5, 10, 20, 30, 37)


def read_orcaflex_export(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read an OrcaFlex CSV with metadata lines preceding its numeric table."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    rows = list(csv.reader(text.splitlines()))
    header_rows = [index for index, row in enumerate(rows)
                   if row and row[0].strip() == "Time"]
    if len(header_rows) != 1:
        raise ValueError(f"{path}: expected exactly one numeric Time header.")
    header_index = header_rows[0]
    names = [name.strip() for name in rows[header_index]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate numeric column names.")
    data_rows = [row for row in rows[header_index + 1:] if row]
    try:
        data = np.asarray([[float(value) for value in row] for row in data_rows], dtype=float)
    except ValueError as error:
        raise ValueError(f"{path}: non-numeric row in time history.") from error
    if data.ndim != 2 or data.shape[1] != len(names) or len(data) < 4:
        raise ValueError(f"{path}: malformed or incomplete numeric table.")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: time history contains nonfinite values.")
    time = data[:, names.index("Time")]
    intervals = np.diff(time)
    if np.any(intervals <= 0) or not np.allclose(
        intervals, np.median(intervals), rtol=1e-6, atol=1e-9
    ):
        raise ValueError(f"{path}: time must be increasing and uniformly sampled.")
    return time, {name: data[:, index] for index, name in enumerate(names)}


def fit_harmonic(time: np.ndarray, values: np.ndarray, frequency_hz: float) -> dict:
    """Fit mean + one sinusoid and return its complex amplitude at t=0."""
    phase = 2.0 * np.pi * frequency_hz * time
    design = np.column_stack((np.ones(len(time)), np.cos(phase), np.sin(phase)))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    residual = values - fitted
    complex_amplitude = coefficients[1] - 1j * coefficients[2]
    return {
        "mean": float(coefficients[0]),
        "complex_amplitude": complex(complex_amplitude),
        "amplitude": float(abs(complex_amplitude)),
        "phase_deg": float(np.angle(complex_amplitude, deg=True)),
        "harmonic_rms": float(abs(complex_amplitude) / np.sqrt(2.0)),
        "std": float(np.std(values)),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "fitted": fitted,
    }


def read_raft_response(path: Path) -> tuple[dict[str, dict], complex, float]:
    """Read the extracted RAFT single-frequency response as complex phasors."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"RAFT response CSV is empty: {path}")
    selected_frequency = float(rows[0]["selected_frequency_hz"])
    if not np.isclose(selected_frequency, FREQUENCY_HZ, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"RAFT selected {selected_frequency:.8f} Hz, expected {FREQUENCY_HZ:.8f} Hz."
        )
    wave_amplitude = float(rows[0]["wave_amplitude_m"])
    wave_phase = float(rows[0].get("wave_phase_deg", 0.0))
    wave = wave_amplitude * np.exp(1j * np.deg2rad(wave_phase))
    responses = {}
    for row in rows:
        amplitude = float(row["displacement_amplitude"])
        phase = float(row["displacement_phase_deg"])
        responses[row["dof"]] = {
            "complex_amplitude": amplitude * np.exp(1j * np.deg2rad(phase)),
            "amplitude": amplitude,
            "phase_deg": phase,
        }
    missing = set(DOFS).difference(responses)
    if missing:
        raise ValueError(f"RAFT response is missing DOFs: {sorted(missing)}")
    if wave_amplitude <= 0.0:
        raise ValueError("RAFT selected wave amplitude must be positive.")
    return responses, complex(wave), selected_frequency


def window_metrics(
    time: np.ndarray,
    channels: dict[str, np.ndarray],
    wave_channel: str | None,
    frequency_hz: float,
) -> list[dict]:
    """Fit candidate trailing integer-cycle windows for both motion sets."""
    output = []
    durations = [cycles / frequency_hz for cycles in WINDOW_CYCLES]
    for duration, cycles in zip(durations, WINDOW_CYCLES):
        start = time[-1] - duration
        mask = time >= start - 1e-9
        fit_time = time[mask]
        wave_fit = fit_harmonic(
            fit_time, channels[wave_channel][mask], frequency_hz
        ) if wave_channel else None
        for channel_set, column_names in CHANNEL_SETS.items():
            for dof, column in zip(DOFS, column_names):
                result = fit_harmonic(
                    fit_time, channels[column][mask], frequency_hz
                )
                relative_phase = None
                if wave_fit and wave_fit["amplitude"] > 1e-10:
                    relative_phase = float(np.angle(
                        result["complex_amplitude"] / wave_fit["complex_amplitude"],
                        deg=True,
                    ))
                output.append({
                    "channel_set": channel_set,
                    "dof": dof,
                    "cycles": cycles,
                    "window_s": duration,
                    "start_s": float(fit_time[0]),
                    "end_s": float(fit_time[-1]),
                    "wave_amplitude_m": wave_fit["amplitude"] if wave_fit else None,
                    "wave_phase_deg": wave_fit["phase_deg"] if wave_fit else None,
                    "mean": result["mean"],
                    "std": result["std"],
                    "amplitude": result["amplitude"],
                    "phase_deg": result["phase_deg"],
                    "phase_relative_to_wave_deg": relative_phase,
                    "harmonic_rms": result["harmonic_rms"],
                    "response_per_wave_amplitude": (
                        result["amplitude"] / wave_fit["amplitude"]
                        if wave_fit and wave_fit["amplitude"] > 1e-10 else None
                    ),
                    "residual_rms": result["residual_rms"],
                    "minimum": result["minimum"],
                    "maximum": result["maximum"],
                })
    return output


def make_summary(
    metrics: list[dict], raft_responses: dict[str, dict], raft_wave: complex,
    window_cycles: int,
) -> list[dict]:
    selected = [row for row in metrics if row["cycles"] == window_cycles]
    summary = []
    for row in selected:
        dof = row["dof"]
        raft_transfer = raft_responses[dof]["complex_amplitude"] / raft_wave
        orca_wave_amplitude = row["wave_amplitude_m"]
        scaled_raft = raft_transfer * orca_wave_amplitude if orca_wave_amplitude else None
        orca_amplitude = row["amplitude"]
        relative_phase = row["phase_relative_to_wave_deg"]
        raft_phase = (
            float(np.angle(raft_transfer, deg=True))
            if abs(raft_transfer) > 1e-8 else None
        )
        summary.append({
            "channel_set": row["channel_set"],
            "dof": dof,
            "unit": UNIT_BY_DOF[dof],
            "window_cycles": window_cycles,
            "window_s": row["window_s"],
            "mean_orcaflex": row["mean"],
            "orcaflex_amplitude": orca_amplitude,
            "orcaflex_harmonic_rms": row["harmonic_rms"],
            "orcaflex_phase_relative_to_wave_deg": relative_phase,
            "orcaflex_response_per_wave_amplitude": row["response_per_wave_amplitude"],
            "orcaflex_fit_residual_rms": row["residual_rms"],
            "raft_amplitude_scaled_to_orcaflex_wave": abs(scaled_raft) if scaled_raft is not None else None,
            "raft_harmonic_rms_scaled": abs(scaled_raft) / np.sqrt(2.0) if scaled_raft is not None else None,
            "raft_phase_relative_to_wave_deg": raft_phase,
            "amplitude_difference_orcaflex_minus_raft": (
                orca_amplitude - abs(scaled_raft) if scaled_raft is not None else None
            ),
            "amplitude_difference_percent_of_raft": (
                100.0 * (orca_amplitude - abs(scaled_raft)) / abs(scaled_raft)
                if scaled_raft is not None and abs(scaled_raft) > 1e-12 else None
            ),
            "phase_difference_orcaflex_minus_raft_deg": (
                float(np.angle(np.exp(1j * np.deg2rad(relative_phase - raft_phase)), deg=True))
                if relative_phase is not None and raft_phase is not None else None
            ),
        })
    return summary


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_raft(output: Path) -> None:
    """Run the exact pasted RAFT case without replacing existing results."""
    subprocess.run([
        sys.executable,
        str(RAFT_SCRIPT),
        "--period", str(PERIOD_S),
        "--height", str(RAFT_HS_M),
        "--heading", "0",
        "--wind-speed", "0",
        "--turbulence", "0",
        "--gamma", str(RAFT_GAMMA),
        "--output", str(output / "raft_single_frequency_response.csv"),
        "--full-output", str(output / "raft_full_frequency_response.csv"),
        "--spectrum-plot", str(output / "raft_jonswap_spectrum.png"),
    ], cwd=ROOT, check=True)


def plot_time_histories(
    path: Path,
    time: np.ndarray,
    channels: dict[str, np.ndarray],
    raft_responses: dict[str, dict],
    raft_wave: complex,
    wave_channel: str | None,
    window_cycles: int,
) -> None:
    duration = window_cycles / FREQUENCY_HZ
    mask = time >= time[-1] - duration - 1e-9
    plot_time = time[mask]
    wave_amplitude = NOMINAL_WAVE_HEIGHT_M / 2.0
    wave_phase = 0.0
    if wave_channel:
        wave_fit = fit_harmonic(plot_time, channels[wave_channel][mask], FREQUENCY_HZ)
        if wave_fit["amplitude"] > 1e-10:
            wave_amplitude = wave_fit["amplitude"]
            wave_phase = wave_fit["phase_deg"]
    reference_wave = wave_amplitude * np.exp(1j * np.deg2rad(wave_phase))
    relative_time = plot_time - plot_time[0]
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for index, (axis, dof) in enumerate(zip(axes.flat, DOFS)):
        column = CHANNEL_SETS["Primary WF"][index]
        values = channels[column][mask]
        fit = fit_harmonic(plot_time, values, FREQUENCY_HZ)
        raft_expected = (raft_responses[dof]["complex_amplitude"] / raft_wave) * reference_wave
        prediction = np.real(raft_expected * np.exp(1j * 2.0 * np.pi * FREQUENCY_HZ * plot_time))
        axis.plot(relative_time, values - fit["mean"], label="OrcaFlex Primary WF (demeaned)", lw=0.8)
        axis.plot(relative_time, prediction, label="RAFT scaled to OrcaFlex wave", lw=1.0)
        axis.set(xlabel="Time within selected window [s]", ylabel=UNIT_BY_DOF[dof])
        axis.set_title(dof.capitalize(), fontsize=11)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        f"16 s harmonic comparison, trailing {window_cycles} cycles ({duration:.0f} s)\n"
        "Primary WF meaning/frame is unverified; RAFT phase uses measured wave when available",
        fontsize=11,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_amplitudes(path: Path, summary: list[dict]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(DOFS))
    width = 0.24
    for offset, channel_set in zip((-width, 0.0, width), ("RAFT", "Dynamic", "Primary WF")):
        rows = []
        for dof in DOFS:
            if channel_set == "RAFT":
                matches = [row for row in summary if row["dof"] == dof]
                rows.append(matches[0]["raft_amplitude_scaled_to_orcaflex_wave"] if matches else 0.0)
            else:
                matches = [row for row in summary if row["dof"] == dof and row["channel_set"] == channel_set]
                rows.append(matches[0]["orcaflex_amplitude"] if matches else 0.0)
        axes[0].bar(x + offset, rows, width, label=channel_set)
    axes[0].set_xticks(x, [name.capitalize() for name in DOFS])
    axes[0].set_ylabel("Harmonic amplitude [m or deg]")
    axes[0].set_title("Response amplitude")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    for offset, channel_set in zip((-width, 0.0, width), ("RAFT", "Dynamic", "Primary WF")):
        phases = []
        for dof in DOFS:
            if channel_set == "RAFT":
                matches = [row for row in summary if row["dof"] == dof]
                phase = matches[0]["raft_phase_relative_to_wave_deg"] if matches else np.nan
            else:
                matches = [row for row in summary if row["dof"] == dof and row["channel_set"] == channel_set]
                phase = matches[0]["orcaflex_phase_relative_to_wave_deg"] if matches else np.nan
            phases.append(phase if phase is not None else np.nan)
        axes[1].bar(x + offset, phases, width, label=channel_set)
    axes[1].set_xticks(x, [name.capitalize() for name in DOFS])
    axes[1].set_ylabel("Phase relative to wave [deg]")
    axes[1].set_ylim(-190, 190)
    axes[1].set_yticks(np.arange(-180, 181, 60))
    axes[1].set_title("Harmonic phase (wrapped)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("RAFT versus OrcaFlex floater motions; channel definitions unverified")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_raft:
        run_raft(output)
    raft_csv = output / "raft_single_frequency_response.csv"
    if not raft_csv.exists():
        raise FileNotFoundError(f"Missing RAFT output {raft_csv}; run without --skip-raft.")

    time, channels = read_orcaflex_export(args.floater_csv)
    required = {column for columns in CHANNEL_SETS.values() for column in columns}
    missing = required.difference(channels)
    if missing:
        raise ValueError(f"OrcaFlex floater export is missing channels: {sorted(missing)}")
    wave_channel = "Sea surface Z" if "Sea surface Z" in channels else None
    raft_responses, raft_wave, selected_frequency = read_raft_response(raft_csv)
    if selected_frequency != FREQUENCY_HZ:
        raise ValueError("Unexpected RAFT selected frequency.")
    metrics = window_metrics(time, channels, wave_channel, FREQUENCY_HZ)
    summary = make_summary(metrics, raft_responses, raft_wave, args.summary_cycles)
    write_rows(output / "window_sensitivity.csv", metrics)
    write_rows(output / "comparison_summary.csv", summary)
    plot_time_histories(
        output / "primary_wf_time_comparison.png", time, channels,
        raft_responses, raft_wave, wave_channel, args.summary_cycles,
    )
    plot_amplitudes(output / "amplitude_phase_comparison.png", summary)

    manifest = {
        "period_s": PERIOD_S,
        "frequency_hz": FREQUENCY_HZ,
        "raft_case": {
            "wave_spectrum": "JONSWAP", "Hs_m": RAFT_HS_M,
            "Tp_s": PERIOD_S, "gamma": RAFT_GAMMA,
            "wave_heading_deg": 0, "wind_speed_m_s": 0,
            "turbulence": 0, "turbine_status": "parked",
            "selected_frequency_hz": selected_frequency,
            "selected_wave_amplitude_m": abs(raft_wave),
        },
        "orcaflex_wave_height_label_m": NOMINAL_WAVE_HEIGHT_M,
        "orcaflex_measured_wave_channel": wave_channel,
        "orcaflex_time_start_s": float(time[0]),
        "orcaflex_time_end_s": float(time[-1]),
        "orcaflex_sample_count": int(len(time)),
        "orcaflex_sample_interval_s": float(np.median(np.diff(time))),
        "summary_window_cycles": args.summary_cycles,
        "window_sensitivity_cycles": list(WINDOW_CYCLES),
        "channel_sets": CHANNEL_SETS,
        "limitations": [
            "OrcaFlex wave direction versus RAFT global +X convention is unverified.",
            "Floater Dynamic and Primary WF frame, origin, and filtering are unverified; both are reported separately.",
            "Sea surface Z is used as the wave reference, but its location/definition is not independently verified.",
            "Mean offsets are reported separately and are not assumed to share the RAFT equilibrium reference.",
            "RAFT is a JONSWAP linear frequency-domain response; comparison is at the selected 16 s component only.",
        ],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.floater_csv, args.turbine_csv, raft_csv,
                         output / "raft_full_frequency_response.csv")
            if path.exists()
        },
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = [
        "# Regular-wave OrcaFlex / RAFT comparison",
        "",
        f"Frequency: {FREQUENCY_HZ:.6f} Hz ({PERIOD_S:.1f} s).",
        f"RAFT selected component amplitude: {abs(raft_wave):.6f} m.",
        f"OrcaFlex measured wave amplitude from {wave_channel}: "
        f"{metrics[-1]['wave_amplitude_m'] if wave_channel else NOMINAL_WAVE_HEIGHT_M / 2:.6f} m.",
        f"Summary uses the trailing {args.summary_cycles}-cycle window; see "
        "`window_sensitivity.csv` before interpreting stability.",
        "",
        "## Interpretation limits",
        "",
        "- The exported OrcaFlex direction, motion frames/origins, and Primary WF filtering are not verified against RAFT.",
        "- Dynamic and Primary WF channels are both retained as alternatives, not treated as interchangeable.",
        "- The RAFT and OrcaFlex means are shown separately; compare wave-frequency amplitudes and wave-relative phases first.",
        "- This is a single-frequency comparison, not a comparison of full JONSWAP and regular-wave statistics.",
        "- Several motion fits have residual RMS larger than their 16 s harmonic amplitude; those harmonic estimates are not clean single-frequency responses. Check `window_sensitivity.csv` and the time-history plot before drawing quantitative conclusions.",
    ]
    (output / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Comparison outputs saved under {output}")
    print(f"OrcaFlex records: {len(time)} samples, {time[-1] - time[0]:.1f} s")
    print(f"Wave reference channel: {wave_channel or 'nominal wave height'}")
    print(f"Summary interval: trailing {args.summary_cycles} cycles")


def self_test() -> None:
    time = np.arange(6001) * 0.1
    frequency = 1.0 / 16.0
    amplitude = 0.37
    phase = np.deg2rad(-43.0)
    signal = 2.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
    result = fit_harmonic(time, signal, frequency)
    if not np.isclose(result["amplitude"], amplitude, rtol=1e-12):
        raise AssertionError("Synthetic harmonic amplitude was not recovered.")
    if not np.isclose(result["phase_deg"], np.rad2deg(phase), atol=1e-10):
        raise AssertionError("Synthetic harmonic phase was not recovered.")
    parsed_time, channels = read_orcaflex_export(FLOAT_CSV)
    if len(parsed_time) != 6001 or "Primary WF surge" not in channels:
        raise AssertionError("Supplied OrcaFlex export did not parse as expected.")
    print("Self-test passed: 16 s harmonic fit and supplied floater export parse.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floater-csv", type=Path, default=FLOAT_CSV)
    parser.add_argument("--turbine-csv", type=Path, default=TURBINE_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--summary-cycles", type=int, choices=WINDOW_CYCLES, default=10)
    parser.add_argument("--skip-raft", action="store_true", help="Use RAFT outputs already in the output folder.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    run_comparison(args)


if __name__ == "__main__":
    main()