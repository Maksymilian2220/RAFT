"""Compare one wind and one no-wind regular-wave OrcaFlex run with RAFT.

Run from the repository root or any directory::

    python OrcaFlex_Results/compare_wind_vs_nowind.py

The existing source exports and prior Comparison outputs are left untouched.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import compare_regular_wave as regular


SOURCE_DIR = regular.ROOT / "OrcaFlex_Results" / "Regular_Dir0_Hs0.5395_Tp16"
OUTPUT_DIR = regular.ROOT / "OrcaFlex_Results" / "Comparison" / "wind_vs_nowind"
WIND_FLOATER = SOURCE_DIR / "IEA15MW_VolturnUS_regular_Hs0.5395_Tp16_floater_17ms.csv"
NOWIND_FLOATER = SOURCE_DIR / "IEA15MW_VolturnUS_regular_Hs0.5395_Tp16_floater_nowind.csv"
RAFT_CSV = regular.ROOT / "OrcaFlex_Results" / "Comparison" / "raft_single_frequency_response.csv"
RAFT_WIND_SPEED = 17.1
SUMMARY_CYCLES = 10


def fit_export(path: Path, group: str) -> tuple[np.ndarray, list[dict]]:
    """Fit wave and motion harmonics over the configured trailing windows."""
    time, channels = regular.read_orcaflex_export(path)
    required = {"Sea surface Z"}
    required.update(column for columns in regular.CHANNEL_SETS.values() for column in columns)
    missing = required.difference(channels)
    if missing:
        raise ValueError(f"{path} is missing channels: {sorted(missing)}")

    rows = []
    for cycles in regular.WINDOW_CYCLES:
        duration = cycles / regular.FREQUENCY_HZ
        mask = time >= time[-1] - duration - 1e-9
        window_time = time[mask]
        wave = regular.fit_harmonic(
            window_time, channels["Sea surface Z"][mask], regular.FREQUENCY_HZ
        )
        for channel_set, columns in regular.CHANNEL_SETS.items():
            for dof, column in zip(regular.DOFS, columns):
                fit = regular.fit_harmonic(
                    window_time, channels[column][mask], regular.FREQUENCY_HZ
                )
                phase_relative = None
                if wave["amplitude"] > 1e-10:
                    phase_relative = float(np.angle(
                        fit["complex_amplitude"] / wave["complex_amplitude"],
                        deg=True,
                    ))
                rows.append({
                    "group": group,
                    "run": path.stem,
                    "source_file": str(path),
                    "channel_set": channel_set,
                    "dof": dof,
                    "unit": regular.UNIT_BY_DOF[dof],
                    "cycles": cycles,
                    "window_s": duration,
                    "start_s": float(window_time[0]),
                    "end_s": float(window_time[-1]),
                    "wave_mean_m": wave["mean"],
                    "wave_amplitude_m": wave["amplitude"],
                    "wave_phase_deg": wave["phase_deg"],
                    "motion_mean": fit["mean"],
                    "motion_std": fit["std"],
                    "motion_amplitude": fit["amplitude"],
                    "motion_harmonic_rms": fit["harmonic_rms"],
                    "motion_phase_deg": fit["phase_deg"],
                    "phase_relative_to_wave_deg": phase_relative,
                    "response_per_wave_amplitude": (
                        fit["amplitude"] / wave["amplitude"]
                        if wave["amplitude"] > 1e-10 else None
                    ),
                    "fit_residual_rms": fit["residual_rms"],
                    "minimum": fit["minimum"],
                    "maximum": fit["maximum"],
                })
    return time, rows


def circular_mean_degrees(values: list[float]) -> float | None:
    if not values:
        return None
    mean = float(np.angle(np.mean(np.exp(1j * np.deg2rad(values))), deg=True))
    return float((mean + 180.0) % 360.0 - 180.0)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def summarize_runs(
    fits: list[dict],
    raft_by_group: dict[str, tuple[dict[str, dict], complex]],
    cycles: int,
) -> list[dict]:
    """Summarize each run against RAFT, then add group statistics."""
    selected = [row for row in fits if row["cycles"] == cycles]
    result = []
    for row in selected:
        raft, raft_wave = raft_by_group[row["group"]]
        transfer = raft[row["dof"]]["complex_amplitude"] / raft_wave
        scaled_amplitude = abs(transfer) * row["wave_amplitude_m"]
        result.append({
            "row_type": "run",
            "group": row["group"],
            "run": row["run"],
            "channel_set": row["channel_set"],
            "dof": row["dof"],
            "unit": row["unit"],
            "cycles": cycles,
            "wave_amplitude_m": row["wave_amplitude_m"],
            "motion_mean": row["motion_mean"],
            "orcaflex_amplitude": row["motion_amplitude"],
            "orcaflex_harmonic_rms": row["motion_harmonic_rms"],
            "orcaflex_phase_relative_to_wave_deg": row["phase_relative_to_wave_deg"],
            "response_per_wave_amplitude": row["response_per_wave_amplitude"],
            "fit_residual_rms": row["fit_residual_rms"],
            "raft_amplitude": abs(raft[row["dof"]]["complex_amplitude"]),
            "raft_response_per_wave_amplitude": abs(transfer),
            "raft_amplitude_scaled_to_run_wave": scaled_amplitude,
            "raft_phase_relative_to_wave_deg": (
                float(np.angle(transfer, deg=True)) if abs(transfer) > 1e-8 else None
            ),
            "amplitude_difference_orcaflex_minus_raft": (
                row["motion_amplitude"] - scaled_amplitude
            ),
        })

    group_keys = sorted({(row["group"], row["channel_set"], row["dof"])
                         for row in selected})
    for group, channel_set, dof in group_keys:
        group_rows = [row for row in selected if (
            row["group"], row["channel_set"], row["dof"]
        ) == (group, channel_set, dof)]
        amplitude_mean, amplitude_sd = _mean_sd(
            [row["motion_amplitude"] for row in group_rows]
        )
        transfer_mean, transfer_sd = _mean_sd(
            [row["response_per_wave_amplitude"] for row in group_rows]
        )
        residual_mean, residual_sd = _mean_sd(
            [row["fit_residual_rms"] for row in group_rows]
        )
        wave_mean, wave_sd = _mean_sd(
            [row["wave_amplitude_m"] for row in group_rows]
        )
        mean_motion, mean_motion_sd = _mean_sd(
            [row["motion_mean"] for row in group_rows]
        )
        relative_phases = [row["phase_relative_to_wave_deg"] for row in group_rows
                           if row["phase_relative_to_wave_deg"] is not None]
        raft, raft_wave = raft_by_group[group]
        raft_transfer = raft[dof]["complex_amplitude"] / raft_wave
        raft_scaled = abs(raft_transfer) * wave_mean
        result.append({
            "row_type": "group_mean_sd",
            "group": group,
            "run": f"n={len(group_rows)}",
            "channel_set": channel_set,
            "dof": dof,
            "unit": regular.UNIT_BY_DOF[dof],
            "cycles": cycles,
            "wave_amplitude_m": wave_mean,
            "wave_amplitude_sd_m": wave_sd,
            "motion_mean": mean_motion,
            "motion_mean_sd": mean_motion_sd,
            "orcaflex_amplitude": amplitude_mean,
            "orcaflex_amplitude_sd": amplitude_sd,
            "orcaflex_harmonic_rms": amplitude_mean / np.sqrt(2.0),
            "orcaflex_phase_relative_to_wave_deg": circular_mean_degrees(relative_phases),
            "response_per_wave_amplitude": transfer_mean,
            "response_per_wave_amplitude_sd": transfer_sd,
            "fit_residual_rms": residual_mean,
            "fit_residual_rms_sd": residual_sd,
            "raft_amplitude": abs(raft[dof]["complex_amplitude"]),
            "raft_response_per_wave_amplitude": abs(raft_transfer),
            "raft_amplitude_scaled_to_run_wave": raft_scaled,
            "raft_phase_relative_to_wave_deg": (
                float(np.angle(raft_transfer, deg=True))
                if abs(raft_transfer) > 1e-8 else None
            ),
            "amplitude_difference_orcaflex_minus_raft": amplitude_mean - raft_scaled,
        })
    return result


def summarize_wind_effect(summary: list[dict], cycles: int) -> list[dict]:
    group_rows = [row for row in summary if row["row_type"] == "group_mean_sd"
                  and row["cycles"] == cycles]
    by_key = {(row["group"], row["channel_set"], row["dof"]): row
              for row in group_rows}
    output = []
    for channel_set in regular.CHANNEL_SETS:
        for dof in regular.DOFS:
            no_wind = by_key.get(("no_wind", channel_set, dof))
            wind = by_key.get(("wind", channel_set, dof))
            if no_wind is None or wind is None:
                continue
            output.append({
                "channel_set": channel_set,
                "dof": dof,
                "unit": regular.UNIT_BY_DOF[dof],
                "cycles": cycles,
                "no_wind_replicates": no_wind["run"],
                "wind_replicates": wind["run"],
                "no_wind_amplitude_mean": no_wind["orcaflex_amplitude"],
                "no_wind_amplitude_sd": no_wind["orcaflex_amplitude_sd"],
                "wind_amplitude_mean": wind["orcaflex_amplitude"],
                "wind_amplitude_sd": wind["orcaflex_amplitude_sd"],
                "wind_minus_no_wind_amplitude": (
                    wind["orcaflex_amplitude"] - no_wind["orcaflex_amplitude"]
                ),
                "wind_minus_no_wind_wave_amplitude": (
                    wind["wave_amplitude_m"] - no_wind["wave_amplitude_m"]
                ),
                "no_wind_phase_relative_to_wave_deg": (
                    no_wind["orcaflex_phase_relative_to_wave_deg"]
                ),
                "wind_phase_relative_to_wave_deg": (
                    wind["orcaflex_phase_relative_to_wave_deg"]
                ),
                "no_wind_fit_residual_rms": no_wind["fit_residual_rms"],
                "wind_fit_residual_rms": wind["fit_residual_rms"],
            })
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row
    ))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, summary: list[dict], cycles: int) -> None:
    """Plot dimensional and wave-normalized RAFT/OrcaFlex responses."""
    rows = [row for row in summary if row["row_type"] == "group_mean_sd"
            and row["channel_set"] == "Primary WF" and row["cycles"] == cycles]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cases = (("no_wind", "OrcaFlex 0 m/s", "C0"),
             ("wind", "OrcaFlex 17.1 m/s", "C1"),
             ("no_wind", "RAFT 0 m/s", "C2"),
             ("wind", "RAFT 17.1 m/s", "C3"))
    metrics = (
        ("orcaflex_amplitude", "raft_amplitude", "orcaflex_amplitude_sd",
         "Harmonic response amplitude", "Amplitude [m or deg]"),
        ("response_per_wave_amplitude", "raft_response_per_wave_amplitude",
         "response_per_wave_amplitude_sd", "Wave-normalized response",
         "Response / wave amplitude [m/m or deg/m]"),
    )
    positions = np.arange(len(regular.DOFS))
    width = 0.2
    for axis, (orcaflex_key, raft_key, deviation_key, title,
               ylabel) in zip(axes, metrics):
        for case_index, (group, label, color) in enumerate(cases):
            values, deviations = [], []
            for dof in regular.DOFS:
                row = next(
                    (item for item in rows
                     if item["group"] == group and item["dof"] == dof), None
                )
                is_raft = case_index >= 2
                values.append(
                    row[raft_key if is_raft else orcaflex_key] if row else np.nan
                )
                deviations.append(row[deviation_key] if row and not is_raft else 0)
            axis.bar(positions + (case_index - 1.5) * width, values, width,
                     yerr=deviations if case_index < 2 else None,
                     capsize=3, color=color, label=label)
        axis.set_xticks(positions, [dof.capitalize() for dof in regular.DOFS])
        axis.set_xlabel("Degree of freedom")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_comparison(args: argparse.Namespace) -> None:
    if not args.wind_csv.exists() or not args.nowind_csv.exists():
        raise FileNotFoundError("Both wind and no-wind floater exports must exist.")
    raft_by_group = {}
    raft_paths = {
        "no_wind": args.output / "raft_no_wind_single_frequency_response.csv",
        "wind": args.output / "raft_17ms_single_frequency_response.csv",
    }
    if args.skip_raft:
        for group, path in raft_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {group} RAFT reference {path}; rerun without --skip-raft."
                )
    else:
        run_raft(args.output, "no_wind", wind_speed=0.0)
        run_raft(args.output, "17ms", wind_speed=RAFT_WIND_SPEED)
    for group, path in raft_paths.items():
        responses, wave, selected_frequency = regular.read_raft_response(path)
        if not np.isclose(selected_frequency, regular.FREQUENCY_HZ, atol=1e-8):
            raise ValueError(f"RAFT {group} reference is not at 16 s.")
        raft_by_group[group] = (responses, wave)

    wind_time, wind_rows = fit_export(args.wind_csv, "wind")
    nowind_time, nowind_rows = fit_export(args.nowind_csv, "no_wind")
    if wind_time.shape != nowind_time.shape or not np.allclose(
        wind_time, nowind_time, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Wind and no-wind runs must have identical time grids.")

    fits = wind_rows + nowind_rows
    summary = summarize_runs(fits, raft_by_group, args.summary_cycles)
    wind_effect = summarize_wind_effect(summary, args.summary_cycles)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "per_run_harmonic_fits.csv", fits)
    write_rows(args.output / "per_run_comparison_summary.csv", summary)
    write_rows(args.output / "wind_effect_summary.csv", wind_effect)
    plot_comparison(
        args.output / "primary_wf_wind_no_wind_vs_raft.png", summary,
        args.summary_cycles,
    )

    manifest = {
        "frequency_hz": regular.FREQUENCY_HZ,
        "period_s": regular.PERIOD_S,
        "summary_cycles": args.summary_cycles,
        "window_cycles": list(regular.WINDOW_CYCLES),
        "time_start_s": float(wind_time[0]),
        "time_end_s": float(wind_time[-1]),
        "sample_count_per_run": int(len(wind_time)),
        "sample_interval_s": float(np.median(np.diff(wind_time))),
        "inputs": {
            "wind_floater_csv": str(args.wind_csv),
            "no_wind_floater_csv": str(args.nowind_csv),
            "raft_reference_csv": str(args.raft_csv),
        },
        "raft_references": {
            "no_wind": {
                "csv": str(raft_paths["no_wind"]),
                "selected_frequency_hz": regular.FREQUENCY_HZ,
                "selected_wave_amplitude_m": abs(raft_by_group["no_wind"][1]),
                "wave_spectrum": "JONSWAP",
                "Hs_m": regular.RAFT_HS_M,
                "Tp_s": regular.PERIOD_S,
                "gamma": regular.RAFT_GAMMA,
                "turbine_status": "parked",
                "wind_speed_m_s": 0.0,
            },
            "wind": {
                "csv": str(raft_paths["wind"]),
                "selected_frequency_hz": regular.FREQUENCY_HZ,
                "selected_wave_amplitude_m": abs(raft_by_group["wind"][1]),
                "wave_spectrum": "JONSWAP",
                "Hs_m": regular.RAFT_HS_M,
                "Tp_s": regular.PERIOD_S,
                "gamma": regular.RAFT_GAMMA,
                "turbine_status": "parked",
                "wind_speed_m_s": RAFT_WIND_SPEED,
                "warning": "RAFT parked mode neglects rotor aerodynamic loads; this is not equivalent to OrcaFlex NPD wind loading.",
            },
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.wind_csv, args.nowind_csv, args.raft_csv)
        },
        "limitations": [
            "The wind-present OrcaFlex export is compared to RAFT with the same nominal 17.1 m/s input, but RAFT parked mode neglects rotor aerodynamic loads and does not reproduce OrcaFlex NPD wind loading.",
            "The YAML states wind speed 17.1 m/s, but exports alone do not prove active wind/aerodynamic load flags for each OrcaFlex run.",
            "Wave direction and motion channel frames/origins/filter definitions require OrcaFlex confirmation.",
            "Primary WF is the wave-band candidate; Dynamic is included as a separate channel family.",
            "The RAFT reference is a selected component from JONSWAP, not a pure regular-wave RAFT run.",
        ],
    }
    (args.output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = [
        "# 16 s Wind / No-Wind OrcaFlex Comparison Against RAFT",
        "",
        f"Frequency: {regular.FREQUENCY_HZ:.6f} Hz ({regular.PERIOD_S:g} s).",
        f"Wind and no-wind records each contain {len(wind_time)} samples from "
        f"{wind_time[0]:.1f} to {wind_time[-1]:.1f} s.",
        f"Summary window: trailing {args.summary_cycles} wave cycles "
        f"({args.summary_cycles / regular.FREQUENCY_HZ:.0f} s).",
        "",
        "See `per_run_comparison_summary.csv` for each OrcaFlex case versus the "
        "RAFT response scaled to that run's fitted wave amplitude. `wind_effect_summary.csv` "
        "reports wind-minus-no-wind differences; `per_run_harmonic_fits.csv` has all windows.",
        "",
        "## Interpretation limits",
        "",
        "- The wind-present OrcaFlex run must not be called a matched RAFT case: RAFT is zero-wind/parked and does not model the OrcaFlex NPD wind loading in this setup.",
        "- A separate RAFT singlef run uses 17.1 m/s input with the turbine parked, as requested; RAFT warns that parked mode neglects rotor aerodynamic loads, so this is not a full match to an OrcaFlex NPD wind case.",
        "- Confirm active OrcaFlex wind-load switches, direction conventions, and Primary WF/Dynamic axes and reference point before interpreting differences or signed phase.",
        "- Group replication is limited to one run in each condition; no between-seed variability estimate is available.",
        "- High fit residuals or strong window sensitivity mean the 16 s harmonic is not isolated; inspect the per-run fits before treating differences as RAOs.",
    ]
    (args.output / "comparison_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Comparison outputs saved under {args.output.resolve()}")
    print(f"Wind run: {args.wind_csv.name}")
    print(f"No-wind run: {args.nowind_csv.name}")
    print(f"Samples per run: {len(wind_time)}")
    print("RAFT references: parked singlef runs at 0 and 17.1 m/s wind input")


def run_raft(output: Path, label: str, *, wind_speed: float) -> None:
    """Run singlef with group-specific output paths and wind input."""
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable,
        str(regular.RAFT_SCRIPT),
        "--period", str(regular.PERIOD_S),
        "--height", str(regular.RAFT_HS_M),
        "--heading", "0",
        "--wind-speed", str(wind_speed),
        "--turbulence", "0",
        "--gamma", str(regular.RAFT_GAMMA),
        "--output", str(output / f"raft_{label}_single_frequency_response.csv"),
        "--full-output", str(output / f"raft_{label}_full_frequency_response.csv"),
        "--spectrum-plot", str(output / f"raft_{label}_jonswap_spectrum.png"),
    ], cwd=regular.ROOT, check=True)


def self_test() -> None:
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    time = np.arange(6001) * 0.1
    frequency = regular.FREQUENCY_HZ
    amplitude = 0.37
    phase = np.deg2rad(-43.0)
    values = 2.0 + amplitude * np.cos(2.0 * np.pi * frequency * time + phase)
    fit = regular.fit_harmonic(time, values, frequency)
    if not np.isclose(fit["amplitude"], amplitude, rtol=1e-12):
        raise AssertionError("Synthetic harmonic amplitude was not recovered.")
    if not np.isclose(fit["phase_deg"], np.rad2deg(phase), atol=1e-10):
        raise AssertionError("Synthetic harmonic phase was not recovered.")
    phase_mean = circular_mean_degrees([179.0, -179.0])
    if phase_mean is None or abs(abs(phase_mean) - 180.0) > 1e-10:
        raise AssertionError("Circular phase mean mishandles wraparound.")
    fits = []
    raft_by_group = {}
    for group, amplitude in (("no_wind", 2.0), ("wind", 3.0)):
        raft_by_group[group] = (
            {dof: {"complex_amplitude": complex(amplitude)}
             for dof in regular.DOFS}, 0.5 + 0j
        )
        for dof in regular.DOFS:
            fits.append({
                "group": group, "run": group, "channel_set": "Primary WF",
                "dof": dof, "unit": regular.UNIT_BY_DOF[dof], "cycles": 10,
                "wave_amplitude_m": 0.25, "motion_mean": 0.0,
                "motion_amplitude": 1.0, "motion_harmonic_rms": 1.0,
                "phase_relative_to_wave_deg": 0.0,
                "response_per_wave_amplitude": 4.0, "fit_residual_rms": 0.0,
            })
    summary = summarize_runs(fits, raft_by_group, 10)
    with TemporaryDirectory() as directory:
        with patch.object(plt, "close"):
            plot_comparison(Path(directory) / "comparison.png", summary, 10)
        figure = plt.gcf()
        for axis, expected in zip(figure.axes, ([1, 1, 2, 3], [4, 4, 4, 6])):
            heights = [[bar.get_height() for bar in container.patches]
                       for container in axis.containers
                       if hasattr(container, "patches")]
            if not np.allclose(heights, np.repeat(
                np.asarray(expected)[:, None], len(regular.DOFS), axis=1
            )):
                raise AssertionError("Comparison bars have incorrect amplitudes.")
        plt.close(figure)
    print("Self-test passed: harmonic fits, phase and comparison bars.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-csv", type=Path, default=WIND_FLOATER)
    parser.add_argument("--nowind-csv", type=Path, default=NOWIND_FLOATER)
    parser.add_argument("--raft-csv", type=Path, default=RAFT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-raft", action="store_true",
        help="Use the two RAFT reference CSVs already in the output folder.",
    )
    parser.add_argument(
        "--summary-cycles", type=int, choices=regular.WINDOW_CYCLES,
        default=SUMMARY_CYCLES,
    )
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
