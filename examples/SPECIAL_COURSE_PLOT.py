"""Create special-course figures from previously exported CSV data.

This script only reads existing CSV files and never runs RAFT.

Run after SPECIAL_COURSE.py has completed:

    python examples/SPECIAL_COURSE_PLOT.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "special_course_full"
NATURAL_MODE_LABELS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
NATURAL_FREQUENCY_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
)


def natural_frequency_label(index, frequency_hz):
    period = 1.0 / frequency_hz if frequency_hz > 0 else np.inf
    dof = NATURAL_MODE_LABELS[index] if index < 6 else "Mode"
    return f"f{index + 1} {dof}: {frequency_hz:.3f} Hz (T={period:.1f} s)"


def frequency_style(index):
    return {
        "color": NATURAL_FREQUENCY_COLORS[
            index % len(NATURAL_FREQUENCY_COLORS)
        ],
        "linestyle": "--",
        "linewidth": 1.1,
        "alpha": 0.55,
    }


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_natural_frequencies(path):
    return np.array([
        float(row["natural_frequency_hz"])
        for row in read_csv(path)
    ])


def plot_frequency_results(output_dir):
    rows = read_csv(output_dir / "plot_frequency_data.csv")
    summary = read_csv(output_dir / "plot_case_summary.csv")
    frequencies = np.array(sorted({float(row["frequency_hz"]) for row in rows}))
    by_case = defaultdict(list)
    for row in rows:
        by_case[int(row["case_index"])].append(row)
    probabilities = {
        int(row["case_index"]): float(row["probability"]) for row in summary
    }
    natural_frequencies = read_natural_frequencies(
        output_dir / "natural_frequencies.csv"
    )

    psd_names = ("heave_PSD", "pitch_PSD", "surge_PSD", "yaw_PSD")
    psd_labels = (
        "Heave PSD (m^2/Hz)", "Pitch PSD (deg^2/Hz)",
        "Surge PSD (m^2/Hz)", "Yaw PSD (deg^2/Hz)",
    )
    common_cases = sorted(
        probabilities, key=probabilities.get, reverse=True
    )[:5]
    summary_values = {
        int(row["case_index"]): np.array([
            float(row[name]) for name in (
                "surge_std", "sway_std", "heave_std",
                "roll_std", "pitch_std", "yaw_std",
            )
        ])
        for row in summary
    }
    scale = np.maximum(
        np.max(np.array(list(summary_values.values())), axis=0), 1e-12
    )
    scores = {
        case: np.max(summary_values[case] / scale) for case in summary_values
    }
    critical_cases = sorted(scores, key=scores.get, reverse=True)[:5]

    def plot_psds(cases, filename, title):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for axis, name, label in zip(axes.flat, psd_names, psd_labels):
            for case in cases:
                case_rows = by_case[case]
                case_summary = next(
                    row for row in summary
                    if int(row["case_index"]) == case
                )
                axis.plot(
                    frequencies,
                    [float(row[name]) for row in case_rows],
                    label=(
                        f"Case {case}: "
                        f"Hs={float(case_summary['wave_height_m']):.2f} m, "
                        f"Tp={float(case_summary['wave_period_s']):.1f} s, "
                        f"P={probabilities[case]:.1%}"
                    ),
                )
            for index, frequency in enumerate(natural_frequencies):
                axis.axvline(
                    frequency,
                    **frequency_style(index),
                    label=natural_frequency_label(index, frequency),
                )
            axis.set_title(label.split(" (")[0])
            axis.set_ylabel(label.split(" (")[1].rstrip(")"))
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        for axis in axes[1]:
            axis.set_xlabel("Frequency (Hz)")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.96)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    plot_psds(
        common_cases,
        "common_case_psds.png",
        "Response PSDs for the five most probable ERA5 cases",
    )
    plot_psds(
        critical_cases,
        "critical_case_psds.png",
        "Response PSDs for the five most critical wave cases",
    )

    rao_names = (
        ("surge_rao", "Surge", "m/m"),
        ("sway_rao", "Sway", "m/m"),
        ("heave_rao", "Heave", "m/m"),
        ("roll_rao", "Roll", "rad/m"),
        ("pitch_rao", "Pitch", "rad/m"),
        ("yaw_rao", "Yaw", "rad/m"),
        ("nacelle_surge_rao", "Nacelle surge", "m/m"),
    )
    def plot_raos(cases, rao_dir, filename_suffix, title_suffix):
        rao_dir.mkdir(parents=True, exist_ok=True)
        for name, label, units in rao_names:
            fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            for case in cases:
                case_summary = next(
                    row for row in summary
                    if int(row["case_index"]) == case
                )
                axis.plot(
                    frequencies,
                    [float(row[name]) for row in by_case[case]],
                    label=(
                        f"Case {case}: "
                        f"Hs={float(case_summary['wave_height_m']):.2f} m, "
                        f"Tp={float(case_summary['wave_period_s']):.1f} s, "
                        f"P={probabilities[case]:.1%}"
                    ),
                )
            for index, frequency in enumerate(natural_frequencies):
                axis.axvline(
                    frequency,
                    **frequency_style(index),
                    label=natural_frequency_label(index, frequency),
                )
            axis.set(
                xlabel="Frequency [Hz]",
                ylabel=f"RAO [{units}]",
                title=f"{label} displacement RAO ({title_suffix} cases)",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
            fig.savefig(
                rao_dir / f"{name}_{filename_suffix}.png",
                dpi=180,
            )
            plt.close(fig)

    plot_raos(common_cases, output_dir / "raos", "common_cases", "common")
    plot_raos(
        critical_cases,
        output_dir / "raos_critical",
        "critical_cases",
        "critical",
    )


def plot_sensitivity(
        output_dir, input_folder="sensitivity_wave_direction",
        plot_folder="sensitivity_wave_direction", case_group="common"):
    rows = read_csv(
        output_dir / input_folder
        / "wave_direction_sensitivity.csv"
    )
    headings = sorted({float(row["wave_heading_deg"]) for row in rows})
    metrics = (
        ("surge_std", "Surge RMS (m)"),
        ("sway_std", "Sway RMS (m)"),
        ("heave_std", "Heave RMS (m)"),
        ("roll_std", "Roll RMS (deg)"),
        ("pitch_std", "Pitch RMS (deg)"),
        ("yaw_std", "Yaw RMS (deg)"),
        ("AxRNA_std", "Nacelle longitudinal acceleration RMS (m/s^2)"),
    )
    plot_dir = output_dir / plot_folder
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in metrics:
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for rank in sorted({int(row["case_rank"]) for row in rows}):
            selected = [
                row for row in rows if int(row["case_rank"]) == rank
            ]
            selected.sort(key=lambda row: float(row["wave_heading_deg"]))
            case = selected[0]
            axis.plot(
                headings,
                [float(row[metric]) for row in selected],
                "o-",
                label=(
                    f"Rank {rank}, Case {case['original_case_index']}: "
                    f"Hs={float(case['wave_height_m']):.2f} m, "
                    f"Tp={float(case['wave_period_s']):.1f} s, "
                    f"P={float(case['probability']):.1%}"
                ),
            )
        axis.set(
            xlabel="Wave heading (deg)",
            ylabel=label,
            title=f"{label} versus wave heading ({case_group} cases)",
            xticks=headings,
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        filename = metric.replace("_std", "") + "_wave_heading.png"
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)


def main():
    required = (
        OUTPUT_DIR / "plot_frequency_data.csv",
        OUTPUT_DIR / "plot_case_summary.csv",
        OUTPUT_DIR / "natural_frequencies.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run SPECIAL_COURSE.py first; missing: " + ", ".join(missing)
        )
    plot_frequency_results(OUTPUT_DIR)
    sensitivity_csv = (
        OUTPUT_DIR / "sensitivity_wave_direction"
        / "wave_direction_sensitivity.csv"
    )
    if sensitivity_csv.exists():
        plot_sensitivity(OUTPUT_DIR)
    critical_sensitivity_csv = (
        OUTPUT_DIR / "sensitivity_wave_direction_critical"
        / "wave_direction_sensitivity.csv"
    )
    if critical_sensitivity_csv.exists():
        plot_sensitivity(
            OUTPUT_DIR,
            input_folder="sensitivity_wave_direction_critical",
            plot_folder="sensitivity_wave_direction_critical",
            case_group="critical",
        )
    else:
        print(
            "Critical sensitivity data not found. To calculate it separately, "
            "run examples/SPECIAL_COURSE_CRITICAL_SENSITIVITY.py."
        )
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
