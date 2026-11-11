"""Calculate wave-direction sensitivity for the five critical cases only.

This is intentionally separate from SPECIAL_COURSE_PLOT.py. It runs RAFT
explicitly and uses the already exported case summary to select the cases.
"""
import csv
from pathlib import Path

import numpy as np

from SPECIAL_COURSE import (
    REPO_ROOT,
    run_wave_direction_sensitivity,
    load_design,
    wave_distribution_to_cases,
)


OUTPUT_DIR = REPO_ROOT / "results" / "special_course_full"


def critical_case_indices(summary_path, count=5):
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary_names = (
        "surge_std", "sway_std", "heave_std",
        "roll_std", "pitch_std", "yaw_std",
    )
    values = np.array([
        [float(row[name]) for name in summary_names]
        for row in rows
    ])
    scale = np.maximum(np.max(values, axis=0), 1e-12)
    scores = np.max(values / scale, axis=1)
    return np.argsort(scores)[::-1][:count]


def main():
    summary_path = OUTPUT_DIR / "plot_case_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Run SPECIAL_COURSE.py first; missing: " + str(summary_path)
        )

    design = load_design()
    cases, probabilities = wave_distribution_to_cases()
    design["cases"]["keys"] = [
        "wind_speed", "wind_heading", "turbulence", "turbine_status",
        "yaw_misalign", "wave_spectrum", "wave_period", "wave_height",
        "wave_heading",
    ]
    design["cases"]["data"] = cases
    indices = critical_case_indices(summary_path)
    run_wave_direction_sensitivity(
        design,
        cases,
        probabilities,
        OUTPUT_DIR,
        indices,
        "sensitivity_wave_direction_critical",
    )


if __name__ == "__main__":
    main()
