"""
Starter script for the DTU special course (RAFT part).
 
Step 1: run the VolturnUS-S example unchanged and inspect the results structure.
Step 2: load wave-state probabilities from the ERA5 preprocessing output.
 
Run from the RAFT/examples folder (or point YAML at the file).
"""
import csv
import copy
import numpy as np
import yaml
import raft
from raft.helpers import JONSWAP
from pathlib import Path
 
REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "designs" / "VolturnUS-S.yaml"
WAVE_DISTRIBUTION = (
    REPO_ROOT / "Pre_Processing" / "wave_results_2006_2020"
    / "raft_event_raster.csv"
)
STEADY_WIND_SPEED = 10.0
STEADY_TURBULENCE = 0.0
SENSITIVITY_WAVE_HEADINGS = (0.0, 15.0, 30.0, 45.0, 60.0)
 
KEYS_DEFAULT = ["wind_speed", "wind_heading", "turbulence", "turbine_status",
                "yaw_misalign", "wave_spectrum", "wave_period", "wave_height",
                "wave_heading"]  # check against the header of YOUR yaml


def load_design(path=YAML):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
 
 
def run(design, display=1):
    model = raft.Model(design)
    model.analyzeUnloaded()
    model.analyzeCases(display=display)
    return model
 
 
def wave_distribution_to_cases(
        csv_path=WAVE_DISTRIBUTION, heading=0.0,
        wind_speed=STEADY_WIND_SPEED, turbulence=STEADY_TURBULENCE,
        min_prob=1e-6):
    """Convert the ERA5 wave-state probability CSV into RAFT cases.

    The CSV is produced by ``Pre_Processing/wave_state_distribution.py``.
    Each retained bin becomes one JONSWAP case at the supplied bin centre.
    Retained conditional probabilities are used and renormalized. Wind is
    deliberately held fixed for this study and is not read from the CSV.
    """
    rows, probabilities = [], []
    required = {
        "hs_center_m", "tp_center_s", "retained_conditional_probability",
    }
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                f"{csv_path} must contain columns {sorted(required)}"
            )
        for record in reader:
            probability = float(record["retained_conditional_probability"])
            if not np.isfinite(probability) or probability < min_prob:
                continue
            hs = float(record["hs_center_m"])
            period = float(record["tp_center_s"])
            rows.append([
                float(wind_speed), heading, float(turbulence), "parked", 0,
                "JONSWAP", period, hs, heading,
            ])
            probabilities.append(probability)

    if not rows:
        raise ValueError(f"No wave cases found in {csv_path}")
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()
    print(
        f"Loaded {len(rows)} wave cases from {csv_path} "
        f"(wind={wind_speed:g} m/s, turbulence={turbulence:g})"
    )
    return rows, probabilities


def save_natural_frequencies(model, output_dir):
    """Solve and save the linearized system natural frequencies and periods."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies_hz, modes = model.solveEigen(display=0)
    frequencies_hz = np.real(np.asarray(frequencies_hz, dtype=float))

    mode_path = output_dir / "natural_modes.json"
    if len(model.fowtList) == 1 and hasattr(
            model.fowtList[0], "write_modes_json"):
        model.fowtList[0].write_modes_json(
            str(mode_path),
            frequencies_hz,
            modes,
        )

    with (output_dir / "natural_frequencies.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "natural_frequency_hz", "natural_period_s"])
        for mode, frequency_hz in enumerate(frequencies_hz, start=1):
            writer.writerow([
                mode,
                frequency_hz,
                1.0 / frequency_hz if frequency_hz > 0.0 else np.inf,
            ])

    print(f"Saved natural frequencies to {output_dir / 'natural_frequencies.csv'}")
    if mode_path.exists():
        print(f"Saved mode shapes to {mode_path}")
    return frequencies_hz, modes


def save_plot_data(model, cases, probabilities, output_dir):
    """Save all frequency-domain data required by the plotting script."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies_rad = np.asarray(model.results["freq_rad"], dtype=float)
    frequencies_hz = frequencies_rad / (2.0 * np.pi)
    fowt = model.fowtList[0]
    metrics = model.results["case_metrics"]
    response_names = (
        ("surge_RA", False),
        ("sway_RA", False),
        ("heave_RA", False),
        ("roll_RA", True),
        ("pitch_RA", True),
        ("yaw_RA", True),
        ("nacelle_surge_RA", False),
    )
    psd_names = ("heave_PSD", "pitch_PSD", "surge_PSD", "yaw_PSD")
    summary_names = (
        "surge_std", "sway_std", "heave_std", "roll_std", "pitch_std",
        "yaw_std",
    )

    with (output_dir / "plot_frequency_data.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_index", "probability", "wave_height_m", "wave_period_s",
            "frequency_hz", *psd_names,
            "surge_rao", "sway_rao", "heave_rao", "roll_rao",
            "pitch_rao", "yaw_rao", "nacelle_surge_rao",
        ])
        for case_index, case in enumerate(cases):
            case_metrics = metrics[case_index][0]
            wave_spectrum = JONSWAP(
                frequencies_rad,
                float(case[7]),
                float(case[6]),
                Gamma=float(case[9]) if len(case) > 9 else 0.0,
            )
            wave_amplitude = np.sqrt(2.0 * wave_spectrum * fowt.dw)
            responses = []
            for response_name, rotational in response_names:
                response = np.asarray(case_metrics[response_name])[0]
                response = np.asarray(response).squeeze()
                if response.ndim != 1 or response.size != len(frequencies_hz):
                    raise ValueError(
                        f"{response_name} must contain one frequency series; "
                        f"got shape {np.asarray(case_metrics[response_name]).shape}"
                    )
                if rotational:
                    response = response * np.pi / 180.0
                responses.append(np.abs(np.divide(
                    response,
                    wave_amplitude,
                    out=np.zeros_like(response, dtype=complex),
                    where=wave_amplitude > 0.0,
                )))
            psds = [np.asarray(case_metrics[name], dtype=float) for name in psd_names]
            for frequency_index, frequency_hz in enumerate(frequencies_hz):
                writer.writerow([
                    case_index + 1,
                    probabilities[case_index],
                    case[7],
                    case[6],
                    frequency_hz,
                    *(psd[frequency_index] for psd in psds),
                    *(response[frequency_index] for response in responses),
                ])

    with (output_dir / "plot_case_summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_index", "probability", "wave_height_m", "wave_period_s",
            *summary_names,
        ])
        for case_index, case in enumerate(cases):
            case_metrics = metrics[case_index][0]
            writer.writerow([
                case_index + 1,
                probabilities[case_index],
                case[7],
                case[6],
                *(float(np.asarray(case_metrics[name]).squeeze())
                  for name in summary_names),
            ])

    print(f"Saved plotting data to {output_dir / 'plot_frequency_data.csv'}")
    print(f"Saved case summary to {output_dir / 'plot_case_summary.csv'}")


def run_wave_direction_sensitivity(
        design, cases, probabilities, output_dir, case_indices,
        folder_name, headings=SENSITIVITY_WAVE_HEADINGS):
    """Run selected cases at several wave headings."""
    output_dir = Path(output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities, dtype=float)
    case_indices = np.asarray(case_indices, dtype=int)
    headings = tuple(float(heading) for heading in headings)

    sensitivity_cases = []
    case_metadata = []
    for rank, case_index in enumerate(case_indices, start=1):
        for heading in headings:
            case = list(cases[int(case_index)])
            case[8] = heading
            sensitivity_cases.append(case)
            case_metadata.append((rank, int(case_index), heading))

    sensitivity_design = copy.deepcopy(design)
    sensitivity_design["cases"] = {
        "keys": list(KEYS_DEFAULT),
        "data": sensitivity_cases,
    }
    sensitivity_model = run(sensitivity_design, display=0)
    metrics = sensitivity_model.results["case_metrics"]

    channel_specs = (
        ("surge_std", "Surge RMS (m)"),
        ("sway_std", "Sway RMS (m)"),
        ("heave_std", "Heave RMS (m)"),
        ("roll_std", "Roll RMS (deg)"),
        ("pitch_std", "Pitch RMS (deg)"),
        ("yaw_std", "Yaw RMS (deg)"),
        ("AxRNA_std", "Nacelle longitudinal acceleration RMS (m/s²)"),
    )
    csv_path = output_dir / "wave_direction_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "case_rank", "original_case_index", "probability",
            "wave_height_m", "wave_period_s", "wave_heading_deg",
            *(name for name, _ in channel_specs),
        ])
        for sensitivity_index, (rank, case_index, heading) in enumerate(
                case_metadata):
            fowt_metrics = metrics[sensitivity_index][0]
            values = []
            for name, _ in channel_specs:
                value = np.asarray(fowt_metrics[name]).squeeze()
                values.append(float(np.ravel(value)[0]))
            writer.writerow([
                rank,
                case_index + 1,
                probabilities[case_index],
                cases[case_index][7],
                cases[case_index][6],
                heading,
                *values,
            ])

    print(f"Saved wave-direction sensitivity results to {csv_path}")
    print(f"Wave headings analyzed: {', '.join(f'{h:g}°' for h in headings)}")
    return sensitivity_model


def step2_workability():
    design = load_design()

    rows, probs = wave_distribution_to_cases()

    design["cases"]["keys"] = KEYS_DEFAULT
    design["cases"]["data"] = rows

    model = run(design)
    total_cases = len(rows)
    print(f"Case 1 out of {total_cases}:")
    print("Available result groups:", model.results.keys())
    print(
        "Available case-metric fields:",
        model.results["case_metrics"].keys(),
    )

    np.save(REPO_ROOT / "case_probs.npy", probs)
    output_dir = REPO_ROOT / "results" / "special_course_full"
    natural_frequencies_hz, _ = save_natural_frequencies(model, output_dir)
    save_plot_data(model, rows, probs, output_dir)
    probabilities = np.asarray(probs, dtype=float)
    common_case_indices = np.argsort(probabilities)[::-1][:5]
    summary_names = (
        "surge_std", "sway_std", "heave_std",
        "roll_std", "pitch_std", "yaw_std",
    )
    summary_values = np.array([
        [
            float(np.asarray(model.results["case_metrics"][case_index][0][name]).squeeze())
            for name in summary_names
        ]
        for case_index in range(len(rows))
    ])
    scale = np.maximum(np.max(summary_values, axis=0), 1e-12)
    scores = np.max(summary_values / scale, axis=1)
    critical_case_indices = np.argsort(scores)[::-1][:5]
    run_wave_direction_sensitivity(
        design, rows, probs, output_dir, common_case_indices,
        "sensitivity_wave_direction",
    )
    run_wave_direction_sensitivity(
        design, rows, probs, output_dir, critical_case_indices,
        "sensitivity_wave_direction_critical",
    )
    return model, probs

 
 
if __name__ == "__main__":
    step2_workability()