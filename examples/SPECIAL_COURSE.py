"""
Starter script for the DTU special course (RAFT part).
 
Step 1: run the VolturnUS-S example unchanged and inspect the results structure.
Step 2: load wave-state probabilities from the ERA5 preprocessing output.
 
Run from the RAFT/examples folder (or point YAML at the file).
"""
import csv
import numpy as np
import yaml
import raft
from pathlib import Path
 
REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "designs" / "VolturnUS-S.yaml"
WAVE_DISTRIBUTION = (
    REPO_ROOT / "Pre_Processing" / "wave_results_2006_2020"
    / "raft_event_raster.csv"
)
STEADY_WIND_SPEED = 10.0
STEADY_TURBULENCE = 0.0
 
KEYS_DEFAULT = ["wind_speed", "wind_heading", "turbulence", "turbine_status",
                "yaw_misalign", "wave_spectrum", "wave_period", "wave_height",
                "wave_heading"]  # check against the header of YOUR yaml
 
 
def load_design(path=YAML):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)
 
 
def run(design):
    model = raft.Model(design)
    model.analyzeUnloaded()
    model.analyzeCases(display=1)
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


def plot_results(model, cases, probabilities, output_dir):
    """Save environmental and response plots for the analyzed cases."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wave_period = np.array([case[6] for case in cases])
    wave_height = np.array([case[7] for case in cases])
    probabilities = np.asarray(probabilities)

    fig, axis = plt.subplots(figsize=(8, 5))
    scatter = axis.scatter(wave_period, wave_height, c=probabilities,
                           s=40 + 900 * probabilities, cmap="viridis",
                           edgecolors="black", linewidths=0.3)
    axis.set_xlabel("Wave period used by RAFT (s)")
    axis.set_ylabel("Significant wave height (m)")
    axis.set_title("ERA5 wave cases and probabilities")
    fig.colorbar(scatter, ax=axis, label="Probability")
    fig.tight_layout()
    fig.savefig(output_dir / "wave_cases.png", dpi=180)
    plt.close(fig)

    metrics = model.results["case_metrics"]
    metric_names = ["surge_std", "sway_std", "heave_std", "roll_std",
                    "pitch_std", "yaw_std"]
    labels = ["Surge (m)", "Sway (m)", "Heave (m)", "Roll (deg)",
              "Pitch (deg)", "Yaw (deg)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, metric_name, label in zip(axes.flat, metric_names, labels):
        values = []
        for case_metrics in metrics.values():
            fowt_metrics = case_metrics[0]
            values.append(float(np.asarray(fowt_metrics[metric_name]).squeeze()))
        axis.scatter(wave_height, values, c=probabilities, cmap="viridis",
                     edgecolors="black", linewidths=0.3)
        axis.set_title(label)
        axis.set_xlabel("Significant wave height (m)")
        axis.set_ylabel("RAFT RMS response")
        axis.grid(alpha=0.25)
    fig.suptitle("RAFT RMS response versus wave height")
    fig.tight_layout()
    fig.savefig(output_dir / "response_rms_vs_height.png", dpi=180)
    plt.close(fig)

    wave_height_norm = plt.Normalize(vmin=np.min(wave_height),
                                     vmax=np.max(wave_height))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, metric_name, label in zip(axes.flat, metric_names, labels):
        values = [float(np.asarray(metrics[case_index][0][metric_name]).squeeze())
                  for case_index in range(len(cases))]
        scatter = axis.scatter(wave_period, values, c=wave_height,
                               cmap="plasma", norm=wave_height_norm,
                               edgecolors="black", linewidths=0.3)
        axis.set_title(label)
        axis.set_xlabel("Wave period (s)")
        axis.set_ylabel("RAFT RMS response")
        axis.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Significant wave height (m)")
    fig.suptitle("RAFT RMS response versus wave period")
    fig.tight_layout()
    fig.savefig(output_dir / "response_rms_vs_period.png", dpi=180)
    plt.close(fig)

    ranked_metric_names = []
    metric_rms = []
    for metric_name in metric_names:
        values = np.asarray([
            float(np.asarray(metrics[case_index][0][metric_name]).squeeze())
            for case_index in range(len(cases))
        ])
        metric_rms.append(values.mean())
        ranked_metric_names.append(metric_name)
    top3 = np.argsort(metric_rms)[::-1][:3]
    dominant_metric_names = [ranked_metric_names[idx] for idx in top3]
    dominant_labels = [labels[idx] for idx in top3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for axis, metric_name, label in zip(axes, dominant_metric_names,
                                        dominant_labels):
        values = [float(np.asarray(metrics[case_index][0][metric_name]).squeeze())
                  for case_index in range(len(cases))]
        scatter = axis.scatter(wave_period, values, c=wave_height,
                               cmap="plasma", norm=wave_height_norm,
                               edgecolors="black", linewidths=0.3)
        axis.set_title(label)
        axis.set_xlabel("Wave period (s)")
        axis.set_ylabel("RAFT RMS response")
        axis.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Significant wave height (m)")
    fig.suptitle("Most affected DOFs by mean RMS response")
    fig.tight_layout()
    fig.savefig(output_dir / "response_rms_vs_period_top3.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True,
                             sharey=True)
    for axis, metric_name, label in zip(axes.flat, metric_names, labels):
        values = np.array([
            float(np.asarray(metrics[case_index][0][metric_name]).squeeze())
            for case_index in range(len(cases))
        ])
        scatter = axis.scatter(wave_period, wave_height, c=values,
                               cmap="viridis", edgecolors="black",
                               linewidths=0.3)
        axis.set_title(label)
        axis.set_xlabel("Wave period (s)")
        axis.set_ylabel("Significant wave height (m)")
        axis.grid(alpha=0.25)
        fig.colorbar(scatter, ax=axis, label=f"{label} RMS")
    fig.suptitle("RAFT RMS response by wave period and significant wave height")
    fig.tight_layout()
    fig.savefig(output_dir / "response_rms_period_height_map.png", dpi=180)
    plt.close(fig)

    envelope_names = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    wave_periods = np.unique(wave_period)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, response_name, label in zip(axes.flat, envelope_names, labels):
        average = [metrics[i][0][response_name + "_avg"]
                   for i in range(len(cases))]
        maximum = [metrics[i][0][response_name + "_max"]
                   for i in range(len(cases))]
        minimum = [metrics[i][0][response_name + "_min"]
                   for i in range(len(cases))]
        for period in wave_periods:
            selected = wave_period == period
            order = np.argsort(wave_height[selected])
            height = wave_height[selected][order]
            axis.plot(height, np.asarray(average)[selected][order],
                      label=f"T={period:.0f} s")
            axis.plot(height, np.asarray(maximum)[selected][order],
                      color=axis.lines[-1].get_color(), linestyle="--")
            axis.plot(height, np.asarray(minimum)[selected][order],
                      color=axis.lines[-1].get_color(), linestyle=":")
        axis.set_title(label)
        axis.set_xlabel("Significant wave height (m)")
        axis.set_ylabel("Response")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, title="Wave period")
    fig.suptitle("RAFT response envelopes by wave height")
    fig.tight_layout()
    fig.savefig(output_dir / "response_envelopes.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, response_name, label in zip(axes.flat, envelope_names, labels):
        average = np.array([metrics[i][0][response_name + "_avg"]
                            for i in range(len(cases))])
        maximum = np.array([metrics[i][0][response_name + "_max"]
                            for i in range(len(cases))])
        minimum = np.array([metrics[i][0][response_name + "_min"]
                            for i in range(len(cases))])
        peak_magnitude = np.maximum(np.abs(maximum), np.abs(minimum))
        for period in wave_periods:
            selected = wave_period == period
            order = np.argsort(wave_height[selected])
            height = wave_height[selected][order]
            axis.fill_between(height, minimum[selected][order],
                              maximum[selected][order], alpha=0.12)
            axis.plot(height, peak_magnitude[selected][order], "o-",
                      label=f"T={period:.0f} s")
        axis.axhline(average[0], color="black", linestyle="--",
                     label="Mean offset")
        axis.set_title(label)
        axis.set_xlabel("Significant wave height (m)")
        axis.set_ylabel("Total response")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, title="Wave period")
    fig.suptitle("Total RAFT response including static offset")
    fig.tight_layout()
    fig.savefig(output_dir / "total_response.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, response_name, metric_name, label in zip(
            axes.flat, envelope_names, metric_names, labels):
        maximum = np.array([metrics[i][0][response_name + "_max"]
                            for i in range(len(cases))])
        rms = np.array([float(np.asarray(metrics[i][0][metric_name]).squeeze())
                        for i in range(len(cases))])
        order = np.argsort(wave_height)
        axis.plot(wave_height[order], maximum[order], "o-",
                  label="Maximum (mean + 3 RMS)")
        axis.plot(wave_height[order], rms[order], "s--",
                  label="RMS / SD")
        axis.set_title(label)
        axis.set_xlabel("Significant wave height (m)")
        axis.set_ylabel("Response")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Maximum and RMS response by wave height")
    fig.tight_layout()
    fig.savefig(output_dir / "response_max_rms_sd.png", dpi=180)
    plt.close(fig)

    weighted_rms = []
    for metric_name in metric_names:
        values = np.array([float(np.asarray(metrics[i][0][metric_name]).squeeze())
                           for i in range(len(cases))])
        weighted_rms.append(np.sum(values * probabilities) / probabilities.sum())
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, weighted_rms, color="steelblue")
    axis.set_ylabel("Probability-weighted RMS response")
    axis.set_title("Probability-weighted RAFT response")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "weighted_response.png", dpi=180)
    plt.close(fig)

    frequencies_hz = np.asarray(model.results["freq_rad"]) / (2.0 * np.pi)
    case_metrics = model.results["case_metrics"]
    common_case_indices = np.argsort(probabilities)[::-1][:5]
    psd_names = ["heave_PSD", "pitch_PSD", "surge_PSD", "yaw_PSD"]
    psd_labels = ["Heave PSD (m^2/Hz)", "Pitch PSD (deg^2/Hz)",
                  "Surge PSD (m^2/Hz)", "Yaw PSD (deg^2/Hz)"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, psd_name, label in zip(axes.flat, psd_names, psd_labels):
        for case_index in common_case_indices:
            spectrum = np.asarray(case_metrics[int(case_index)][0][psd_name])
            axis.plot(
                frequencies_hz,
                spectrum,
                label=(f"Case {case_index + 1}: Hs={cases[case_index][7]:.2f} m, "
                       f"T={cases[case_index][6]:.1f} s, "
                       f"P={probabilities[case_index]:.1%}"),
            )
        axis.set_title(label.split(" (")[0])
        axis.set_ylabel(label.split(" (")[1].rstrip(")"))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes[1]:
        axis.set_xlabel("Frequency (Hz)")
    fig.suptitle("Response PSDs for the five most probable ERA5 cases")
    fig.tight_layout()
    fig.savefig(output_dir / "common_case_psds.png", dpi=180)
    plt.close(fig)

    response_values = np.array([
        [
            float(np.asarray(case_metrics[case_index][0][metric_name]).squeeze())
            for metric_name in metric_names
        ]
        for case_index in range(len(cases))
    ])
    response_scales = np.maximum(response_values.max(axis=0), 1e-12)
    response_score = (response_values / response_scales).max(axis=1)
    critical_case_count = min(5, len(cases))
    critical_case_indices = np.argsort(response_score)[::-1][:critical_case_count]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, psd_name, label in zip(axes.flat, psd_names, psd_labels):
        for rank, case_index in enumerate(critical_case_indices, start=1):
            spectrum = np.asarray(case_metrics[case_index][0][psd_name])
            axis.plot(
                frequencies_hz,
                spectrum,
                label=(f"Response rank {rank}: Case {case_index + 1}, "
                       f"Hs={wave_height[case_index]:.2f} m, "
                       f"T={wave_period[case_index]:.1f} s, "
                       f"score={response_score[case_index]:.2f}"),
            )
        axis.set_title(label.split(" (")[0])
        axis.set_ylabel(label.split(" (")[1].rstrip(")"))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("Frequency (Hz)")
    fig.suptitle("Response PSDs for the five most critical wave cases")
    fig.tight_layout()
    fig.savefig(output_dir / "critical_case_psds.png", dpi=180)
    plt.close(fig)
 
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
    plot_results(model, rows, probs, REPO_ROOT / "results")
    return model, probs
 
 
if __name__ == "__main__":
    step2_workability()