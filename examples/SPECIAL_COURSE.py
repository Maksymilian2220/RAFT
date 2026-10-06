"""
Starter script for the DTU special course (RAFT part).
 
Step 1: run the VolturnUS-S example unchanged and inspect the results structure.
Step 2: replace the cases with scatter-diagram bins; probabilities stay OUTSIDE the yaml.
 
Untested: written without RAFT installed. Check key names against your RAFT version.
Run from the RAFT/examples folder (or point YAML at the file).
"""
import numpy as np
import yaml
import raft
from pathlib import Path
 
REPO_ROOT = Path(__file__).resolve().parents[1]
YAML = REPO_ROOT / "designs" / "VolturnUS-S.yaml"
 
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
 
 
# ---------------------------------------------------------------- step 1
def step1_inspect():
    design = load_design()
    print("cases keys:", design["cases"]["keys"])
    print("n cases   :", len(design["cases"]["data"]))
    model = run(design)
    cm = model.results["case_metrics"]
    print("type of case_metrics:", type(cm))
    # Look at the structure once and adapt the post-processing to it
    first = cm[0] if isinstance(cm, (list, dict)) and 0 in (range(len(cm)) if isinstance(cm, list) else cm) else cm
    print(first if not isinstance(first, dict) else list(first.keys()))
    return model
 
 
# ---------------------------------------------------------------- step 2
def scatter_to_cases(hs_bins, tp_bins, prob, u_of_hs, heading=0.0,
                     turbulence=0.1, status="operating", min_prob=1e-6):
    """
    hs_bins, tp_bins : 1D arrays of bin centres
    prob             : 2D array [len(hs_bins), len(tp_bins)], sums to <= 1
    u_of_hs          : function Hs -> conditional mean wind speed (your metocean model)
    Returns (rows, probs): RAFT case rows and a matching probability array.
    """
    rows, probs = [], []
    for i, hs in enumerate(hs_bins):
        for j, tp in enumerate(tp_bins):
            p = prob[i, j]
            if p < min_prob:
                continue
            rows.append([float(u_of_hs(hs)), heading, turbulence, status, 0,
                         "JONSWAP", float(tp), float(hs), heading])
            probs.append(p)
    return rows, np.array(probs)


def grib_to_cases(grib_path, wind_speed, latitude, longitude,
                  hs_edges=None, tp_edges=None, heading=0.0,
                  turbulence=0.1, min_prob=1e-4):
    """Convert hourly GRIB wave data at one grid point into RAFT cases.

    The supplied GRIB file contains ``swh`` and ``mwp`` but no wind speed,
    so ``wind_speed`` must come from another source or be an explicit study
    assumption.
    """
    import xarray as xr

    if hs_edges is None:
        hs_edges = np.arange(0.0, 8.5, 0.5)
    if tp_edges is None:
        tp_edges = np.arange(2.0, 20.0, 2.0)

    with xr.open_dataset(grib_path, engine="cfgrib") as dataset:
        point = dataset.sel(latitude=latitude, longitude=longitude,
                            method="nearest")
        hs = np.asarray(point["swh"].values, dtype=float)
        tp = np.asarray(point["mwp"].values, dtype=float)

    valid = np.isfinite(hs) & np.isfinite(tp)
    counts, _, _ = np.histogram2d(hs[valid], tp[valid],
                                  bins=[hs_edges, tp_edges])
    probabilities = counts / valid.sum()
    hs_centres = 0.5 * (hs_edges[:-1] + hs_edges[1:])
    tp_centres = 0.5 * (tp_edges[:-1] + tp_edges[1:])

    rows, probs = [], []
    for i, j in np.argwhere(probabilities > min_prob):
        rows.append([float(wind_speed), heading, turbulence, "operating", 0,
                     "JONSWAP", float(tp_centres[j]), float(hs_centres[i]),
                     heading])
        probs.append(probabilities[i, j])

    return rows, np.asarray(probs)
 
 
def hindcast_to_cases(csv_path, edges=None, min_prob=1e-4, heading=0.0,
                      z_ref=None, z_hub=150.0, shear=0.12,
                      cut_in=3.0, cut_out=25.0, turbulence=0.1):
    """
    Build RAFT case rows + probabilities from an hourly hindcast CSV with
    columns time, ws, hs, tp (adapt names!). Wind speed is extrapolated to hub
    height only if z_ref (height of the wind data, e.g. 10) is given.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=["time"])          # adapt names/format
    df = df.dropna(subset=["ws", "hs", "tp"])
    if z_ref is not None:
        df["ws"] = df["ws"] * (z_hub / z_ref) ** shear
 
    if edges is None:
        edges = [np.arange(0, 27, 2), np.arange(0, 8.5, 0.5), np.arange(2, 20, 2)]
    counts, _ = np.histogramdd(df[["ws", "hs", "tp"]].to_numpy(), bins=edges)
    n_in, n_all = counts.sum(), len(df)
    print(f"samples inside bins: {n_in:.0f} / {n_all} ({n_in / n_all:.1%})")
    prob = counts / n_all            # divide by ALL samples so lost mass is visible
    centres = [0.5 * (e[:-1] + e[1:]) for e in edges]
 
    rows, probs = [], []
    for i, j, k in np.argwhere(prob > min_prob):
        u = centres[0][i]
        status = "operating" if cut_in <= u <= cut_out else "parked"
        rows.append([float(u), heading, turbulence, status, 0,
                     "JONSWAP", float(centres[2][k]), float(centres[1][j]), heading])
        probs.append(prob[i, j, k])
    probs = np.array(probs)
    print(f"{len(rows)} cases, covering {probs.sum():.1%} of the record")
    return rows, probs
 
 
def extreme_gaussian(sigma, duration_s, tz_s):
    """Expected maximum of a zero-mean Gaussian process over duration_s.
    tz_s = mean zero-crossing period (ASSUMPTION: take it from the response
    spectrum rather than from Tp if you can)."""
    n = duration_s / tz_s
    return sigma * np.sqrt(2.0 * np.log(n))


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
    axis.set_title("GRIB wave cases and probabilities")
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

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, metric_name, label in zip(axes.flat, metric_names, labels):
        values = [float(np.asarray(metrics[case_index][0][metric_name]).squeeze())
                  for case_index in range(len(cases))]
        scatter = axis.scatter(wave_period, values, c=wave_height,
                               cmap="plasma", edgecolors="black",
                               linewidths=0.3)
        axis.set_title(label)
        axis.set_xlabel("Wave period (s)")
        axis.set_ylabel("RAFT RMS response")
        axis.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes, label="Significant wave height (m)")
    fig.suptitle("RAFT RMS response versus wave period")
    fig.tight_layout()
    fig.savefig(output_dir / "response_rms_vs_period.png", dpi=180)
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
    fig.suptitle("Response PSDs for the five most probable GRIB cases")
    fig.tight_layout()
    fig.savefig(output_dir / "common_case_psds.png", dpi=180)
    plt.close(fig)
 
 
def step2_workability():
    design = load_design()

    rows, probs = grib_to_cases(
        REPO_ROOT / "examples" / "data.grib",
        wind_speed=12.0,       # temporary assumption
        latitude=42.5,
        longitude=-11.0,
    )

    design["cases"]["keys"] = KEYS_DEFAULT
    design["cases"]["data"] = rows

    model = run(design)
    print(model.results.keys())
    print(model.results["case_metrics"].keys())

    np.save(REPO_ROOT / "case_probs.npy", probs)
    plot_results(model, rows, probs, REPO_ROOT / "results")
    return model, probs
 
 
if __name__ == "__main__":
    step2_workability()