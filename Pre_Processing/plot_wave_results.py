"""Regenerate ERA5 wave-analysis plots from saved result CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.switch_backend("Agg")


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = BASE_DIR / "wave_swell_wind_comparison_results"
METRIC_PLOTS = (
    ("wind_speed_10m", "10 m wind speed", "m/s"),
    ("wind_speed_100m", "100 m wind speed", "m/s"),
    ("wind_gust_10m", "10 m wind gust", "m/s"),
    ("wave_spectral_peakedness", "Wave spectral peakedness", ""),
)
DIRECTION_LABELS = {
    "wind_10m": "10 m wind",
    "wind_100m": "100 m wind",
    "mean_wave": "mean waves",
    "wind_waves": "wind waves",
    "total_swell": "total swell",
}


def read_csv_rows(
    path: Path, required_columns: set[str]
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def numeric(value: str, path: Path, column: str) -> float:
    try:
        return float(value) if value.strip() else float("nan")
    except ValueError as error:
        raise ValueError(
            f"Invalid number in {path.name}, column {column}: {value!r}"
        ) from error


def edges_from_bounds(
    rows: list[dict[str, str]], lower_column: str, upper_column: str,
    path: Path,
) -> np.ndarray:
    bounds = sorted({
        (numeric(row[lower_column], path, lower_column),
         numeric(row[upper_column], path, upper_column))
        for row in rows
    })
    if not bounds:
        raise ValueError(f"No bin bounds found in {path.name}")
    edges = np.array([bounds[0][0], *(upper for _, upper in bounds)])
    for index, (lower, upper) in enumerate(bounds):
        if not np.isclose(edges[index], lower) or upper <= lower:
            raise ValueError(
                f"Inconsistent or nonpositive bins in {path.name}"
            )
        if index and not np.isclose(bounds[index - 1][1], lower):
            raise ValueError(f"Noncontiguous bins in {path.name}")
    return edges


def matrix_from_rows(
    rows: list[dict[str, str]],
    path: Path,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hs_edges = edges_from_bounds(
        rows, "hs_bin_lower_m", "hs_bin_upper_m", path
    )
    period_edges = edges_from_bounds(
        rows, "peak_period_bin_lower_s", "peak_period_bin_upper_s", path
    )
    matrix = np.full((hs_edges.size - 1, period_edges.size - 1), np.nan)
    hs_indices = {value: index for index, value in enumerate(hs_edges[:-1])}
    period_indices = {
        value: index for index, value in enumerate(period_edges[:-1])
    }
    for row in rows:
        hs_lower = numeric(row["hs_bin_lower_m"], path, "hs_bin_lower_m")
        period_lower = numeric(
            row["peak_period_bin_lower_s"], path,
            "peak_period_bin_lower_s",
        )
        index = (hs_indices[hs_lower], period_indices[period_lower])
        if np.isfinite(matrix[index]):
            raise ValueError(f"Duplicate bin row in {path.name}: {index}")
        matrix[index] = numeric(row[value_column], path, value_column)
    return hs_edges, period_edges, matrix


def plot_scenario_joint(results_dir: Path, output_dir: Path) -> None:
    joint_path = results_dir / "scenario_joint_probabilities.csv"
    columns = {
        "scenario", "hs_bin_lower_m", "hs_bin_upper_m",
        "peak_period_bin_lower_s", "peak_period_bin_upper_s", "probability",
    }
    joint_rows = read_csv_rows(joint_path, columns)
    summary_rows = read_csv_rows(
        results_dir / "scenario_summary.csv",
        {"scenario", "filters", "valid_hours", "share_of_all_valid_hours"},
    )
    summaries = {row["scenario"]: row for row in summary_rows}
    scenarios = list(dict.fromkeys(row["scenario"] for row in joint_rows))
    if not scenarios:
        raise ValueError(
            "No scenarios found in scenario_joint_probabilities.csv"
        )
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(6 * len(scenarios), 6),
        constrained_layout=True,
        squeeze=False,
    )
    scenario_data = []
    positive_values = []
    for scenario in scenarios:
        rows = [row for row in joint_rows if row["scenario"] == scenario]
        hs_edges, period_edges, matrix = matrix_from_rows(
            rows, joint_path, "probability"
        )
        if not np.isclose(np.nansum(matrix), 1.0):
            raise ValueError(
                f"Scenario probabilities do not sum to 1: {scenario}"
            )
        positive_values.extend(matrix[np.isfinite(matrix) & (matrix > 0)])
        scenario_data.append((scenario, hs_edges, period_edges, matrix))

    norm = LogNorm(vmin=min(positive_values), vmax=max(positive_values))
    colormap = plt.get_cmap("turbo").copy()
    colormap.set_bad("#e6e6e6")
    for axis, (scenario, hs_edges, period_edges, matrix) in zip(
        axes[0], scenario_data
    ):
        image = axis.pcolormesh(
            hs_edges, period_edges,
            np.ma.masked_less_equal(matrix.T, 0), shading="auto",
            cmap=colormap, norm=norm,
        )
        summary = summaries.get(scenario, {})
        description = summary.get("filters", scenario)
        count = int(float(summary.get("valid_hours", 0)))
        share = float(summary.get("share_of_all_valid_hours", 0))
        axis.set(
            xlabel="Wind-wave significant height Hs (m)",
            ylabel="Peak wave period (s)",
            title=f"{description}\nn={count:,} ({share:.1%} of valid hours)",
        )
    figure.colorbar(
        image, ax=axes[0].tolist(), label="Conditional probability (log scale)"
    )
    figure.savefig(
        output_dir / "three_scenario_joint_probabilities.png", dpi=200
    )
    plt.close(figure)


def plot_full_and_workable(
    results_dir: Path, output_dir: Path, workability_hs_max: float,
) -> None:
    hs_rows = read_csv_rows(
        results_dir / "hs_workability_marginal.csv",
        {
            "hs_bin_lower_m", "hs_bin_upper_m", "bin_center",
            "full_probability", "workable_probability",
        },
    )
    period_rows = read_csv_rows(
        results_dir / "peak_period_workability_marginal.csv",
        {
            "peak_period_bin_lower_s", "peak_period_bin_upper_s",
            "bin_center", "full_probability", "workable_probability",
        },
    )
    hs_edges = edges_from_bounds(
        hs_rows, "hs_bin_lower_m", "hs_bin_upper_m",
        results_dir / "hs_workability_marginal.csv",
    )
    period_edges = edges_from_bounds(
        period_rows, "peak_period_bin_lower_s",
        "peak_period_bin_upper_s",
        results_dir / "peak_period_workability_marginal.csv",
    )
    joint_path = results_dir / "joint_probability.csv"
    full_rows = read_csv_rows(
        joint_path,
        {
            "hs_bin_lower_m", "hs_bin_upper_m",
            "peak_period_bin_lower_s", "peak_period_bin_upper_s", "value",
        },
    )
    workable_path = results_dir / "workable_joint_probability.csv"
    workable_rows = read_csv_rows(
        workable_path,
        {
            "hs_bin_lower_m", "hs_bin_upper_m",
            "peak_period_bin_lower_s", "peak_period_bin_upper_s", "value",
        },
    )
    full_hs, full_period, full_matrix = matrix_from_rows(
        full_rows, joint_path, "value"
    )
    retained_hs, retained_period, retained_matrix = matrix_from_rows(
        workable_rows, workable_path, "value"
    )
    if not (
        np.array_equal(hs_edges, full_hs)
        and np.array_equal(hs_edges, retained_hs)
        and np.array_equal(period_edges, full_period)
        and np.array_equal(period_edges, retained_period)
    ):
        raise ValueError("Marginal and joint-probability bins do not match")
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 10), constrained_layout=True
    )
    hs_centers = np.array([float(row["bin_center"]) for row in hs_rows])
    hs_bin_width = np.diff(hs_edges)
    axes[0, 0].bar(
        hs_centers - hs_bin_width * 0.2,
        [float(row["full_probability"]) for row in hs_rows],
        width=hs_bin_width * 0.4, label="All observations",
    )
    axes[0, 0].bar(
        hs_centers + hs_bin_width * 0.2,
        [float(row["workable_probability"]) for row in hs_rows],
        width=hs_bin_width * 0.4, label="Workable observations",
    )
    axes[0, 0].axvline(
        workability_hs_max, color="black", linestyle="--", linewidth=1
    )
    axes[0, 0].set(
        xlabel="Wind-wave significant height Hs (m)", ylabel="Probability",
        title="Hs marginal",
    )
    axes[0, 0].legend()
    period_centers = np.array([
        float(row["bin_center"]) for row in period_rows
    ])
    period_bin_width = np.diff(period_edges)
    axes[0, 1].bar(
        period_centers - period_bin_width * 0.2,
        [float(row["full_probability"]) for row in period_rows],
        width=period_bin_width * 0.4, label="All observations",
    )
    axes[0, 1].bar(
        period_centers + period_bin_width * 0.2,
        [float(row["workable_probability"]) for row in period_rows],
        width=period_bin_width * 0.4, label="Workable observations",
    )
    axes[0, 1].set(
        xlabel="Peak wave period (s)", ylabel="Probability",
        title="Peak period marginal",
    )
    axes[0, 1].legend()
    for axis, matrix, title in (
        (axes[1, 0], full_matrix, "Joint probability: all observations"),
        (
            axes[1, 1], retained_matrix,
            "Joint probability: workable observations",
        ),
    ):
        positive = np.concatenate([
            full_matrix[np.isfinite(full_matrix) & (full_matrix > 0)],
            retained_matrix[
                np.isfinite(retained_matrix) & (retained_matrix > 0)
            ],
        ])
        norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        colormap = plt.get_cmap("turbo").copy()
        colormap.set_bad("#e6e6e6")
        image = axis.pcolormesh(
            hs_edges, period_edges,
            np.ma.masked_less_equal(matrix.T, 0), shading="auto",
            cmap=colormap, norm=norm,
        )
        axis.set(
            xlabel="Wind-wave significant height Hs (m)",
            ylabel="Peak wave period (s)", title=title,
        )
        figure.colorbar(image, ax=axis, label="Probability (log scale)")
    figure.savefig(output_dir / "wave_state_distributions.png", dpi=200)
    plt.close(figure)


def plot_additional_metrics(results_dir: Path, output_dir: Path) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True
    )
    for axis, (name, title, unit) in zip(axes.flat, METRIC_PLOTS):
        path = results_dir / f"{name}_marginal.csv"
        if not path.is_file():
            axis.set_title(f"{title} (not available)")
            axis.set_axis_off()
            continue
        unit_suffix = {"m/s": "m_s**-1", "": "dimensionless"}[unit]
        lower_column = f"{name}_bin_lower_{unit_suffix}"
        upper_column = f"{name}_bin_upper_{unit_suffix}"
        rows = read_csv_rows(path, {
            lower_column, upper_column, "probability",
        })
        if not rows:
            axis.set_title(f"{title} (not available)")
            axis.set_axis_off()
            continue
        lower = np.array([
            numeric(row[lower_column], path, lower_column) for row in rows
        ])
        upper = np.array([
            numeric(row[upper_column], path, upper_column) for row in rows
        ])
        probability = np.array([
            numeric(row["probability"], path, "probability") for row in rows
        ])
        if np.any(upper <= lower) or not np.isclose(probability.sum(), 1.0):
            raise ValueError(
                f"Invalid marginal bins or probabilities in {path.name}"
            )
        density = probability / (upper - lower)
        axis.bar(lower, density, width=upper - lower, align="edge")
        xlabel = f"{title} ({unit})" if unit else title
        axis.set(xlabel=xlabel, ylabel="Density", title=title)
    figure.savefig(output_dir / "additional_metric_distributions.png", dpi=200)
    plt.close(figure)


def plot_wave_height_comparison(
    results_dir: Path, output_dir: Path
) -> None:
    path = results_dir / "wave_height_comparison.csv"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    if not path.is_file():
        for axis in axes:
            axis.text(
                0.5, 0.5, "Comparison data unavailable; rerun analysis",
                ha="center", va="center", transform=axis.transAxes,
            )
            axis.set_axis_off()
    else:
        rows = read_csv_rows(
            path, {"swh_m", "shww_m", "swh_minus_shww_m"}
        )
        combined = np.array([
            numeric(row["swh_m"], path, "swh_m") for row in rows
        ])
        wind_wave = np.array([
            numeric(row["shww_m"], path, "shww_m") for row in rows
        ])
        difference = np.array([
            numeric(row["swh_minus_shww_m"], path, "swh_minus_shww_m")
            for row in rows
        ])
        valid = (
            np.isfinite(combined) & np.isfinite(wind_wave)
            & np.isfinite(difference)
        )
        combined = combined[valid]
        wind_wave = wind_wave[valid]
        difference = difference[valid]
        if combined.size:
            maximum = max(float(combined.max()), float(wind_wave.max()), 1.0)
            image = axes[0].hexbin(
                wind_wave, combined, gridsize=55, mincnt=1, bins="log",
            )
            axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
            axes[0].set(
                xlim=(0, maximum), ylim=(0, maximum),
                xlabel="Wind-wave significant height shww (m)",
                ylabel="Combined significant height swh (m)",
                title="Hourly height comparison",
            )
            figure.colorbar(
                image, ax=axes[0], label="Hourly observations (log scale)"
            )
            axes[1].hist(difference, bins="auto")
            axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
            axes[1].set(
                xlabel="Combined minus wind-wave height, swh - shww (m)",
                ylabel="Hourly observations",
                title="Height increment distribution",
            )
        else:
            for axis in axes:
                axis.text(
                    0.5, 0.5, "No paired height observations",
                    ha="center", va="center", transform=axis.transAxes,
                )
                axis.set_axis_off()
    figure.savefig(output_dir / "wave_height_comparison.png", dpi=200)
    plt.close(figure)


def plot_direction_rose(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "wind_direction_frequency.csv"
    rows = read_csv_rows(
        path,
        {
            "direction_source", "direction_sector", "sector_center_deg",
            "probability",
        },
    )
    sources = list(dict.fromkeys(row["direction_source"] for row in rows))
    if not sources:
        raise ValueError(f"No direction frequencies found in {path.name}")
    figure, axes = plt.subplots(
        1, len(sources), figsize=(5.5 * len(sources), 5),
        subplot_kw={"projection": "polar"}, constrained_layout=True,
        squeeze=False,
    )
    for axis, source in zip(axes[0], sources):
        selected = [row for row in rows if row["direction_source"] == source]
        centers = np.array([
            numeric(row["sector_center_deg"], path, "sector_center_deg")
            for row in selected
        ])
        probabilities = np.array([
            numeric(row["probability"], path, "probability")
            for row in selected
        ])
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(
                f"Direction probabilities do not sum to 1: {source}"
            )
        angles = np.deg2rad(centers)
        axis.bar(angles, probabilities, width=np.deg2rad(22.5))
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_title(
            f"{DIRECTION_LABELS.get(source, source)} direction from"
        )
    figure.savefig(output_dir / "wind_direction_rose.png", dpi=200)
    plt.close(figure)


def plot_height_cdf(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "wave_height_cdf.csv"
    if not path.is_file():
        return
    rows = read_csv_rows(path, {"source", "sample", "hs_m", "cdf", "hours"})
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    labels = {"swh": "Combined", "shww": "Wind waves", "shts": "Total swell"}
    colors = {"swh": "#242424", "shww": "#0072b2", "shts": "#d55e00"}
    for source, label in labels.items():
        for sample in ("available", "common_three_heights"):
            selected = [row for row in rows if row["source"] == source
                        and row["sample"] == sample]
            if not selected:
                continue
            heights = np.array([float(row["hs_m"]) for row in selected])
            cdf = np.array([float(row["cdf"]) for row in selected])
            count = int(selected[0]["hours"])
            axis = axes[0] if sample == "available" else axes[1]
            axis.step(heights, cdf, where="post", color=colors[source],
                      label=f"{label}, n={count:,}")
            if sample == "available":
                positive = 1 - cdf > 0
                axes[2].step(heights[positive], (1 - cdf)[positive],
                             where="post", color=colors[source], label=label)
    for axis, title in zip(axes, (
        "Available observations per variable",
        "Common three-height observations",
        "Exceedance: available observations",
    )):
        axis.set(xlabel="Significant wave height Hs (m)", title=title)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No common observations", ha="center",
                      transform=axis.transAxes)
    axes[0].set(ylabel="Cumulative probability P(Hs <= h)", ylim=(0, 1))
    axes[1].set(ylabel="Cumulative probability P(Hs <= h)", ylim=(0, 1))
    axes[2].set(ylabel="Exceedance probability P(Hs > h)", yscale="log")
    figure.savefig(
        output_dir / "wave_height_cumulative_probability.png", dpi=200
    )
    plt.close(figure)


def plot_partition_diagnostics(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "wave_height_comparison.csv"
    if not path.is_file():
        return
    rows = read_csv_rows(path, {"swh_m", "shww_m"})
    if not rows or "energy_residual_m2" not in rows[0]:
        return
    columns = ("swh_m", "shww_m", "shts_m", "energy_swell_proxy_m",
               "energy_residual_m2", "wind_speed_10m_ms")
    data = {name: np.array([numeric(row[name], path, name) for row in rows])
            for name in columns}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    selected = (
        np.isfinite(data["shts_m"])
        & np.isfinite(data["energy_swell_proxy_m"])
    )
    if selected.any():
        image = axes[0].hexbin(
            data["shts_m"][selected], data["energy_swell_proxy_m"][selected],
            gridsize=45, mincnt=1, bins="log", cmap="turbo",
        )
        maximum = float(max(data["shts_m"][selected].max(),
                            data["energy_swell_proxy_m"][selected].max(), 1))
        axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=1)
        figure.colorbar(
            image, ax=axes[0], label="Hourly observations (log scale)"
        )
        axes[1].hist(
            data["energy_residual_m2"][selected], bins=80, color="#0072b2"
        )
        axes[1].axvline(0, color="black", linestyle="--")
    else:
        for axis in axes[:2]:
            axis.text(0.5, 0.5, "No three-height observations", ha="center",
                      transform=axis.transAxes)
    axes[0].set(xlabel="ERA5 total-swell Hs (m)",
                ylabel="Energy-based swell proxy (m)",
                title=f"Partition comparison, n={selected.sum():,}")
    axes[1].set(xlabel="swh^2 - shww^2 - shts^2 (m^2)",
                ylabel="Hourly observations",
                title="Partition energy residual")
    wind_selected = np.isfinite(data["wind_speed_10m_ms"])
    if wind_selected.any():
        image = axes[2].hexbin(
            data["wind_speed_10m_ms"][wind_selected],
            data["shww_m"][wind_selected],
            gridsize=55, mincnt=1, bins="log", cmap="turbo",
        )
        figure.colorbar(
            image, ax=axes[2], label="Hourly observations (log scale)"
        )
    axes[2].set(xlabel="10 m wind speed (m/s)", ylabel="Wind-wave Hs (m)",
                title="Concurrent wind and wind waves")
    figure.savefig(output_dir / "wave_partition_diagnostics.png", dpi=200)
    plt.close(figure)


def plot_directional_probability(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "directional_wave_probability.csv"
    if not path.is_file():
        return
    rows = read_csv_rows(path, {
        "height_source", "hs_m", "tp_s", "direction_deg",
        "probability", "paired_hours",
    })
    if not rows:
        print("No paired wave-direction observations; skipping 4D plot")
        return
    sources = list(dict.fromkeys(row["height_source"] for row in rows))
    figure = plt.figure(figsize=(9 * len(sources), 7), constrained_layout=True)
    probabilities = np.array([float(row["probability"]) for row in rows])
    norm = LogNorm(
        vmin=float(probabilities.min()), vmax=float(probabilities.max())
    )
    for index, source in enumerate(sources):
        selected = [row for row in rows if row["height_source"] == source]
        axis = figure.add_subplot(1, len(sources), index + 1, projection="3d")
        image = axis.scatter(
            [float(row["hs_m"]) for row in selected],
            [float(row["tp_s"]) for row in selected],
            [float(row["direction_deg"]) for row in selected],
            c=[float(row["probability"]) for row in selected], cmap="turbo",
            norm=norm, s=12, alpha=0.8,
        )
        hours = int(selected[0]["paired_hours"])
        axis.set(xlabel=f"{source} Hs (m)", ylabel="Combined peak period (s)",
                 zlabel="Wave direction from north (deg)", zlim=(0, 360),
                 title=f"{source}: direction-available hours, n={hours:,}")
        figure.colorbar(image, ax=axis, shrink=0.6,
                        label="Joint bin probability (log scale)")
    figure.savefig(output_dir / "direction_hs_tp_probability.png", dpi=200)
    plt.close(figure)


def plot_results(results_dir: Path, output_dir: Path) -> None:
    metadata_path = results_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Required result file not found: {metadata_path}"
        )
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    period = metadata.get("period", {})
    try:
        workability_hs_max = float(metadata["workability"]["hs_max_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Missing or invalid workability.hs_max_m in {metadata_path.name}"
        ) from error
    print(
        f"Plotting saved results for {period.get('start_year', '?')}-"
        f"{period.get('end_year', '?')} from {results_dir.resolve()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scenario_joint(results_dir, output_dir)
    plot_full_and_workable(results_dir, output_dir, workability_hs_max)
    plot_additional_metrics(results_dir, output_dir)
    plot_wave_height_comparison(results_dir, output_dir)
    plot_direction_rose(results_dir, output_dir)
    plot_height_cdf(results_dir, output_dir)
    plot_partition_diagnostics(results_dir, output_dir)
    plot_directional_probability(results_dir, output_dir)
    plot_variable_coverage(results_dir, output_dir)
    plot_jonswap_fits(results_dir, output_dir)
    plot_jonswap_constructed(results_dir, output_dir)
    print(f"Plots written to {output_dir.resolve()}")


def plot_variable_coverage(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "wave_variable_coverage.csv"
    if not path.is_file():
        return
    rows = read_csv_rows(path, {"month", "source", "coverage"})
    months = sorted({row["month"] for row in rows})
    sources = list(dict.fromkeys(row["source"] for row in rows))
    matrix = np.zeros((len(sources), len(months)))
    for row in rows:
        matrix[sources.index(row["source"]), months.index(row["month"])] = (
            float(row["coverage"])
        )
    figure, axis = plt.subplots(figsize=(15, 5), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ticks = [
        index for index, month in enumerate(months) if month.endswith("-01")
    ]
    axis.set_xticks(ticks, [months[index][:4] for index in ticks], rotation=45)
    axis.set_yticks(range(len(sources)), sources)
    axis.set(xlabel="Year", title="Monthly finite-data coverage per variable")
    figure.colorbar(image, ax=axis, label="Fraction of hours with data")
    figure.savefig(output_dir / "wave_variable_coverage.png", dpi=200)
    plt.close(figure)


def plot_jonswap_fits(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "jonswap_fits.csv"
    if not path.is_file():
        return
    fits = read_csv_rows(path, {"spectrum_id", "gamma"})
    curves = read_csv_rows(
        results_dir / "jonswap_spectral_curves.csv",
        {"spectrum_id", "frequency_hz", "observed_m2_per_hz",
         "fitted_m2_per_hz"},
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    for fit in fits[:6]:
        rows = [row for row in curves
                if row["spectrum_id"] == fit["spectrum_id"]]
        frequency = [float(row["frequency_hz"]) for row in rows]
        line, = axes[0].loglog(
            frequency, [float(row["observed_m2_per_hz"]) for row in rows],
            label=f"{fit['spectrum_id']}, gamma={float(fit['gamma']):.2f}",
        )
        axes[0].loglog(
            frequency, [float(row["fitted_m2_per_hz"]) for row in rows],
            linestyle="--", color=line.get_color(),
        )
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Elevation PSD (m^2/Hz)",
                title="Measured (solid), fitted (dashed); first six spectra")
    axes[0].legend(fontsize=8)
    gammas = [float(fit["gamma"]) for fit in fits]
    axes[1].hist(gammas, bins=np.linspace(1, 10, 19), color="#0072b2")
    axes[1].axvline(np.mean(gammas), color="black", linestyle="--")
    axes[1].set(xlabel="Fitted JONSWAP gamma", ylabel="Spectra",
                title=f"All fits: mean={np.mean(gammas):.2f}, n={len(fits)}")
    figure.savefig(output_dir / "jonswap_fitting.png", dpi=200)
    plt.close(figure)


def plot_jonswap_constructed(results_dir: Path, output_dir: Path) -> None:
    path = results_dir / "jonswap_constructed_states.csv"
    if not path.is_file():
        return
    states = read_csv_rows(
        path, {"state_id", "hs_m", "tp_s", "probability", "iec_gamma"})
    curves = read_csv_rows(
        results_dir / "jonswap_constructed_spectra.csv",
        {"state_id", "model", "gamma", "frequency_hz", "psd_m2_per_hz"},
    )
    distribution = read_csv_rows(
        results_dir / "jonswap_iec_gamma_distribution.csv",
        {"gamma_lower", "gamma_upper", "count",
         "probability_of_positive_hs_hours"},)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"iec": "#202020", "assumed_1": "#0072b2",
              "assumed_3.3": "#d55e00", "assumed_5": "#009e73"}
    for axis, state in zip(axes.flat, states):
        rows = [row for row in curves if row["state_id"] == state["state_id"]]
        for model in dict.fromkeys(row["model"] for row in rows):
            selected = [row for row in rows if row["model"] == model]
            label = "IEC rule" if model == "iec" else "Assumed"
            gamma = float(selected[0]["gamma"])
            axis.plot(
                [float(row["frequency_hz"]) for row in selected],
                [float(row["psd_m2_per_hz"]) for row in selected],
                label=f"{label} gamma={gamma:.3g}", color=colors.get(model),
                linestyle="-" if model == "iec" else "--",
                linewidth=2.2 if model == "iec" else 1.4,
                zorder=3 if model == "iec" else 2,
            )
        axis.set(
            xlim=(0, 3 / float(state["tp_s"])), xlabel="Frequency (Hz)",
            ylabel="Elevation PSD (m^2/Hz)",
            title=(f"{state['state_id'].replace('_', ' ')}: "
                   f"combined Hs={float(state['hs_m']):.3f} m, "
                   f"Tp={float(state['tp_s']):.3f} s\n"
                   "Occupied bin occurrence: "
                   f"{100 * float(state['probability']):.3f}%"),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "Constructed combined-sea JONSWAP spectra, not measured fits")
    figure.savefig(output_dir / "jonswap_constructed_spectra.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    lower = np.array([float(row["gamma_lower"]) for row in distribution])
    upper = np.array([float(row["gamma_upper"]) for row in distribution])
    probability = np.array(
        [float(row["probability_of_positive_hs_hours"])
         for row in distribution]
    )
    hours = sum(int(row["count"]) for row in distribution)
    axis.bar(
        lower,
        100 * probability,
        width=upper - lower,
        align="edge",
        color="#0072b2",
        edgecolor="white")
    axis.set(xlabel="IEC-rule gamma from hourly combined Hs/Tp (not fitted)",
             ylabel="Positive-Hs hours (%)", xlim=(lower[0], upper[-1]),
             title=("Hourly-weighted empirical-rule distribution: "
                    f"{hours:,} hours"))
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(output_dir / "jonswap_iec_gamma_distribution.png", dpi=200)
    plt.close(figure)


def run_self_tests() -> None:
    import tempfile
    import unittest

    class PlotDataTests(unittest.TestCase):
        def test_matrix_orientation_and_unspecified_bins(self):
            rows = [
                {
                    "hs_bin_lower_m": "0", "hs_bin_upper_m": "1",
                    "peak_period_bin_lower_s": "0",
                    "peak_period_bin_upper_s": "2",
                    "value": "0.25",
                },
                {
                    "hs_bin_lower_m": "1", "hs_bin_upper_m": "2",
                    "peak_period_bin_lower_s": "0",
                    "peak_period_bin_upper_s": "2",
                    "value": "0.75",
                },
            ]
            path = Path("matrix.csv")
            hs_edges, period_edges, matrix = matrix_from_rows(
                rows, path, "value"
            )
            np.testing.assert_array_equal(hs_edges, [0, 1, 2])
            np.testing.assert_array_equal(period_edges, [0, 2])
            np.testing.assert_allclose(matrix[:, 0], [0.25, 0.75])

        def test_plot_all_saved_figures(self):
            results_dir = BASE_DIR / "wave_results_2006_2020"
            with tempfile.TemporaryDirectory() as temporary:
                output_dir = Path(temporary)
                plot_results(results_dir, output_dir)
                expected = {
                    "three_scenario_joint_probabilities.png",
                    "wave_state_distributions.png",
                    "additional_metric_distributions.png",
                    "wave_height_comparison.png",
                    "wind_direction_rose.png",
                }
                self.assertEqual(
                    {path.name for path in output_dir.glob("*.png")}, expected
                )
                self.assertTrue(all(
                    path.stat().st_size > 0
                    for path in output_dir.glob("*.png")
                ))

        def test_height_comparison_plot_with_paired_data(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                results_dir = root / "results"
                output_dir = root / "plots"
                results_dir.mkdir()
                output_dir.mkdir()
                with (results_dir / "wave_height_comparison.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow([
                        "valid_time", "swh_m", "shww_m", "swh_minus_shww_m",
                    ])
                    writer.writerows([
                        ["2006-01-01T00:00", 1.0, 0.6, 0.4],
                        ["2006-01-01T01:00", 1.5, 1.2, 0.3],
                    ])
                plot_wave_height_comparison(results_dir, output_dir)
                self.assertGreater(
                    (output_dir / "wave_height_comparison.png").stat().st_size,
                    0,
                )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PlotDataTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--self-test", action="store_true",
        help=(
            "Run plotting regression tests against the default saved results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
        return
    output_dir = args.output_dir or args.results_dir / "plots"
    plot_results(args.results_dir, output_dir)


if __name__ == "__main__":
    main()
