"""Analyze ERA5 wave states from archived NetCDF files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "era5_monthly"
START_YEAR = 2006
END_YEAR = 2020
DEFAULT_OUTPUT = BASE_DIR / f"wave_results_{START_YEAR}_{END_YEAR}"
SITE_LATITUDE = 58.6296
SITE_LONGITUDE = -3.8403666667
DOWNLOAD_AREA = (59.0, -4.0, 58.75, -3.75)
HS_BIN = 0.25
PERIOD_BIN = 0.25
WORKABILITY_HS_MAX = 2.5
WIND_SPEED_MAX = 20.0
HEIGHT_VARIABLE = "shww"
PERIOD_VARIABLE = "pp1d"
WAVE_METRICS = (HEIGHT_VARIABLE, PERIOD_VARIABLE)
OPTIONAL_WAVE_METRICS = (
    "wsp", "mwp", "mwd", "mdww", "mdts", "mp2", "mpww", "mpts",
    "wdw", "dwww",
    "dwps", "wss", "wsk", "hmax", "tmax", "wmb",
)
WIND_METRICS = ("u10", "v10", "u100", "v100", "i10fg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--hs-bin", type=float, default=HS_BIN)
    parser.add_argument("--period-bin", type=float, default=PERIOD_BIN)
    parser.add_argument(
        "--workability-hs-max", type=float, default=WORKABILITY_HS_MAX
    )
    parser.add_argument("--wind-speed-max", type=float, default=WIND_SPEED_MAX)
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run built-in regression tests without reading ERA5 archives.",
    )
    return parser.parse_args()


def read_archive(path: Path, temporary_dir: Path, index: int):
    try:
        with ZipFile(path) as archive:
            members = archive.namelist()
            streams = {
                "wave": [
                    name for name in members
                    if "wave_stepType-instant.nc" in name
                ],
                "wind": [
                    name for name in members
                    if "oper_stepType-instant.nc" in name
                ],
            }
            for label, matches in streams.items():
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {label} stream in {path.name}; "
                        f"found {len(matches)}"
                    )
            extracted = temporary_dir / f"wave_stream_{index:04d}.nc"
            extracted.write_bytes(archive.read(streams["wave"][0]))
            wind_extracted = temporary_dir / f"wind_stream_{index:04d}.nc"
            wind_extracted.write_bytes(archive.read(streams["wind"][0]))
    except BadZipFile as error:
        raise ValueError(f"Not a valid ERA5 ZIP archive: {path}") from error

    with (
        xr.open_dataset(extracted, engine="netcdf4") as wave_dataset,
        xr.open_dataset(wind_extracted, engine="netcdf4") as wind_dataset,
    ):
        missing_wave = set(WAVE_METRICS) - set(wave_dataset.data_vars)
        missing_wind = {"u10", "v10"} - set(wind_dataset.data_vars)
        if missing_wave or missing_wind:
            raise KeyError(
                f"{path.name} missing wave variables {sorted(missing_wave)} "
                f"or wind variables {sorted(missing_wind)}"
            )

        times = np.asarray(wave_dataset["valid_time"].values).astype(
            "datetime64[ns]"
        )
        wind_times = np.asarray(wind_dataset["valid_time"].values).astype(
            "datetime64[ns]"
        )
        if not np.array_equal(times, wind_times):
            raise ValueError(f"Wave/wind timestamps differ in {path.name}")

        wave_dataset = wave_dataset.sel(
            latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE,
            method="nearest",
        )
        latitude = float(wave_dataset["latitude"].values)
        longitude = float(wave_dataset["longitude"].values)
        metrics = {}
        units = {}
        for name in (*WAVE_METRICS, *OPTIONAL_WAVE_METRICS):
            if name not in wave_dataset.data_vars:
                metrics[name] = np.full(times.size, np.nan)
                units[name] = "unavailable in source archive"
                continue
            variable = wave_dataset[name]
            values = np.asarray(variable.values, dtype=float).reshape(-1)
            if values.size == 1:
                values = np.full(times.size, values[0])
            metrics[name] = values
            units[name] = variable.attrs.get("units", "unknown")

        wind_at_wave_point = wind_dataset.sel(
            latitude=latitude, longitude=longitude, method="nearest"
        )
        selected_latitude = float(wind_at_wave_point["latitude"].values)
        selected_longitude = float(wind_at_wave_point["longitude"].values)
        if (selected_latitude, selected_longitude) != (latitude, longitude):
            raise ValueError(
                f"No matching wind grid point for wave point in {path.name}"
            )
        for name in WIND_METRICS:
            if name not in wind_at_wave_point:
                metrics[name] = np.full(times.size, np.nan)
                units[name] = "unavailable in source archive"
                continue
            variable = wind_at_wave_point[name]
            metrics[name] = np.asarray(
                variable.values, dtype=float
            ).reshape(-1)
            units[name] = variable.attrs.get("units", "unknown")

    if any(values.size != times.size for values in metrics.values()):
        raise ValueError(f"Time and data lengths differ in {path.name}")
    extracted.unlink()
    wind_extracted.unlink()
    return times, metrics, latitude, longitude, units


def discover_archives(input_dir: Path, start_year: int, end_year: int):
    owners = {}
    pattern = re.compile(
        r"pentland_era5_wind_waves_(\d{4})_(\d{2})(?:-(\d{2}))?\.(nc|zip)$"
    )
    paths = sorted(input_dir.glob("pentland_era5_wind_waves_*"))
    paths.sort(key=lambda path: path.suffix == ".zip")
    for path in paths:
        match = pattern.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        year, first = int(match[1]), int(match[2])
        last = int(match[3] or first)
        if not start_year <= year <= end_year:
            continue
        if not 1 <= first <= last <= 12:
            raise ValueError(f"Invalid month range: {path.name}")
        for month in range(first, last + 1):
            owners[f"{year}-{month:02d}"] = path
    expected = [
        f"{year}-{month:02d}"
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]
    missing = [month for month in expected if month not in owners]
    if missing:
        raise ValueError("Missing ERA5 months: " + ", ".join(missing))
    return owners


def wind_speed_and_direction(
    eastward: np.ndarray,
    northward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(eastward, northward)
    direction = (270 - np.degrees(np.arctan2(northward, eastward))) % 360
    direction = np.where(speed > 0.05, direction, np.nan)
    return speed, direction


def serviceability_filters(
    heights, periods, eastward, northward, wind_max, height_max
):
    valid = (
        np.isfinite(heights) & np.isfinite(periods)
        & (heights >= 0) & (periods >= 0)
    )
    count = int(valid.sum())
    if not count:
        raise ValueError("No valid paired height/period observations")
    speed = np.hypot(eastward, northward)
    wind_missing = valid & ~np.isfinite(speed)
    wind_discard = valid & np.isfinite(speed) & (speed >= wind_max)
    height_discard = valid & (heights >= height_max)
    retained = valid & np.isfinite(speed) & (speed < wind_max)
    retained &= heights < height_max
    masks = {
        "wind_constraint": wind_discard,
        "height_constraint": height_discard,
        "wind_only": wind_discard & ~height_discard,
        "height_only": height_discard & ~wind_discard,
        "both_constraints": wind_discard & height_discard,
        "either_constraint": wind_discard | height_discard,
        "missing_wind": wind_missing,
        "total_excluded": valid & ~retained,
        "retained": retained,
    }
    summary = {
        name: {"hours": int(mask.sum()),
               "percentage_of_valid": float(100 * mask.sum() / count)}
        for name, mask in masks.items()
    }
    return valid, retained, summary


def circular_mean_degrees(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    radians = np.deg2rad(finite)
    mean_angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float(np.rad2deg(mean_angle) % 360)


def make_edges(values: np.ndarray, width: float) -> np.ndarray:
    upper = (np.floor(np.max(values) / width) + 1) * width
    return np.arange(0, upper + width * 0.5, width)


def write_marginal(
    path: Path,
    edges: np.ndarray,
    counts: np.ndarray,
    name: str,
    unit: str,
) -> None:
    probabilities = counts / counts.sum()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            f"{name}_bin_lower_{unit}", f"{name}_bin_upper_{unit}",
            "bin_center", "count", "probability",
        ])
        for index, lower in enumerate(edges[:-1]):
            upper = edges[index + 1]
            writer.writerow([
                lower, upper, (lower + upper) / 2,
                int(counts[index]), probabilities[index],
            ])


def write_matrix(
    path: Path,
    height_edges: np.ndarray,
    period_edges: np.ndarray,
    values: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hs_bin_lower_m", "hs_bin_upper_m",
            "peak_period_bin_lower_s", "peak_period_bin_upper_s", "value",
        ])
        for row, height_lower in enumerate(height_edges[:-1]):
            for column, period_lower in enumerate(period_edges[:-1]):
                value = values[row, column]
                writer.writerow([
                    height_lower, height_edges[row + 1], period_lower,
                    period_edges[column + 1],
                    "" if np.isnan(value) else value,
                ])


def wave_bin_statistics(
    heights: np.ndarray,
    periods: np.ndarray,
    wind_speed_10m: np.ndarray,
    wind_speed_100m: np.ndarray,
    height_edges: np.ndarray,
    period_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = (
        np.isfinite(heights) & np.isfinite(periods)
        & (heights >= 0) & (periods >= 0)
    )
    counts = np.histogram2d(
        heights[valid], periods[valid], bins=(height_edges, period_edges)
    )[0].astype(np.int64)
    statistics = {"count": counts}
    for height, speeds in (("10m", wind_speed_10m), ("100m", wind_speed_100m)):
        selected = valid & np.isfinite(speeds)
        wind_counts = np.histogram2d(
            heights[selected], periods[selected],
            bins=(height_edges, period_edges),
        )[0].astype(np.int64)
        wind_sums = np.histogram2d(
            heights[selected], periods[selected],
            bins=(height_edges, period_edges), weights=speeds[selected],
        )[0]
        statistics[f"wind_sample_count_{height}"] = wind_counts
        statistics[f"mean_wind_speed_{height}_ms"] = np.divide(
            wind_sums, wind_counts, out=np.full(counts.shape, np.nan),
            where=wind_counts > 0,
        )
    return statistics


def write_wave_bin_table(
    path: Path,
    metrics: dict[str, np.ndarray],
    scenario_masks: dict[str, np.ndarray],
    height_edges: np.ndarray,
    period_edges: np.ndarray,
    times: np.ndarray | None = None,
    pool_calendar_months: bool = False,
) -> None:
    if pool_calendar_months and times is None:
        raise ValueError("Calendar-month pooling requires timestamps")
    month_values = None if times is None else times.astype("datetime64[M]")
    if pool_calendar_months:
        month_values = month_values.astype(int) % 12 + 1
    groups = [None] if month_values is None else np.unique(month_values)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            *([] if times is None else (
                ["month"] if pool_calendar_months else ["year", "month"]
            )),
            "scenario", "hs_bin_lower_m", "hs_bin_upper_m", "hs_center_m",
            "peak_period_bin_lower_s", "peak_period_bin_upper_s",
            "tp_center_s",
            "count", "scenario_valid_hours", "all_valid_hours",
            "conditional_probability", "probability_of_all_valid_hours",
            "wind_sample_count_10m", "mean_wind_speed_10m_ms",
            "wind_sample_count_100m", "mean_wind_speed_100m_ms",
        ])
        for group in groups:
            group_mask = (
                np.ones(metrics[HEIGHT_VARIABLE].shape, dtype=bool)
                if group is None else month_values == group
            )
            all_count = int(
                (group_mask & scenario_masks["all_observations"]).sum()
            )
            if all_count == 0:
                continue
            prefix = []
            if pool_calendar_months:
                prefix = [int(group)]
            elif group is not None:
                year, month = str(group).split("-")
                prefix = [int(year), int(month)]
            for scenario_name, scenario_mask in scenario_masks.items():
                selected = group_mask & scenario_mask
                statistics = wave_bin_statistics(
                    metrics[HEIGHT_VARIABLE][selected],
                    metrics[PERIOD_VARIABLE][selected],
                    metrics["wind_speed_10m"][selected],
                    metrics["wind_speed_100m"][selected],
                    height_edges, period_edges,
                )
                counts = statistics["count"]
                scenario_count = int(counts.sum())
                for row, column in np.argwhere(counts > 0):
                    count = int(counts[row, column])
                    wind_values = []
                    for height in ("10m", "100m"):
                        mean = statistics[f"mean_wind_speed_{height}_ms"][
                            row, column
                        ]
                        wind_values.extend([
                            int(statistics[f"wind_sample_count_{height}"][
                                row, column
                            ]),
                            "" if np.isnan(mean) else float(mean),
                        ])
                    writer.writerow([
                        *prefix, scenario_name,
                        height_edges[row], height_edges[row + 1],
                        (height_edges[row] + height_edges[row + 1]) / 2,
                        period_edges[column], period_edges[column + 1],
                        (period_edges[column] + period_edges[column + 1]) / 2,
                        count, scenario_count, all_count,
                        count / scenario_count, count / all_count,
                        *wind_values,
                    ])


def write_monthly_summary(
    path: Path,
    times: np.ndarray,
    metrics: dict[str, np.ndarray],
    valid: np.ndarray,
    workable: np.ndarray,
    start_year: int,
    end_year: int,
    pool_calendar_months: bool = False,
) -> None:
    monthly_metrics = (
        ("shww", "mean_shww_m", False),
        ("pp1d", "mean_peak_period_s", False),
        ("mwp", "mean_wave_period_s", False),
        ("wind_speed_10m", "mean_wind_speed_10m_ms", False),
        ("wind_direction_10m", "mean_wind_direction_10m_deg_from_north", True),
        ("wind_speed_100m", "mean_wind_speed_100m_ms", False),
        (
            "wind_direction_100m",
            "mean_wind_direction_100m_deg_from_north",
            True,
        ),
        ("i10fg", "mean_gust_10m_ms", False),
        ("wsp", "mean_wave_peakedness", False),
        ("mwd", "mean_wave_direction_deg_from_north", True),
        ("mdww", "mean_wind_wave_direction_deg_from_north", True),
        ("mdts", "mean_total_swell_direction_deg_from_north", True),
        ("mp2", "mean_zero_crossing_period_s", False),
        ("mpww", "mean_wind_wave_period_s", False),
        ("mpts", "mean_total_swell_period_s", False),
        ("wdw", "mean_wave_spectral_directional_width_rad", False),
        ("dwww", "mean_wind_wave_spectral_directional_width_rad", False),
        ("dwps", "mean_swell_spectral_directional_width_rad", False),
        ("wss", "mean_wave_spectral_skewness", False),
        ("wsk", "mean_wave_spectral_kurtosis", False),
        ("hmax", "mean_maximum_individual_wave_height_m", False),
        ("tmax", "mean_period_of_maximum_individual_wave_s", False),
        ("wmb", "model_bathymetry_m", False),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            *(["month"] if pool_calendar_months else ["year", "month"]),
            "valid_paired_hours", "workable_hours",
            "workable_occurrence_probability", "mean_shww_m",
            *[column for _, column, _ in monthly_metrics[1:]],
            "workable_occurrence_percent",
        ])
        months = times.astype("datetime64[M]")
        years = [None] if pool_calendar_months else range(
            start_year, end_year + 1
        )
        for year in years:
            for month in range(1, 13):
                calendar_month = (
                    months.astype(int) % 12 + 1 == month
                    if pool_calendar_months
                    else months == np.datetime64(f"{year}-{month:02d}")
                )
                prefix = [month] if pool_calendar_months else [year, month]
                month_mask = calendar_month & valid
                count = int(month_mask.sum())
                if count == 0:
                    writer.writerow(
                        prefix + [""] * (4 + len(monthly_metrics))
                    )
                    continue
                workable_month = month_mask & workable
                values = []
                for name, _, is_direction in monthly_metrics:
                    data = metrics[name]
                    metric_mask = calendar_month & np.isfinite(data)
                    if name in ("shww", "pp1d"):
                        metric_mask &= valid
                    selected = data[metric_mask]
                    if selected.size == 0:
                        values.append("")
                    elif is_direction:
                        values.append(circular_mean_degrees(selected))
                    else:
                        values.append(float(selected.mean()))
                writer.writerow([
                    *prefix, count, int(workable_month.sum()),
                    float(workable_month.sum() / count),
                    *values,
                    float(100 * workable_month.sum() / count),
                ])


def write_metric_summary(
    path: Path,
    metrics: dict[str, np.ndarray],
    units: dict[str, str],
) -> None:
    derived = {
        "wind_speed_10m": (metrics["wind_speed_10m"], "m s**-1"),
        "wind_speed_100m": (metrics["wind_speed_100m"], "m s**-1"),
        "wind_gust_10m": (metrics["i10fg"], units["i10fg"]),
        "wave_spectral_peakedness": (metrics["wsp"], units["wsp"]),
        "zero_crossing_period": (metrics["mp2"], units["mp2"]),
        "mean_wave_period": (metrics["mwp"], units["mwp"]),
        "wind_wave_period": (metrics["mpww"], units["mpww"]),
        "total_swell_period": (metrics["mpts"], units["mpts"]),
        "wave_directional_width": (metrics["wdw"], units["wdw"]),
        "wind_wave_directional_width": (metrics["dwww"], units["dwww"]),
        "swell_directional_width": (metrics["dwps"], units["dwps"]),
        "wave_spectral_skewness": (metrics["wss"], units["wss"]),
        "wave_spectral_kurtosis": (metrics["wsk"], units["wsk"]),
        "maximum_individual_wave_height": (metrics["hmax"], units["hmax"]),
        "period_of_maximum_individual_wave": (metrics["tmax"], units["tmax"]),
        "model_bathymetry": (metrics["wmb"], units["wmb"]),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "metric", "units", "valid_observations", "mean", "median",
            "standard_deviation", "p95", "minimum", "maximum",
        ])
        for name, (values, unit) in derived.items():
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            writer.writerow([
                name, unit, finite.size, float(finite.mean()),
                float(np.median(finite)), float(finite.std()),
                float(np.percentile(finite, 95)),
                float(finite.min()), float(finite.max()),
            ])


def write_direction_frequency(
    path: Path,
    directions: dict[str, np.ndarray],
) -> None:
    sectors = (
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    )
    edges = np.arange(0, 360 + 22.5, 22.5)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "direction_source", "direction_sector", "sector_center_deg",
            "count", "probability",
        ])
        for name, values in directions.items():
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            shifted = (finite + 11.25) % 360
            counts, _ = np.histogram(shifted, bins=edges)
            probabilities = (
                counts / counts.sum()
                if counts.sum()
                else np.zeros(counts.shape, dtype=float)
            )
            for index, sector in enumerate(sectors):
                writer.writerow([
                    name, sector, index * 22.5,
                    int(counts[index]), probabilities[index],
                ])


def run_archive_analysis(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("start-year must be no later than end-year")
    if args.hs_bin <= 0 or args.period_bin <= 0:
        raise ValueError("Bin widths must be positive")
    if args.wind_speed_max <= 0 or args.workability_hs_max <= 0:
        raise ValueError("Wind and height thresholds must be positive")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    owners = discover_archives(
        args.input_dir, args.start_year, args.end_year
    )
    files = sorted(set(owners.values()))

    timestamps = []
    metric_parts = {
        name: []
        for name in (*WAVE_METRICS, *OPTIONAL_WAVE_METRICS, *WIND_METRICS)
    }
    locations = set()
    metric_units = {}
    with tempfile.TemporaryDirectory(prefix="era5_wave_analysis_") as temp:
        temporary_dir = Path(temp)
        for index, path in enumerate(files):
            print(f"Reading {path.name} ...", flush=True)
            result = read_archive(path, temporary_dir, index)
            times, file_metrics, latitude, longitude, file_units = result
            year_mask = np.isin(
                times.astype("datetime64[M]"),
                np.array([
                    month for month, owner in owners.items() if owner == path
                ], dtype="datetime64[M]"),
            )
            timestamps.append(times[year_mask])
            for name, values in file_metrics.items():
                metric_parts[name].append(values[year_mask])
            metric_units.update(file_units)
            locations.add((latitude, longitude))

    if len(locations) != 1:
        raise ValueError(f"Inconsistent wave grid points: {sorted(locations)}")

    times = np.concatenate(timestamps)
    order = np.argsort(times)
    times = times[order]
    metrics = {
        name: np.concatenate(parts)[order]
        for name, parts in metric_parts.items()
    }
    if np.unique(times).size != times.size:
        raise ValueError("Input archives contain overlapping timestamps")

    expected_times = np.arange(
        np.datetime64(f"{args.start_year}-01-01T00", "ns"),
        np.datetime64(f"{args.end_year + 1}-01-01T00", "ns"),
        np.timedelta64(1, "h"),
    )
    if not np.array_equal(times, expected_times):
        raise ValueError(
            "Input timestamps are not a complete hourly record for the "
            f"requested period: found {times.size}, expected "
            f"{expected_times.size}"
        )

    heights = metrics[HEIGHT_VARIABLE]
    peak_period = metrics[PERIOD_VARIABLE]
    valid = (
        np.isfinite(heights) & np.isfinite(peak_period)
        & (heights >= 0) & (peak_period >= 0)
    )
    hs = heights[valid]
    peak_period = peak_period[valid]
    if hs.size == 0:
        raise ValueError("No valid paired shww/pp1d observations found")

    speed_10m, direction_10m = wind_speed_and_direction(
        metrics["u10"], metrics["v10"]
    )
    speed_100m, direction_100m = wind_speed_and_direction(
        metrics["u100"], metrics["v100"]
    )
    metrics["wind_speed_10m"] = speed_10m
    metrics["wind_direction_10m"] = direction_10m
    metrics["wind_speed_100m"] = speed_100m
    metrics["wind_direction_100m"] = direction_100m

    valid, workable_all_times, discard_summary = serviceability_filters(
        heights, metrics[PERIOD_VARIABLE], metrics["u10"], metrics["v10"],
        args.wind_speed_max, args.workability_hs_max,
    )
    retained_label = (
        f"10 m wind < {args.wind_speed_max:g} m/s, "
        f"Hs < {args.workability_hs_max:g} m"
    )

    hs_edges = make_edges(hs, args.hs_bin)
    period_edges = make_edges(peak_period, args.period_bin)
    joint_counts, _, _ = np.histogram2d(
        hs, peak_period, bins=(hs_edges, period_edges)
    )
    joint_counts = joint_counts.astype(np.int64)
    joint_probability = joint_counts / joint_counts.sum()
    hs_counts = joint_counts.sum(axis=1)
    period_counts = joint_counts.sum(axis=0)
    workable = workable_all_times[valid]
    if not np.any(workable):
        raise ValueError("No observations pass both serviceability filters")
    workable_counts, _, _ = np.histogram2d(
        hs[workable], peak_period[workable], bins=(hs_edges, period_edges)
    )
    workable_counts = workable_counts.astype(np.int64)
    workable_joint_probability = workable_counts / workable_counts.sum()
    workable_occurrence = float(workable.sum() / hs.size)
    dominant = np.unravel_index(
        np.argmax(joint_probability), joint_probability.shape
    )

    wind_filter = (
        valid
        & np.isfinite(speed_10m)
        & (speed_10m < args.wind_speed_max)
    )
    scenario_specs = (
        ("all_observations", "All valid paired observations", valid),
        (
            "wind_filter",
            f"10 m wind speed < {args.wind_speed_max:g} m/s",
            wind_filter,
        ),
        (
            "wind_and_height_filter",
            retained_label,
            workable_all_times,
        ),
    )
    scenarios = []
    for scenario_name, scenario_description, scenario_mask in scenario_specs:
        count = int(scenario_mask.sum())
        if count == 0:
            raise ValueError(
                f"Scenario has no valid observations: {scenario_name}"
            )
        counts, _, _ = np.histogram2d(
            heights[scenario_mask],
            peak_period[scenario_mask],
            bins=(hs_edges, period_edges),
        )
        counts = counts.astype(np.int64)
        probability = counts / counts.sum()
        scenarios.append({
            "name": scenario_name,
            "description": scenario_description,
            "mask": scenario_mask,
            "count": count,
            "share_of_all_valid": float(count / valid.sum()),
            "counts": counts,
            "probability": probability,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "discard_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "hours", "percentage_of_valid"])
        for name, result in discard_summary.items():
            writer.writerow([
                name, result["hours"], result["percentage_of_valid"]
            ])
    with (args.output_dir / "raft_event_raster.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "hs_lower_m", "hs_upper_m", "hs_center_m", "tp_lower_s",
            "tp_upper_s", "tp_center_s", "full_count", "full_probability",
            "retained_count", "retained_probability_of_all",
            "retained_conditional_probability",
        ])
        for row, column in np.argwhere(workable_counts > 0):
            writer.writerow([
                hs_edges[row], hs_edges[row + 1],
                (hs_edges[row] + hs_edges[row + 1]) / 2,
                period_edges[column], period_edges[column + 1],
                (period_edges[column] + period_edges[column + 1]) / 2,
                int(joint_counts[row, column]),
                joint_probability[row, column],
                int(workable_counts[row, column]),
                workable_counts[row, column] / hs.size,
                workable_joint_probability[row, column],
            ])
    with (args.output_dir / "scenario_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario", "filters", "valid_hours",
            "share_of_all_valid_hours", "joint_probability_sum",
        ])
        for scenario in scenarios:
            writer.writerow([
                scenario["name"], scenario["description"],
                scenario["count"], scenario["share_of_all_valid"],
                float(scenario["probability"].sum()),
            ])

    with (args.output_dir / "scenario_joint_probabilities.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario", "hs_bin_lower_m", "hs_bin_upper_m",
            "peak_period_bin_lower_s", "peak_period_bin_upper_s",
            "probability",
        ])
        for scenario in scenarios:
            matrix = scenario["probability"]
            for hs_index, hs_lower in enumerate(hs_edges[:-1]):
                for period_index, period_lower in enumerate(
                    period_edges[:-1]
                ):
                    writer.writerow([
                        scenario["name"], hs_lower, hs_edges[hs_index + 1],
                        period_lower, period_edges[period_index + 1],
                        matrix[hs_index, period_index],
                    ])

    with (args.output_dir / "scenario_marginals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "scenario", "metric", "bin_lower", "bin_upper", "bin_center",
            "count", "probability",
        ])
        for scenario in scenarios:
            matrix_counts = scenario["counts"]
            for metric, edges, counts in (
                ("shww", hs_edges, matrix_counts.sum(axis=1)),
                ("pp1d", period_edges, matrix_counts.sum(axis=0)),
            ):
                probabilities = counts / counts.sum()
                for index, lower in enumerate(edges[:-1]):
                    upper = edges[index + 1]
                    writer.writerow([
                        scenario["name"], metric, lower, upper,
                        (lower + upper) / 2, int(counts[index]),
                        probabilities[index],
                    ])

    scenario_figure, scenario_axes = plt.subplots(
        1, len(scenarios), figsize=(18, 6), constrained_layout=True
    )
    for axis, scenario in zip(scenario_axes, scenarios):
        image = axis.pcolormesh(
            hs_edges, period_edges, scenario["probability"].T,
            shading="auto",
        )
        axis.set(
            xlabel="Wind-wave significant height Hs (m)",
            ylabel="Peak wave period (s)",
            title=(
                f"{scenario['description']}\n"
                f"n={scenario['count']:,} "
                f"({scenario['share_of_all_valid']:.1%} of valid hours)"
            ),
        )
        scenario_figure.colorbar(
            image, ax=axis, label="Conditional probability"
        )
    scenario_figure.savefig(
        args.output_dir / "three_scenario_joint_probabilities.png", dpi=200
    )
    plt.close(scenario_figure)

    write_marginal(
        args.output_dir / "hs_marginal.csv",
        hs_edges, hs_counts, "hs", "m",
    )
    write_marginal(
        args.output_dir / "peak_period_marginal.csv",
        period_edges, period_counts, "peak_period", "s",
    )
    write_matrix(
        args.output_dir / "joint_probability.csv",
        hs_edges, period_edges, joint_probability,
    )
    write_matrix(
        args.output_dir / "workable_joint_probability.csv",
        hs_edges, period_edges, workable_joint_probability,
    )
    write_matrix(
        args.output_dir / "period_given_hs.csv",
        hs_edges, period_edges,
        np.divide(
            joint_counts, hs_counts[:, None],
            out=np.full(joint_counts.shape, np.nan),
            where=hs_counts[:, None] != 0,
        ),
    )
    write_matrix(
        args.output_dir / "hs_given_peak_period.csv",
        hs_edges, period_edges,
        np.divide(
            joint_counts, period_counts[None, :],
            out=np.full(joint_counts.shape, np.nan),
            where=period_counts[None, :] != 0,
        ),
    )
    np.savetxt(
        args.output_dir / "joint_counts.csv", joint_counts,
        delimiter=",", fmt="%d",
    )

    hs_full_marginal, _ = np.histogram(hs, bins=hs_edges)
    hs_workable_marginal, _ = np.histogram(hs[workable], bins=hs_edges)
    period_full_marginal, _ = np.histogram(peak_period, bins=period_edges)
    period_workable_marginal, _ = np.histogram(
        peak_period[workable], bins=period_edges
    )
    for filename, edges, full, retained, label, unit in (
        ("hs_workability_marginal.csv", hs_edges, hs_full_marginal,
         hs_workable_marginal, "hs", "m"),
        ("peak_period_workability_marginal.csv", period_edges,
         period_full_marginal, period_workable_marginal,
         "peak_period", "s"),
    ):
        with (args.output_dir / filename).open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                f"{label}_bin_lower_{unit}", f"{label}_bin_upper_{unit}",
                "bin_center", "full_count", "full_probability",
                "workable_count", "workable_probability",
            ])
            full_probability = full / full.sum()
            retained_probability = retained / retained.sum()
            for index, lower in enumerate(edges[:-1]):
                upper = edges[index + 1]
                writer.writerow([
                    lower, upper, (lower + upper) / 2,
                    int(full[index]), full_probability[index],
                    int(retained[index]), retained_probability[index],
                ])

    scenario_masks = {name: mask for name, _, mask in scenario_specs}
    write_wave_bin_table(
        args.output_dir / "wave_bin_statistics.csv", metrics,
        scenario_masks, hs_edges, period_edges,
    )
    write_wave_bin_table(
        args.output_dir / "monthly_wave_bin_statistics.csv", metrics,
        scenario_masks, hs_edges, period_edges, times=times,
    )
    write_monthly_summary(
        args.output_dir / "monthly_summary.csv", times, metrics,
        valid, workable_all_times, args.start_year, args.end_year,
    )
    write_wave_bin_table(
        args.output_dir / "calendar_month_wave_bin_statistics.csv", metrics,
        scenario_masks, hs_edges, period_edges, times=times,
        pool_calendar_months=True,
    )
    write_monthly_summary(
        args.output_dir / "calendar_month_summary.csv", times, metrics,
        valid, workable_all_times, args.start_year, args.end_year,
        pool_calendar_months=True,
    )
    write_metric_summary(
        args.output_dir / "additional_metrics_summary.csv",
        metrics, metric_units,
    )
    write_direction_frequency(
        args.output_dir / "wind_direction_frequency.csv",
        {
            "wind_10m": direction_10m,
            "wind_100m": direction_100m,
            "mean_wave": metrics["mwd"],
            "wind_waves": metrics["mdww"],
            "total_swell": metrics["mdts"],
        },
    )

    additional_metrics = {
        "wind_speed_10m": (speed_10m, "m s**-1", 1.0),
        "wind_speed_100m": (speed_100m, "m s**-1", 1.0),
        "wind_gust_10m": (metrics["i10fg"], metric_units["i10fg"], 1.0),
        "wave_spectral_peakedness": (
            metrics["wsp"], metric_units["wsp"], 0.25
        ),
        "zero_crossing_period": (metrics["mp2"], metric_units["mp2"], 0.25),
        "mean_wave_period": (metrics["mwp"], metric_units["mwp"], 0.25),
        "wind_wave_period": (metrics["mpww"], metric_units["mpww"], 0.25),
        "total_swell_period": (metrics["mpts"], metric_units["mpts"], 0.25),
        "wave_directional_width": (metrics["wdw"], metric_units["wdw"], 0.1),
        "wind_wave_directional_width": (
            metrics["dwww"], metric_units["dwww"], 0.1
        ),
        "swell_directional_width": (
            metrics["dwps"], metric_units["dwps"], 0.1
        ),
        "wave_spectral_skewness": (metrics["wss"], metric_units["wss"], 0.1),
        "wave_spectral_kurtosis": (metrics["wsk"], metric_units["wsk"], 0.25),
        "maximum_individual_wave_height": (
            metrics["hmax"], metric_units["hmax"], 0.25
        ),
        "period_of_maximum_individual_wave": (
            metrics["tmax"], metric_units["tmax"], 0.25
        ),
        "model_bathymetry": (metrics["wmb"], metric_units["wmb"], 5.0),
    }
    for name, (values, units, width) in additional_metrics.items():
        finite = values[np.isfinite(values) & (values >= 0)]
        if finite.size == 0:
            continue
        edges = make_edges(finite, width)
        counts, _ = np.histogram(finite, bins=edges)
        write_marginal(
            args.output_dir / f"{name}_marginal.csv",
            edges, counts, name, units.replace(" ", "_"),
        )

    height_units = metric_units[HEIGHT_VARIABLE] or "unknown"
    period_units = metric_units[PERIOD_VARIABLE] or "unknown"
    metadata = {
        "input_directory": str(args.input_dir.resolve()),
        "site": {"latitude": SITE_LATITUDE, "longitude": SITE_LONGITUDE},
        "download_area_north_west_south_east": DOWNLOAD_AREA,
        "grid_selection": "nearest available wave point to site",
        "site_inside_download_area": (
            DOWNLOAD_AREA[2] <= SITE_LATITUDE <= DOWNLOAD_AREA[0]
            and DOWNLOAD_AREA[1] <= SITE_LONGITUDE <= DOWNLOAD_AREA[3]
        ),
        "discard_summary": discard_summary,
        "discard_denominator": int(valid.sum()),
        "source_packages": [str(path.resolve()) for path in files],
        "monthly_source_selection": {
            month: path.name for month, path in sorted(owners.items())
        },
        "source_files": len(files),
        "period": {"start_year": args.start_year, "end_year": args.end_year},
        "valid_time_start": str(times[0]),
        "valid_time_end": str(times[-1]),
        "expected_hourly_observations": int(expected_times.size),
        "valid_paired_observations": int(hs.size),
        "height_variable": HEIGHT_VARIABLE,
        "height_label": "Significant height of wind waves",
        "height_units": height_units,
        "peak_period_variable": PERIOD_VARIABLE,
        "peak_period_label": "Peak wave period",
        "peak_period_units": period_units,
        "available_additional_wave_variables": [
            name for name in OPTIONAL_WAVE_METRICS
            if np.isfinite(metrics[name]).any()
        ],
        "additional_metrics": {
            "wind_speed_10m": {
                "formula": "sqrt(u10^2 + v10^2)",
                "units": "m s**-1",
            },
            "wind_direction_10m": {
                "formula": "meteorological direction from atan2(v10, u10)",
                "convention": (
                    "degrees clockwise from north; calm values omitted"
                ),
            },
            "wind_speed_100m": {
                "formula": "sqrt(u100^2 + v100^2)",
                "units": "m s**-1",
            },
            "wind_direction_100m": {
                "formula": "meteorological direction from atan2(v100, u100)",
                "convention": (
                    "degrees clockwise from north; calm values omitted"
                ),
            },
            "wind_gust_10m": {
                "source_variable": "i10fg",
                "units": metric_units["i10fg"],
            },
            "wave_spectral_peakedness": {
                "source_variable": "wsp",
                "units": metric_units["wsp"],
            },
            "wind_grid_point": {
                "latitude": next(iter(locations))[0],
                "longitude": next(iter(locations))[1],
                "selection": "exact match to wave grid point",
            },
        },
        "grid_points": [
            {"latitude": latitude, "longitude": longitude}
            for latitude, longitude in sorted(locations)
        ],
        "weighting": (
            "equal valid hourly observations at the downloaded "
            "ERA5 wave grid point"
        ),
        "bin_widths": {
            "height_m": args.hs_bin,
            "peak_period_s": args.period_bin,
        },
        "wave_bin_statistics": {
            "files": [
                "wave_bin_statistics.csv", "monthly_wave_bin_statistics.csv",
                "calendar_month_wave_bin_statistics.csv",
            ],
            "rows": "occupied bins only, per scenario; shared global edges",
            "monthly_grouping": (
                "individual year-month, not pooled across years"
            ),
            "calendar_month_grouping": (
                "same calendar month pooled across all selected years; "
                "counts and finite wind samples pooled before averaging"
            ),
            "calendar_month_summary": "calendar_month_summary.csv",
            "annual_weighting": (
                "all valid hourly observations, not equal monthly weights; "
                "bin wind means weighted by finite wind sample counts"
            ),
            "conditional_probability": (
                "bin count / scenario valid hours in group"
            ),
            "probability_of_all_valid_hours": (
                "bin count / all valid wave hours in group"
            ),
            "wind_means": (
                "arithmetic mean of finite hourly speeds "
                "in the same scenario and bin"
            ),
            "missing_wind": "blank mean with zero sample count; not zero wind",
            "wind_reference_heights_m": [10, 100],
        },
        "height_range_m": [float(hs.min()), float(hs.max())],
        "peak_period_range_s": [
            float(peak_period.min()), float(peak_period.max())
        ],
        "dominant_hs_bin_m": [
            float(hs_edges[dominant[0]]), float(hs_edges[dominant[0] + 1])
        ],
        "dominant_peak_period_bin_s": [
            float(period_edges[dominant[1]]),
            float(period_edges[dominant[1] + 1]),
        ],
        "dominant_joint_probability": float(joint_probability[dominant]),
        "workability": {
            "hs_max_m": args.workability_hs_max,
            "wind_speed_10m_max_ms": args.wind_speed_max,
            "retention_rule": "wind_speed_10m < wind_max and shww < hs_max",
            "workable_observations": int(workable.sum()),
            "workable_occurrence_probability": workable_occurrence,
            "workable_joint_probability_sum": float(
                workable_joint_probability.sum()
            ),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(
        2, 2, figsize=(13, 10), constrained_layout=True
    )
    latitude, longitude = next(iter(locations))
    figure.suptitle(
        f"ERA5 wind-wave states, {args.start_year}-{args.end_year} "
        f"| {latitude:g} N, {abs(longitude):g} W",
        fontsize=14,
    )
    hs_centers = (hs_edges[:-1] + hs_edges[1:]) / 2
    period_centers = (period_edges[:-1] + period_edges[1:]) / 2
    axes[0, 0].bar(
        hs_centers - args.hs_bin * 0.2,
        hs_full_marginal / hs_full_marginal.sum(),
        width=args.hs_bin * 0.4, label="All observations",
    )
    axes[0, 0].bar(
        hs_centers + args.hs_bin * 0.2,
        hs_workable_marginal / hs_workable_marginal.sum(),
        width=args.hs_bin * 0.4,
        label=retained_label,
    )
    axes[0, 0].axvline(
        args.workability_hs_max, color="black", linestyle="--", linewidth=1
    )
    axes[0, 0].set(
        xlabel="Wind-wave significant height Hs (m)",
        ylabel="Probability",
        title="Hs marginal",
    )
    axes[0, 0].legend()
    axes[0, 1].bar(
        period_centers - args.period_bin * 0.2,
        period_full_marginal / period_full_marginal.sum(),
        width=args.period_bin * 0.4, label="All observations",
    )
    axes[0, 1].bar(
        period_centers + args.period_bin * 0.2,
        period_workable_marginal / period_workable_marginal.sum(),
        width=args.period_bin * 0.4,
        label=retained_label,
    )
    axes[0, 1].set(
        xlabel="Peak wave period (s)",
        ylabel="Probability",
        title="Peak period marginal",
    )
    axes[0, 1].legend()
    for axis, matrix, title in (
        (axes[1, 0], joint_probability, "Joint probability: all observations"),
        (axes[1, 1], workable_joint_probability,
         f"Joint probability: {retained_label}"),
    ):
        image = axis.pcolormesh(
            hs_edges, period_edges, matrix.T, shading="auto"
        )
        axis.set(
            xlabel="Wind-wave significant height Hs (m)",
            ylabel="Peak wave period (s)",
            title=title,
        )
        figure.colorbar(image, ax=axis, label="Probability")
    figure.savefig(args.output_dir / "wave_state_distributions.png", dpi=200)
    plt.close(figure)

    metric_figure, metric_axes = plt.subplots(
        2, 2, figsize=(12, 9), constrained_layout=True
    )
    for axis, values, title, unit in (
        (metric_axes[0, 0], speed_10m, "10 m wind speed", "m/s"),
        (metric_axes[0, 1], speed_100m, "100 m wind speed", "m/s"),
        (metric_axes[1, 0], metrics["i10fg"], "10 m wind gust", "m/s"),
        (metric_axes[1, 1], metrics["wsp"], "Wave spectral peakedness", ""),
    ):
        finite = values[np.isfinite(values)]
        axis.hist(finite, bins=40, density=True)
        xlabel = f"{title} ({unit})" if unit else title
        axis.set(xlabel=xlabel, ylabel="Density", title=title)
    metric_figure.savefig(
        args.output_dir / "additional_metric_distributions.png", dpi=200
    )
    plt.close(metric_figure)

    direction_series = {
        "10 m wind": direction_10m,
        "100 m wind": direction_100m,
        "mean waves": metrics["mwd"],
        "wind waves": metrics["mdww"],
        "total swell": metrics["mdts"],
    }
    direction_series = {
        label: values for label, values in direction_series.items()
        if np.isfinite(values).any()
    }
    direction_figure, direction_axes = plt.subplots(
        1, len(direction_series),
        figsize=(5.5 * len(direction_series), 5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for axis, (label, values) in zip(
        np.atleast_1d(direction_axes), direction_series.items()
    ):
        finite = values[np.isfinite(values)]
        shifted = (finite + 11.25) % 360
        counts, _ = np.histogram(
            shifted, bins=np.arange(0, 360 + 22.5, 22.5)
        )
        probabilities = counts / counts.sum()
        angles = np.deg2rad(np.arange(16) * 22.5)
        axis.bar(angles, probabilities, width=np.deg2rad(22.5))
        axis.set_theta_zero_location("N")
        axis.set_theta_direction(-1)
        axis.set_title(f"{label} direction from")
    direction_figure.savefig(
        args.output_dir / "wind_direction_rose.png", dpi=200
    )
    plt.close(direction_figure)

    assert np.isclose(joint_probability.sum(), 1.0)
    assert np.isclose(workable_joint_probability.sum(), 1.0)
    assert np.allclose(
        joint_probability.sum(axis=1), hs_counts / hs_counts.sum()
    )
    assert np.allclose(
        joint_probability.sum(axis=0), period_counts / period_counts.sum()
    )
    print(f"Valid paired hourly observations: {hs.size:,}")
    print(
        f"Workable occurrence ({retained_label}): "
        f"{workable_occurrence:.2%}"
    )
    print(
        "Dominant state: "
        f"Hs {metadata['dominant_hs_bin_m']} m, "
        f"peak period {metadata['dominant_peak_period_bin_s']} s"
    )
    print(f"Results written to {args.output_dir.resolve()}")
    for category, result in discard_summary.items():
        print(f"{category}: {result['hours']:,} hours "
              f"({result['percentage_of_valid']:.2f}% of valid pairs)")


class WaveBinTests(unittest.TestCase):
    def test_wind_means_counts_and_histogram_boundaries(self):
        result = wave_bin_statistics(
            np.array([0.2, 0.4, 1., 2., np.nan, -1.]),
            np.array([0.5, 0.5, 1., 2., 0.5, 0.5]),
            np.array([2., 6., np.nan, 12., 100., 100.]),
            np.array([4., np.nan, 10., 14., 100., 100.]),
            np.array([0., 1., 2.]), np.array([0., 1., 2.]),
        )
        np.testing.assert_array_equal(result["count"], [[2, 0], [0, 2]])
        np.testing.assert_array_equal(
            result["wind_sample_count_10m"], [[2, 0], [0, 1]],
        )
        np.testing.assert_array_equal(
            result["wind_sample_count_100m"], [[1, 0], [0, 2]],
        )
        for height in ("10m", "100m"):
            np.testing.assert_allclose(
                result[f"mean_wind_speed_{height}_ms"],
                [[4., np.nan], [np.nan, 12.]], equal_nan=True,
            )

    def test_empty_input_has_zero_counts_and_missing_means(self):
        empty = np.array([])
        result = wave_bin_statistics(
            empty, empty, empty, empty,
            np.array([0., 1.]), np.array([0., 1.]),
        )
        self.assertEqual(result["count"].item(), 0)
        self.assertTrue(np.isnan(result["mean_wind_speed_10m_ms"]).all())
        self.assertTrue(np.isnan(result["mean_wind_speed_100m_ms"]).all())


class DistributionOutputTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([
            "2006-01-01", "2006-01-02", "2006-02-01", "2007-01-01",
        ], dtype="datetime64[h]")
        self.metrics = {
            name: np.full(4, np.nan)
            for name in (
                *WAVE_METRICS, *OPTIONAL_WAVE_METRICS,
                "wind_speed_10m", "wind_speed_100m",
                "wind_direction_10m", "wind_direction_100m", "i10fg",
            )
        }
        self.metrics.update({
            "shww": np.array([0.2, 0.4, 1.5, 0.5]),
            "pp1d": np.array([0.5, 0.5, 1.5, 0.5]),
            "wind_speed_10m": np.array([2., 20., 4., 8.]),
            "wind_speed_100m": np.array([4., np.nan, np.nan, 12.]),
        })
        self.valid = np.ones(4, dtype=bool)
        self.workable = np.array([True, False, True, True])
        self.scenarios = {
            "all_observations": self.valid,
            "wind_and_height_filter": self.workable,
        }
        self.edges = np.array([0., 1., 2.])

    def test_monthly_bins_preserve_year_and_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monthly.csv"
            write_wave_bin_table(
                path, self.metrics, self.scenarios, self.edges, self.edges,
                times=self.times,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        january, retained = rows[:2]
        self.assertEqual((january["year"], january["month"]), ("2006", "1"))
        self.assertEqual(january["mean_wind_speed_10m_ms"], "11.0")
        self.assertEqual(january["wind_sample_count_100m"], "1")
        self.assertEqual(retained["mean_wind_speed_10m_ms"], "2.0")
        self.assertEqual(retained["probability_of_all_valid_hours"], "0.5")
        self.assertEqual(rows[2]["mean_wind_speed_100m_ms"], "")
        self.assertEqual(rows[4]["year"], "2007")
        self.assertEqual(rows[4]["mean_wind_speed_10m_ms"], "8.0")
        for row in rows:
            self.assertEqual(float(row["conditional_probability"]), 1.)

    def test_calendar_month_pooling_uses_sample_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pooled.csv"
            write_wave_bin_table(
                path, self.metrics, self.scenarios, self.edges, self.edges,
                times=self.times, pool_calendar_months=True,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertNotIn("year", rows[0])
            self.assertEqual(rows[0]["month"], "1")
            self.assertEqual(rows[0]["count"], "3")
            self.assertEqual(rows[0]["mean_wind_speed_10m_ms"], "10.0")
            self.assertEqual(rows[0]["wind_sample_count_100m"], "2")
            self.assertEqual(rows[0]["mean_wind_speed_100m_ms"], "8.0")
            self.assertEqual(rows[1]["mean_wind_speed_10m_ms"], "5.0")
            self.assertAlmostEqual(
                float(rows[1]["probability_of_all_valid_hours"]), 2 / 3,
            )
            write_monthly_summary(
                path, self.times, self.metrics, self.valid, self.workable,
                2006, 2007, pool_calendar_months=True,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 12)
        self.assertNotIn("year", summary[0])
        self.assertEqual(summary[0]["valid_paired_hours"], "3")
        self.assertAlmostEqual(
            float(summary[0]["workable_occurrence_percent"]), 200 / 3,
        )
        self.assertEqual(summary[2]["workable_occurrence_percent"], "")
        self.assertTrue(all(None not in row for row in summary))
        self.assertEqual(sum(
            int(row["valid_paired_hours"])
            for row in summary if row["valid_paired_hours"]
        ), int(self.valid.sum()))

    def test_calendar_month_pooling_includes_leap_day(self):
        self.times = np.array([
            "2007-02-28", "2008-02-28", "2008-02-29", "2008-03-01",
        ], dtype="datetime64[h]")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pooled.csv"
            write_wave_bin_table(
                path, self.metrics, self.scenarios, self.edges, self.edges,
                times=self.times, pool_calendar_months=True,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        february = [
            row for row in rows if row["month"] == "2"
            and row["scenario"] == "all_observations"
        ]
        self.assertEqual(sum(int(row["count"]) for row in february), 3)
        self.assertAlmostEqual(sum(
            float(row["conditional_probability"]) for row in february
        ), 1.)

    def test_full_period_probabilities_and_empty_scenario(self):
        self.scenarios["empty"] = np.zeros(4, dtype=bool)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bins.csv"
            write_wave_bin_table(
                path, self.metrics, self.scenarios, self.edges, self.edges,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertNotIn("year", rows[0])
        self.assertEqual(rows[0]["count"], "3")
        self.assertEqual(rows[0]["mean_wind_speed_10m_ms"], "10.0")
        self.assertEqual(rows[0]["mean_wind_speed_100m_ms"], "8.0")
        for scenario in self.scenarios:
            selected = [row for row in rows if row["scenario"] == scenario]
            if selected:
                self.assertAlmostEqual(sum(
                    float(row["conditional_probability"]) for row in selected
                ), 1.)
        self.assertAlmostEqual(sum(
            float(row["probability_of_all_valid_hours"])
            for row in rows if row["scenario"] == "wind_and_height_filter"
        ), 0.75)

    def test_monthly_workability_percentage_and_empty_month(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.csv"
            write_monthly_summary(
                path, self.times, self.metrics, self.valid, self.workable,
                2006, 2007,
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["workable_occurrence_probability"], "0.5")
        self.assertEqual(rows[0]["workable_occurrence_percent"], "50.0")
        self.assertEqual(rows[1]["workable_occurrence_percent"], "100.0")
        self.assertEqual(rows[2]["workable_occurrence_percent"], "")
        self.assertEqual(rows[12]["workable_occurrence_percent"], "100.0")
        self.assertTrue(all(None not in row for row in rows))


class ServiceabilityTests(unittest.TestCase):
    def test_thresholds_overlap_and_missing_wind(self):
        valid, retained, summary = serviceability_filters(
            np.array([1., 1., 2.5, 3., 1., np.nan]),
            np.ones(6), np.array([0., 12., 0., 20., np.nan, 0.]),
            np.array([0., 16., 0., 0., 0., 0.]), 20., 2.5,
        )
        self.assertEqual(int(valid.sum()), 5)
        self.assertEqual(retained.tolist(),
                         [True, False, False, False, False, False])
        self.assertEqual(summary["wind_constraint"]["hours"], 2)
        self.assertEqual(summary["height_constraint"]["hours"], 2)
        self.assertEqual(summary["both_constraints"]["hours"], 1)
        self.assertEqual(summary["either_constraint"]["hours"], 3)
        self.assertEqual(summary["missing_wind"]["hours"], 1)
        self.assertEqual(summary["retained"]["percentage_of_valid"], 20.)

    def test_signed_wind_is_converted_to_magnitude(self):
        _, retained, _ = serviceability_filters(
            np.ones(2), np.ones(2), np.array([-20., -19.]),
            np.zeros(2), 20., 2.5,
        )
        self.assertEqual(retained.tolist(), [False, True])

    def test_archive_ownership_and_missing_months(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "pentland_era5_wind_waves_2006_01-12.nc"
            bundle.touch()
            monthly = root / "pentland_era5_wind_waves_2006_01.zip"
            monthly.touch()
            owners = discover_archives(root, 2006, 2006)
            self.assertEqual(len(owners), 12)
            self.assertEqual(owners["2006-01"], monthly)
            self.assertEqual(owners["2006-02"], bundle)
            with self.assertRaisesRegex(ValueError, "Missing ERA5 months"):
                discover_archives(root, 2006, 2007)


def run_self_tests() -> None:
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        for test_case in (
            WaveBinTests, DistributionOutputTests, ServiceabilityTests
        )
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
    else:
        run_archive_analysis(args)


if __name__ == "__main__":
    main()
