"""Prepare monthly ERA5 archives as a pandas-friendly Parquet cache."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd
import xarray as xr

from era5_cache import (
    DEFAULT_CACHE,
    METRIC_COLUMNS,
    OPTIONAL_CACHE_COLUMNS,
    WAVE_SUPPLEMENT_VARIABLES,
)
from wave_state_distribution import (
    DEFAULT_INPUT,
    SITE_LATITUDE,
    SITE_LONGITUDE,
    discover_archives,
    read_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--supplement-dir", type=Path,
        help="Overlay wave-only supplement archives on the prepared data.",
    )
    parser.add_argument(
        "--base-cache", type=Path,
        help="Merge supplements into an existing cache instead of archives.",
    )
    return parser.parse_args()


def validate_hourly_record(
    times: np.ndarray, start_year: int, end_year: int
) -> None:
    expected = np.arange(
        np.datetime64(f"{start_year}-01-01T00", "ns"),
        np.datetime64(f"{end_year + 1}-01-01T00", "ns"),
        np.timedelta64(1, "h"),
    )
    if not np.array_equal(times, expected):
        raise ValueError(
            "Input timestamps are not a complete hourly record for the "
            f"requested period: found {times.size}, expected {expected.size}"
        )


def discover_supplement_archives(
    input_dir: Path, start_year: int, end_year: int
) -> list[Path]:
    monthly_pattern = re.compile(
        r"pentland_era5_wave_supplement(?:_v2)?_(\d{4})_(\d{2})"
        r"(?:-(\d{2}))?\.zip$"
    )
    grouped_pattern = re.compile(
        r"pentland_era5_wave_supplement(?:_v2)?_(\d{4})-(\d{4})_"
        r"m(\d{2}(?:-\d{2})*)_d(\d{2})-(\d{2})\.zip$"
    )
    files = []
    for path in sorted(input_dir.glob("pentland_era5_wave_supplement_*.zip")):
        if not path.is_file() or not is_zipfile(path):
            continue
        monthly_match = monthly_pattern.fullmatch(path.name)
        grouped_match = grouped_pattern.fullmatch(path.name)
        if monthly_match:
            first_year = int(monthly_match[1])
            last_year = first_year
            first_month = int(monthly_match[2])
            last_month = int(monthly_match[3] or first_month)
            if not 1 <= first_month <= last_month <= 12:
                raise ValueError(f"Invalid month range: {path.name}")
        elif grouped_match:
            first_year = int(grouped_match[1])
            last_year = int(grouped_match[2])
            months = [int(value) for value in grouped_match[3].split("-")]
            first_day, last_day = int(grouped_match[4]), int(grouped_match[5])
            if (not all(1 <= month <= 12 for month in months)
                    or not 1 <= first_day <= last_day <= 31):
                raise ValueError(f"Invalid grouped package range: {path.name}")
        else:
            continue
        if last_year < start_year or first_year > end_year:
            continue
        files.append(path)
    if not files:
        raise ValueError(f"No wave supplement archives found in {input_dir}")
    return files


def read_supplement_archives(
    input_dir: Path,
    start_year: int,
    end_year: int,
    latitude: float,
    longitude: float,
):
    files = discover_supplement_archives(input_dir, start_year, end_year)
    time_parts = []
    metric_parts = {name: [] for name in WAVE_SUPPLEMENT_VARIABLES.values()}
    units = {}
    locations = set()
    with tempfile.TemporaryDirectory(prefix="era5_wave_supplement_") as temp:
        temporary_dir = Path(temp)
        for index, path in enumerate(files):
            with ZipFile(path) as archive:
                matches = [
                    name for name in archive.namelist()
                    if "wave_stepType-instant.nc" in name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one wave stream in {path.name}; "
                        f"found {len(matches)}"
                    )
                extracted = temporary_dir / f"wave_supplement_{index:04d}.nc"
                extracted.write_bytes(archive.read(matches[0]))
            try:
                with xr.open_dataset(extracted, engine="netcdf4") as dataset:
                    selected = dataset.sel(
                        latitude=latitude, longitude=longitude,
                        method="nearest",
                    )
                    selected_location = (
                        float(selected["latitude"].values),
                        float(selected["longitude"].values),
                    )
                    if selected_location != (latitude, longitude):
                        raise ValueError(
                            f"Supplement grid point {selected_location} does "
                            "not match base cache point "
                            f"{(latitude, longitude)}"
                        )
                    locations.add(selected_location)
                    times = np.asarray(
                        selected["valid_time"].values
                    ).astype("datetime64[ns]")
                    file_metrics = {}
                    for _, metric in WAVE_SUPPLEMENT_VARIABLES.items():
                        if metric not in selected.data_vars:
                            file_metrics[metric] = np.full(times.size, np.nan)
                            continue
                        data_array = selected[metric]
                        values = np.asarray(
                            data_array.values, dtype=float
                        ).reshape(-1)
                        if values.size == 1:
                            values = np.full(times.size, values[0])
                        file_metrics[metric] = values
                        units[metric] = data_array.attrs.get(
                            "units", "unknown"
                        )
                    if any(values.size != times.size
                           for values in file_metrics.values()):
                        raise ValueError(
                            f"Time and wave data lengths differ in {path.name}"
                        )
            finally:
                extracted.unlink(missing_ok=True)

            requested_start = np.datetime64(
                f"{start_year}-01-01T00", "ns"
            )
            requested_end = np.datetime64(
                f"{end_year + 1}-01-01T00", "ns"
            )
            mask = (times >= requested_start) & (times < requested_end)
            time_parts.append(times[mask])
            for name in metric_parts:
                metric_parts[name].append(file_metrics[name][mask])

    if len(locations) != 1:
        raise ValueError(f"Inconsistent supplement grid points: {locations}")
    times = np.concatenate(time_parts)
    order = np.argsort(times)
    times = times[order]
    metrics = {
        name: np.concatenate(parts)[order]
        for name, parts in metric_parts.items()
    }
    unique_times, first_indices, inverse, counts = np.unique(
        times, return_index=True, return_inverse=True, return_counts=True
    )
    duplicate_groups = np.flatnonzero(counts > 1)
    if duplicate_groups.size:
        for name, values in metrics.items():
            deduplicated = values[first_indices].copy()
            for group in duplicate_groups:
                group_values = values[inverse == group]
                finite = group_values[np.isfinite(group_values)]
                if finite.size > 1 and not np.allclose(
                    finite, finite[0], rtol=1e-6, atol=1e-8
                ):
                    raise ValueError(
                        f"Conflicting overlapping supplement values for {name}"
                    )
                if finite.size:
                    deduplicated[group] = finite[0]
            metrics[name] = deduplicated
    times = unique_times
    validate_hourly_record(times, start_year, end_year)
    return times, metrics, units, next(iter(locations)), [
        path.name for path in files
    ]


def overlay_supplement_metrics(
    base_times: np.ndarray,
    base_metrics: dict[str, np.ndarray],
    supplement_times: np.ndarray,
    supplement_metrics: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    positions = np.searchsorted(base_times, supplement_times)
    if (np.any(positions >= base_times.size)
            or not np.array_equal(base_times[positions], supplement_times)):
        raise ValueError(
            "Wave supplement timestamps do not match base cache timestamps"
        )
    merged = {name: values.copy() for name, values in base_metrics.items()}
    filled = {}
    for name, values in supplement_metrics.items():
        if name not in merged:
            continue
        existing = merged[name][positions]
        use_supplement = ~np.isfinite(existing) & np.isfinite(values)
        existing[use_supplement] = values[use_supplement]
        merged[name][positions] = existing
        filled[name] = int(use_supplement.sum())
    return merged, filled


def write_cache(
    frame: pd.DataFrame, output: Path, metadata: dict
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    frame.to_parquet(temporary_output, index=False)
    temporary_output.replace(output)
    metadata_path = output.with_suffix(".json")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)


def merge_existing_cache(
    base_cache: Path,
    supplement_dir: Path,
    start_year: int,
    end_year: int,
    output: Path,
) -> None:
    if output.resolve() == base_cache.resolve():
        raise ValueError("Output cache must differ from the base cache")
    metadata_path = base_cache.with_suffix(".json")
    if not base_cache.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Base cache or metadata not found: {base_cache}"
        )
    frame = pd.read_parquet(base_cache)
    required = {
        "valid_time", *(set(METRIC_COLUMNS) - set(OPTIONAL_CACHE_COLUMNS))
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Base cache is missing columns: " + ", ".join(sorted(missing))
        )
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="raise")
    frame = frame.loc[
        frame["valid_time"].dt.year.between(start_year, end_year)
    ].sort_values("valid_time").reset_index(drop=True)
    times = frame["valid_time"].to_numpy(dtype="datetime64[ns]")
    if np.unique(times).size != times.size:
        raise ValueError("Base cache contains duplicate timestamps")
    validate_hourly_record(times, start_year, end_year)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    location = metadata.get("location", {})
    try:
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Base cache has invalid location metadata") from error

    supplement_times, supplement_metrics, supplement_units, _, files = (
        read_supplement_archives(
            supplement_dir, start_year, end_year, latitude, longitude
        )
    )
    base_metrics = {
        name: (
            frame[name].to_numpy(dtype=float)
            if name in frame else np.full(times.size, np.nan)
        )
        for name in METRIC_COLUMNS
    }
    merged, filled = overlay_supplement_metrics(
        times, base_metrics, supplement_times, supplement_metrics
    )
    for name in METRIC_COLUMNS:
        frame[name] = merged[name]
    metadata["start_time"] = str(times[0])
    metadata["end_time"] = str(times[-1])
    metadata["rows"] = int(times.size)
    metadata["variables"] = list(METRIC_COLUMNS)
    metadata["supplement_source_files"] = files
    metadata["supplement_variables"] = list(WAVE_SUPPLEMENT_VARIABLES)
    metadata["supplement_values_filled"] = filled
    metadata.setdefault("units", {}).update(supplement_units)
    write_cache(frame, output, metadata)
    print(f"Wrote supplemented cache to {output}", flush=True)


def prepare_cache(
    input_dir: Path, start_year: int, end_year: int, output: Path,
    supplement_dir: Path | None = None,
) -> None:
    if start_year > end_year:
        raise ValueError("start-year must be no later than end-year")
    owners = discover_archives(input_dir, start_year, end_year)
    files = sorted(set(owners.values()))
    timestamps = []
    metric_parts = {name: [] for name in METRIC_COLUMNS}
    locations = set()
    units = {}
    with tempfile.TemporaryDirectory(prefix="era5_cache_prepare_") as temp:
        temporary_dir = Path(temp)
        for index, path in enumerate(files):
            print(f"Reading {path.name} ...", flush=True)
            times, metrics, latitude, longitude, file_units = read_archive(
                path, temporary_dir, index, include_swh=True
            )
            months_for_file = [
                month for month, owner in owners.items() if owner == path
            ]
            month_values = np.array(months_for_file, dtype="datetime64[M]")
            mask = np.isin(times.astype("datetime64[M]"), month_values)
            timestamps.append(times[mask])
            for name in METRIC_COLUMNS:
                metric_parts[name].append(metrics[name][mask])
            units.update(file_units)
            locations.add((latitude, longitude))

    if len(locations) != 1:
        raise ValueError(f"Inconsistent wave grid points: {sorted(locations)}")
    times = np.concatenate(timestamps)
    order = np.argsort(times)
    times = times[order]
    if np.unique(times).size != times.size:
        raise ValueError("Input archives contain overlapping timestamps")
    expected = np.arange(
        np.datetime64(f"{start_year}-01-01T00", "ns"),
        np.datetime64(f"{end_year + 1}-01-01T00", "ns"),
        np.timedelta64(1, "h"),
    )
    if not np.array_equal(times, expected):
        raise ValueError(
            "Input timestamps are not a complete hourly record for the "
            f"requested period: found {times.size}, expected {expected.size}"
        )

    data = {"valid_time": times}
    for name, parts in metric_parts.items():
        data[name] = np.concatenate(parts)[order]
    frame = pd.DataFrame(data)

    latitude, longitude = next(iter(locations))
    supplement_files = []
    if supplement_dir is not None:
        supplement_times, supplement_metrics, supplement_units, \
            supplement_location, supplement_files = read_supplement_archives(
                supplement_dir, start_year, end_year, latitude, longitude
            )
        if supplement_location != (latitude, longitude):
            raise ValueError(
                "Supplement grid point does not match primary archives"
            )
        base_metrics = {
            name: frame[name].to_numpy(dtype=float)
            for name in METRIC_COLUMNS
        }
        merged_metrics, filled = overlay_supplement_metrics(
            times, base_metrics, supplement_times, supplement_metrics
        )
        for name, values in merged_metrics.items():
            frame[name] = values
        units.update(supplement_units)
    else:
        filled = {}

    metadata = {
        "format": "parquet",
        "valid_time_column": "valid_time",
        "start_time": str(times[0]),
        "end_time": str(times[-1]),
        "rows": int(times.size),
        "location": {"latitude": latitude, "longitude": longitude},
        "nominal_site": {
            "latitude": SITE_LATITUDE,
            "longitude": SITE_LONGITUDE,
        },
        "variables": list(METRIC_COLUMNS),
        "units": units,
        "source_files": [path.name for path in files],
    }
    if supplement_dir is not None:
        metadata["supplement_source_files"] = supplement_files
        metadata["supplement_variables"] = list(WAVE_SUPPLEMENT_VARIABLES)
        metadata["supplement_values_filled"] = filled
    write_cache(frame, output, metadata)
    print(f"Wrote {len(frame):,} hourly rows to {output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.base_cache is not None:
        if args.supplement_dir is None:
            raise SystemExit("--base-cache requires --supplement-dir")
        merge_existing_cache(
            args.base_cache, args.supplement_dir, args.start_year,
            args.end_year, args.output,
        )
        return
    prepare_cache(
        args.input_dir, args.start_year, args.end_year, args.output,
        args.supplement_dir,
    )


if __name__ == "__main__":
    main()
