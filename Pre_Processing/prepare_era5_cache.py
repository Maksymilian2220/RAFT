"""Prepare monthly ERA5 archives as a pandas-friendly Parquet cache."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from era5_cache import DEFAULT_CACHE, METRIC_COLUMNS
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
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--output", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def prepare_cache(
    input_dir: Path, start_year: int, end_year: int, output: Path
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
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    frame.to_parquet(temporary_output, index=False)
    temporary_output.replace(output)

    latitude, longitude = next(iter(locations))
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
    metadata_path = output.with_suffix(".json")
    temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    print(f"Wrote {len(frame):,} hourly rows to {output}", flush=True)


def main() -> None:
    args = parse_args()
    prepare_cache(args.input_dir, args.start_year, args.end_year, args.output)


if __name__ == "__main__":
    main()
