"""Prepare paired ERA5 wave data for Hs/Tp and height-comparison plots."""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
import unittest
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import xarray as xr

from era5_cache import load_cache


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "era5_monthly"
DEFAULT_OUTPUT = BASE_DIR / "wave_swell_wind_comparison_results"
DEFAULT_CACHE = BASE_DIR / "era5_processed" / "era5_point_2006_2023.parquet"
START_YEAR = 2006
END_YEAR = 2023
SITE_LATITUDE = 58.6296
SITE_LONGITUDE = -3.8403666667
WAVE_VARIABLES = ("swh", "shww", "shts", "pp1d")
REQUIRED_WAVE_VARIABLES = ("swh", "shww", "pp1d")
OUTPUT_CSV = "wave_state_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--cache-file", type=Path, default=DEFAULT_CACHE,
        help="Read prepared Parquet input instead of monthly archives.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_archive(path: Path, temporary_dir: Path, index: int):
    extracted = temporary_dir / f"wave_stream_{index:04d}.nc"
    try:
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
            extracted.write_bytes(archive.read(matches[0]))
    except BadZipFile as error:
        raise ValueError(f"Not a valid ERA5 ZIP archive: {path}") from error

    try:
        with xr.open_dataset(extracted, engine="netcdf4") as dataset:
            missing = set(REQUIRED_WAVE_VARIABLES) - set(dataset.data_vars)
            if missing:
                raise KeyError(
                    f"{path.name} missing wave variables {sorted(missing)}"
                )
            selected = dataset.sel(
                latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE,
                method="nearest",
            )
            times = np.asarray(selected["valid_time"].values).astype(
                "datetime64[ns]"
            )
            metrics = {
                name: (
                    np.asarray(selected[name].values, dtype=float).reshape(-1)
                    if name in selected.data_vars
                    else np.full(times.size, np.nan)
                )
                for name in WAVE_VARIABLES
            }
    finally:
        extracted.unlink(missing_ok=True)

    if any(values.size != times.size for values in metrics.values()):
        raise ValueError(f"Time and wave data lengths differ in {path.name}")
    return times, metrics


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


def read_archives(input_dir: Path, start_year: int, end_year: int):
    owners = discover_archives(input_dir, start_year, end_year)
    files = sorted(set(owners.values()))
    time_parts = []
    metric_parts = {name: [] for name in WAVE_VARIABLES}
    with tempfile.TemporaryDirectory(prefix="era5_wave_plot_data_") as temp:
        temporary_dir = Path(temp)
        for index, path in enumerate(files):
            print(f"Reading {path.name} ...", flush=True)
            times, file_metrics = read_archive(path, temporary_dir, index)
            owned_months = np.array([
                month for month, owner in owners.items() if owner == path
            ], dtype="datetime64[M]")
            selected = np.isin(times.astype("datetime64[M]"), owned_months)
            time_parts.append(times[selected])
            for name in WAVE_VARIABLES:
                metric_parts[name].append(file_metrics[name][selected])

    times = np.concatenate(time_parts)
    order = np.argsort(times)
    times = times[order]
    metrics = {
        name: np.concatenate(parts)[order]
        for name, parts in metric_parts.items()
    }
    if np.unique(times).size != times.size:
        raise ValueError("Input archives contain overlapping timestamps")
    return times, metrics


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


def write_wave_state_data(
    path: Path, times: np.ndarray, metrics: dict[str, np.ndarray]
) -> int:
    valid = np.ones(times.size, dtype=bool)
    for name in REQUIRED_WAVE_VARIABLES:
        valid &= np.isfinite(metrics[name]) & (metrics[name] >= 0)
    if not valid.any():
        raise ValueError("No valid paired swh/shww/pp1d observations found")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "valid_time", "swh_m", "shww_m", "shts_m", "peak_period_s",
            "swh_minus_shww_m",
        ])
        swell_height = metrics.get("shts", np.full(times.size, np.nan))
        for index in np.flatnonzero(valid):
            combined = metrics["swh"][index]
            wind_wave = metrics["shww"][index]
            total_swell = swell_height[index]
            writer.writerow([
                str(times[index]), combined, wind_wave,
                total_swell if np.isfinite(total_swell) else "",
                metrics["pp1d"][index], combined - wind_wave,
            ])
    return int(valid.sum())


def run_analysis(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("start-year must be no later than end-year")
    if args.cache_file.is_file():
        times, metrics, _, _, _ = load_cache(
            args.cache_file, args.start_year, args.end_year
        )
        metrics = {name: metrics[name] for name in WAVE_VARIABLES}
    else:
        if args.cache_file != DEFAULT_CACHE:
            raise FileNotFoundError(
                "Parquet cache not found: "
                f"{args.cache_file}"
            )
        if not args.input_dir.is_dir():
            raise FileNotFoundError(
                "Input directory not found: "
                f"{args.input_dir}"
            )
        times, metrics = read_archives(
            args.input_dir, args.start_year, args.end_year
        )

    validate_hourly_record(times, args.start_year, args.end_year)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = write_wave_state_data(
        args.output_dir / OUTPUT_CSV, times, metrics
    )
    print(f"Saved {count:,} paired wave observations to {args.output_dir}")


class WaveStateOutputTests(unittest.TestCase):
    def test_writes_only_valid_paired_wave_states(self):
        times = np.array(
            ["2006-01-01T00", "2006-01-01T01"], dtype="datetime64[ns]"
        )
        metrics = {
            "swh": np.array([1.2, 1.4]),
            "shww": np.array([0.8, 0.9]),
            "shts": np.array([0.5, 0.6]),
            "pp1d": np.array([8.0, np.nan]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / OUTPUT_CSV
            count = write_wave_state_data(path, times, metrics)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["swh_minus_shww_m"]), 0.4)
        self.assertEqual(float(rows[0]["shts_m"]), 0.5)

    def test_swell_height_may_be_missing_from_legacy_data(self):
        times = np.array(["2006-01-01T00"], dtype="datetime64[ns]")
        metrics = {
            "swh": np.array([1.2]),
            "shww": np.array([0.8]),
            "pp1d": np.array([8.0]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / OUTPUT_CSV
            write_wave_state_data(path, times, metrics)
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["shts_m"], "")

    def test_hourly_record_validation(self):
        expected = np.arange(
            np.datetime64("2006-01-01T00", "ns"),
            np.datetime64("2007-01-01T00", "ns"),
            np.timedelta64(1, "h"),
        )
        validate_hourly_record(expected, 2006, 2006)
        with self.assertRaisesRegex(ValueError, "complete hourly record"):
            validate_hourly_record(expected[:-1], 2006, 2006)


def run_self_tests() -> None:
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(WaveStateOutputTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_tests()
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
