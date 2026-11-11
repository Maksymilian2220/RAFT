"""Backfill swell height, partition mean periods and wave directions."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import netCDF4
import numpy as np
import pandas as pd

from download_era5 import (
    download_job, discover_existing_months, grouped_supplement_jobs,
    download_grouped_supplement_job,
)
from era5_cache import load_cache
from prepare_era5_cache import (
    merge_existing_cache, read_supplement_archives,
)


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_PREFIX = "pentland_era5_shts"
VARIABLES = ["significant_height_of_total_swell"]
DEFAULT_BASE = BASE_DIR / "era5_processed" / "era5_point_2006_2025.parquet"
DEFAULT_OUTPUT = DEFAULT_BASE.with_name(
    "era5_point_2006_2025_with_shts.parquet"
)
DEFAULT_DOWNLOAD_DIR = BASE_DIR / "era5_shts_backfill"
GROUPED_DOWNLOAD_DIR = BASE_DIR / "era5_bulk_backfill_v1"
GROUPED_OUTPUT = DEFAULT_BASE.with_name(
    "era5_point_2006_2025_with_wave_bulk.parquet"
)
BULK_VARIABLES = {
    "significant_height_of_total_swell": ("shts", "m"),
    "mean_period_of_wind_waves": ("mpww", "s"),
    "mean_period_of_total_swell": ("mpts", "s"),
    "mean_wave_direction": ("mwd", "degrees"),
    "mean_direction_of_wind_waves": ("mdww", "degrees"),
    "mean_direction_of_total_swell": ("mdts", "degrees"),
}
FIRST_MONTH = "2006-01"
LAST_MONTH = "2025-06"


def requested_months():
    return pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")


def missing_jobs(output_dir):
    existing = discover_existing_months(output_dir, ARCHIVE_PREFIX)
    return [
        (month.year, [month.month]) for month in requested_months()
        if (month.year, month.month) not in existing
    ]


def grouped_jobs(output_dir, years_per_package=10,
                 start_year=2006, end_year=2025):
    if not 1 <= years_per_package <= 10:
        raise ValueError("years-per-package must be between 1 and 10")
    return grouped_supplement_jobs(Namespace(
        start_year=start_year, end_year=end_year, start_month=1, end_month=12,
        years_per_package=years_per_package,
    ), output_dir)


def bundled_jobs(output_dir, years_per_package=10, months_per_package=3,
                 start_year=2006, end_year=2025):
    if not 1 <= months_per_package <= 12:
        raise ValueError("months-per-package must be between 1 and 12")
    monthly, skipped = grouped_jobs(
        output_dir, years_per_package, start_year, end_year
    )
    compatible = {}
    for job in monthly:
        key = (tuple(job["years"]), tuple(job["days"]))
        compatible.setdefault(key, []).append(job)
    jobs = []
    for (years, days), group in compatible.items():
        for offset in range(0, len(group), months_per_package):
            batch = group[offset:offset + months_per_package]
            months = [job["month"] for job in batch]
            month_label = "-".join(f"{month:02d}" for month in months)
            name = (
                f"pentland_era5_wave_supplement_v2_{years[0]}-{years[-1]}_"
                f"m{month_label}_d{days[0]:02d}-{days[-1]:02d}.zip"
            )
            target = output_dir / name
            if target.is_file() and is_zipfile(target):
                skipped += 1
                continue
            jobs.append({
                "years": list(years), "months": months,
                "month": months[0], "days": list(days),
                "target_name": name,
            })
    return jobs, skipped


def merge_grouped_backfill(base_cache, archive_dir, output,
                           start_year=2006, end_year=2025):
    if output.resolve() == base_cache.resolve():
        raise ValueError("Output must differ from the base cache")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Output already exists: {output}")
    _, _, latitude, longitude, _ = load_cache(
        base_cache, start_year, end_year
    )
    _, metrics, units, _, _ = read_supplement_archives(
        archive_dir, start_year, end_year, latitude, longitude
    )
    for metric, expected_unit in BULK_VARIABLES.values():
        values = metrics[metric]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"Missing or invalid backfill values for {metric}"
            )
        if expected_unit == "degrees" and (values > 360).any():
            raise ValueError(f"Invalid direction values for {metric}")
        accepted_units = (
            {"degrees", "Degree true"} if expected_unit == "degrees"
            else {expected_unit}
        )
        if units.get(metric) not in accepted_units:
            raise ValueError(
                f"Unexpected units for {metric}: {units.get(metric)}"
            )
    merge_existing_cache(
        base_cache, archive_dir, start_year, end_year, output
    )


def merge_backfill(base_cache, archive_dir, output):
    if output.resolve() == base_cache.resolve():
        raise ValueError("Output must differ from the base cache")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output}"
        )
    times, metrics, latitude, longitude, _ = load_cache(base_cache, 2006, 2025)
    swell = metrics["shts"].copy()
    filled_hours = 0
    sources = []
    for month in requested_months():
        path = archive_dir / (
            f"{ARCHIVE_PREFIX}_{month.year}_{month.month:02d}.zip"
        )
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist()
                       if "wave_stepType-instant.nc" in name]
            if len(members) != 1:
                raise ValueError(f"Expected one wave stream in {path.name}")
            payload = archive.read(members[0])
            with netCDF4.Dataset("memory", memory=payload) as dataset:
                latitudes = np.asarray(dataset.variables["latitude"][:])
                longitudes = np.asarray(dataset.variables["longitude"][:])
                latitude_indices = np.flatnonzero(latitudes == latitude)
                longitude_indices = np.flatnonzero(longitudes == longitude)
                if latitude_indices.size != 1 or longitude_indices.size != 1:
                    raise ValueError(
                        f"No exact base-cache grid point in {path.name}"
                    )
                time_variable = dataset.variables["valid_time"]
                dates = netCDF4.num2date(
                    time_variable[:], time_variable.units,
                    calendar=getattr(time_variable, "calendar", "standard"),
                    only_use_cftime_datetimes=False,
                )
                archive_times = pd.to_datetime(list(dates)).to_numpy(
                    dtype="datetime64[ns]"
                )
                variable = dataset.variables["shts"]
                selection = []
                for dimension in variable.dimensions:
                    if dimension == "valid_time":
                        selection.append(slice(None))
                    elif dimension == "latitude":
                        selection.append(int(latitude_indices[0]))
                    elif dimension == "longitude":
                        selection.append(int(longitude_indices[0]))
                    elif len(dataset.dimensions[dimension]) == 1:
                        selection.append(0)
                    else:
                        raise ValueError(
                            f"Unsupported dimension {dimension} in {path.name}"
                        )
                values = np.ma.asarray(
                    variable[tuple(selection)], dtype=float
                ).filled(np.nan)
                values = np.asarray(values).reshape(-1)
                if getattr(variable, "units", None) != "m":
                    raise ValueError(f"Expected shts units m in {path.name}")
        selected = times.astype("datetime64[M]") == np.datetime64(
            str(month), "M"
        )
        if not np.array_equal(archive_times, times[selected]):
            raise ValueError(
                f"Incomplete or misaligned timestamps in {path.name}"
            )
        if (values.shape != times[selected].shape
                or not np.isfinite(values).all() or (values < 0).any()):
            raise ValueError(f"Invalid or missing shts values in {path.name}")
        previous = swell[selected]
        overlap = np.isfinite(previous)
        if not np.allclose(
            previous[overlap], values[overlap], rtol=0, atol=0.001
        ):
            raise ValueError(
                f"Conflicting existing shts values in {path.name}"
            )
        filled_hours += int((~overlap).sum())
        swell[selected] = np.where(overlap, previous, values)
        sources.append(path.name)
    if not np.isfinite(swell).all() or (swell < 0).any():
        raise ValueError(
            "Merged shts is not complete for 2006-2025; "
            "check July-December 2025"
        )
    frame = pd.read_parquet(base_cache)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"])
    frame = frame.sort_values("valid_time").reset_index(drop=True)
    if not np.array_equal(
        frame["valid_time"].to_numpy(dtype="datetime64[ns]"), times
    ):
        raise ValueError(
            "Base cache must contain exactly the 2006-2025 period"
        )
    frame["shts"] = swell
    metadata = json.loads(
        base_cache.with_suffix(".json").read_text(encoding="utf-8")
    )
    metadata.setdefault("units", {})["shts"] = "m"
    metadata["variables"] = list(dict.fromkeys([
        *metadata.get("variables", []), "shts"
    ]))
    metadata["shts_backfill"] = {
        "base_cache": str(base_cache.resolve()),
        "archive_directory": str(archive_dir.resolve()),
        "source_files": sources,
        "filled_hours": filled_hours,
        "first_month": FIRST_MONTH, "last_month": LAST_MONTH,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Filled {filled_hours:,} shts hours; wrote {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir", type=Path
    )
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage", choices=("2025", "history", "all"), default="2025",
        help="2025 first (default), 2006-2024 history, or all years.",
    )
    parser.add_argument(
        "--include-spectra", action="store_true",
        help="Also retrieve full ERA5 spectra as monthly GRIB files.",
    )
    parser.add_argument("--spectra-dir", type=Path)
    parser.add_argument(
        "--years-per-package", type=int, default=10,
        help="Years per calendar-month request (1-10; default: 10).",
    )
    parser.add_argument(
        "--months-per-package", type=int, default=3,
        help="Compatible months per request (1-12; default: 3).",
    )
    parser.add_argument(
        "--monthly-shts-only", action="store_true",
        help="Use the original monthly January 2006-June 2025 shts mode.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true",
        help="List jobs without CDS requests or writes.",
    )
    mode.add_argument(
        "--merge-only", action="store_true",
        help="Merge completed downloads without CDS requests.",
    )
    args = parser.parse_args(argv)
    start_year, end_year = {
        "2025": (2025, 2025), "history": (2006, 2024), "all": (2006, 2025),
    }[args.stage]
    if not 1 <= args.years_per_package <= 10:
        parser.error("--years-per-package must be between 1 and 10")
    if not 1 <= args.months_per_package <= 12:
        parser.error("--months-per-package must be between 1 and 12")
    args.download_dir = args.download_dir or (
        DEFAULT_DOWNLOAD_DIR if args.monthly_shts_only
        else GROUPED_DOWNLOAD_DIR
    )
    args.output = args.output or (
        DEFAULT_OUTPUT if args.monthly_shts_only else (
            GROUPED_OUTPUT if args.stage == "all" else
            GROUPED_OUTPUT.with_name(
                f"era5_point_{start_year}_{end_year}_with_wave_bulk.parquet"
            )
        )
    )
    if args.monthly_shts_only and args.include_spectra:
        parser.error("--include-spectra requires the grouped workflow")
    if not args.monthly_shts_only:
        jobs, skipped = bundled_jobs(
            args.download_dir, args.years_per_package, args.months_per_package,
            start_year, end_year,
        )
        print(f"{len(jobs)} grouped requests missing; {skipped} complete.")
        print("Variables: " + ", ".join(BULK_VARIABLES))
        if args.dry_run:
            for job in jobs:
                print(f"Would download {job['target_name']}")
            print(f"Merge target: {args.output}")
            if args.include_spectra:
                from download_era5_spectra import retrieve_spectra
                retrieve_spectra(
                    start_year, end_year, args.spectra_dir, dry_run=True
                )
            return
        if args.output.resolve() == args.base_cache.resolve():
            parser.error("--output must differ from --base-cache")
        if args.output.exists() or args.output.with_suffix(".json").exists():
            parser.error("Output already exists; choose a new --output")
        load_cache(args.base_cache, start_year, end_year)
        if not args.merge_only:
            args.download_dir.mkdir(parents=True, exist_ok=True)
            for job in jobs:
                download_grouped_supplement_job(
                    job, args.download_dir, list(BULK_VARIABLES)
                )
            if args.include_spectra:
                from download_era5_spectra import retrieve_spectra
                retrieve_spectra(start_year, end_year, args.spectra_dir)
        merge_grouped_backfill(
            args.base_cache, args.download_dir, args.output,
            start_year, end_year,
        )
        return
    jobs = missing_jobs(args.download_dir)
    print(
        f"{len(jobs)} of {len(requested_months())} "
        "monthly shts packages missing.", flush=True,
    )
    if args.dry_run:
        for year, months in jobs:
            print(f"Would download shts {year}-{months[0]:02d}")
        print(f"Merge target: {args.output}")
        return
    if not args.merge_only:
        if args.output.resolve() == args.base_cache.resolve():
            parser.error("--output must differ from --base-cache")
        if args.output.exists() or args.output.with_suffix(".json").exists():
            parser.error("Output already exists; choose a new --output")
        load_cache(args.base_cache, 2006, 2025)
        args.download_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            download_job(job, args.download_dir, VARIABLES, ARCHIVE_PREFIX)
    merge_backfill(args.base_cache, args.download_dir, args.output)


if __name__ == "__main__":
    main()
