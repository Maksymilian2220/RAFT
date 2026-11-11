"""Continue the standard monthly ERA5 download from July 2025 onward."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_era5 import (
    OUTPUT_DIR,
    VARIABLES,
    download_job,
    discover_existing_months,
)


START_YEAR = 2025
START_MONTH = 7
END_YEAR = 2025
END_MONTH = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--start-month", type=int, default=START_MONTH)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--end-month", type=int, default=END_MONTH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List missing monthly packages without submitting CDS requests.",
    )
    args = parser.parse_args()
    if not 1 <= args.start_month <= 12:
        parser.error("--start-month must be between 1 and 12")
    if not 1 <= args.end_month <= 12:
        parser.error("--end-month must be between 1 and 12")
    if (args.start_year, args.start_month) > (
        args.end_year, args.end_month
    ):
        parser.error("Start date must be no later than end date")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = discover_existing_months(args.output_dir)
    jobs = [
        (year, [month])
        for year in range(args.start_year, args.end_year + 1)
        for month in range(
            args.start_month if year == args.start_year else 1,
            args.end_month + 1 if year == args.end_year else 13,
        )
        if (year, month) not in existing
    ]
    total_months = sum(
        1
        for year in range(args.start_year, args.end_year + 1)
        for month in range(
            args.start_month if year == args.start_year else 1,
            args.end_month + 1 if year == args.end_year else 13,
        )
    )
    print(
        f"Found {total_months - len(jobs)} existing months; "
        f"{len(jobs)} monthly packages are missing.",
        flush=True,
    )
    if args.dry_run:
        for year, months in jobs:
            print(f"Would download {year}-{months[0]:02d}", flush=True)
        return

    for job in jobs:
        download_job(
            job, args.output_dir, VARIABLES, "pentland_era5_wind_waves"
        )


if __name__ == "__main__":
    main()
