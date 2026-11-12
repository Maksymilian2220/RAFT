"""Retrieve full ERA5 wave spectra for subsequent decoding and fitting."""

import argparse
import calendar
import json
from pathlib import Path
import shutil

import cdsapi


DEFAULT_DIR = Path(__file__).resolve().parent / "era5_spectra"


def spectral_request(year, month, day=None):
    last_day = calendar.monthrange(year, month)[1]
    if day is not None and not 1 <= day <= last_day:
        raise ValueError("Day is outside the requested month")
    dates = (
        f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}"
        if day is None else f"{year}-{month:02d}-{day:02d}"
    )
    return {
        "class": "ea", "expver": "1", "stream": "wave",
        "type": "an", "levtype": "sfc", "param": "140251",
        "date": dates,
        "time": "/".join(f"{hour:02d}:00:00" for hour in range(24)),
        "frequency": "1/to/30", "direction": "1/to/24",
        "area": "59/-4/59/-4", "grid": "0.5/0.5",
        "format": "grib",
    }


def valid_grib(path):
    if not path.is_file() or path.stat().st_size < 12:
        return False
    with path.open("rb") as handle:
        if handle.read(4) != b"GRIB":
            return False
        handle.seek(-4, 2)
        return handle.read(4) == b"7777"


def retrieve_spectra(start_year, end_year, output_dir=None, dry_run=False):
    output_dir = Path(output_dir) if output_dir else DEFAULT_DIR
    if start_year > end_year:
        raise ValueError("Start year must not exceed end year")
    client = None
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            target = output_dir / f"era5_spectra_{year}_{month:02d}.grib"
            if target.exists():
                if not valid_grib(target):
                    raise ValueError(f"Invalid existing GRIB: {target}")
                print(f"Skipping {target.name}", flush=True)
                continue
            request = spectral_request(year, month)
            if dry_run:
                days = calendar.monthrange(year, month)[1]
                print(
                    f"Would retrieve {days} daily requests, "
                    f"720 spectral bins/hour: {target}"
                )
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".grib.part")
            metadata = target.with_suffix(".request.json")
            metadata.write_text(json.dumps({
                "dataset": "reanalysis-era5-complete", "request": request,
                "request_partition": "One request per calendar day",
                "interpretation": (
                    "Raw archived spectra; not yet decoded or fitted. "
                    "GRIB envelope checks do not prove complete coverage. "
                    "Verify log10 encoding, missing bins, frequency and "
                    "direction metadata before integrating or fitting."
                ),
            }, indent=2) + "\n", encoding="utf-8")
            if not valid_grib(partial):
                parts_dir = output_dir / f"{target.stem}.parts"
                parts_dir.mkdir(parents=True, exist_ok=True)
                daily_paths = []
                for day in range(1, calendar.monthrange(year, month)[1] + 1):
                    daily = parts_dir / f"day_{day:02d}.grib"
                    daily_paths.append(daily)
                    if valid_grib(daily):
                        continue
                    if daily.exists():
                        raise ValueError(f"Invalid existing GRIB: {daily}")
                    daily_partial = daily.with_suffix(".grib.part")
                    if not valid_grib(daily_partial):
                        if client is None:
                            client = cdsapi.Client()
                        print(
                            f"Retrieving full spectra: {target.stem} "
                            f"day {day:02d}", flush=True,
                        )
                        client.retrieve(
                            "reanalysis-era5-complete",
                            spectral_request(year, month, day),
                            str(daily_partial),
                        )
                    if not valid_grib(daily_partial):
                        raise ValueError(
                            f"Response is not a GRIB file: {daily_partial}"
                        )
                    daily_partial.replace(daily)
                assembly = target.with_suffix(".grib.assembling")
                with assembly.open("wb") as destination:
                    for daily in daily_paths:
                        with daily.open("rb") as source:
                            shutil.copyfileobj(source, destination)
                assembly.replace(partial)
            if not valid_grib(partial):
                raise ValueError(f"Response is not a GRIB file: {partial}")
            partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    retrieve_spectra(
        args.start_year, args.end_year, args.output_dir, args.dry_run
    )


if __name__ == "__main__":
    main()
