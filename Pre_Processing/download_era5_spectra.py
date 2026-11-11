"""Retrieve full ERA5 wave spectra for subsequent decoding and fitting."""

import argparse
import calendar
import json
from pathlib import Path

import cdsapi


DEFAULT_DIR = Path(__file__).resolve().parent / "era5_spectra"


def spectral_request(year, month):
    last_day = calendar.monthrange(year, month)[1]
    return {
        "class": "ea", "expver": "1", "stream": "wave",
        "type": "an", "levtype": "sfc", "param": "140251",
        "date": f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day}",
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
                print(f"Would retrieve 720 spectral bins/hour: {target}")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".grib.part")
            metadata = target.with_suffix(".request.json")
            metadata.write_text(json.dumps({
                "dataset": "reanalysis-era5-complete", "request": request,
                "interpretation": (
                    "Raw archived spectra; not yet decoded or fitted. "
                    "GRIB envelope checks do not prove complete coverage. "
                    "Verify log10 encoding, missing bins, frequency and "
                    "direction metadata before integrating or fitting."
                ),
            }, indent=2) + "\n", encoding="utf-8")
            if not valid_grib(partial):
                if client is None:
                    client = cdsapi.Client()
                print(f"Retrieving full spectra: {target.name}", flush=True)
                client.retrieve(
                    "reanalysis-era5-complete", request, str(partial)
                )
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
