import argparse
import calendar
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
import filecmp
from pathlib import Path
from threading import local
import zipfile

import cdsapi


START_YEAR = 2019
START_MONTH = 7
END_YEAR = 2025
PACKAGE_MONTHS = 1
OUTPUT_DIR = Path(__file__).resolve().parent / 'era5_monthly'

VARIABLES = [
    # Wind parameters
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'instantaneous_10m_wind_gust',
    # Wave parameters
    'significant_height_of_combined_wind_waves_and_swell',
    'significant_height_of_wind_waves',
    'peak_wave_period',
    'mean_wave_period',
    'wave_spectral_peakedness',
    # Period descriptors for sea-state analysis and later JONSWAP fitting
    'mean_zero_crossing_wave_period',
    'mean_period_of_wind_waves',
    'mean_period_of_total_swell',
]

TIMES = [f'{hour:02d}:00' for hour in range(24)]
CLIENT_LOCAL = local()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workers', type=int, default=2,
        help='number of concurrent CDS downloads (default: 2)',
    )
    parser.add_argument('--end-year', type=int, default=END_YEAR)
    parser.add_argument('--end-month', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if not 1 <= args.end_month <= 12:
        parser.error('--end-month must be between 1 and 12')
    if args.end_year < START_YEAR:
        parser.error('--end-year cannot be before the configured start year')
    return args


def download_months(client, year, months):
    label = f'{months[0]:02d}'
    if len(months) > 1:
        label += f'-{months[-1]:02d}'
    target = OUTPUT_DIR / f'pentland_era5_wind_waves_{year}_{label}.zip'
    temporary = target.with_suffix('.zip.part')

    if target.exists():
        if not zipfile.is_zipfile(target):
            raise RuntimeError(f'Existing target is not a valid ZIP: {target}')
        print(f'Skipping existing {target}', flush=True)
        return

    if temporary.exists():
        if zipfile.is_zipfile(temporary):
            try:
                temporary.replace(target)
                print(f'Recovered completed download {target}', flush=True)
                return
            except PermissionError:
                if target.exists() and zipfile.is_zipfile(target):
                    if filecmp.cmp(temporary, target, shallow=False):
                        temporary.unlink()
                        print(
                            f'Archive already finalized by another process: '
                            f'{target}',
                            flush=True,
                        )
                        return
                raise
        temporary.unlink()

    days = max(calendar.monthrange(year, month)[1] for month in months)
    request = {
        'product_type': ['reanalysis'],
        'variable': VARIABLES,
        'year': [str(year)],
        'month': [f'{month:02d}' for month in months],
        'day': [f'{day:02d}' for day in range(1, days + 1)],
        'time': TIMES,
        'area': [59.0, -4.0, 58.75, -3.75],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }

    print(f'Downloading {year}-{label} to {target}...', flush=True)
    try:
        client.retrieve(
            'reanalysis-era5-single-levels', request, str(temporary)
        )
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f'Download produced no data: {temporary}')
        if not zipfile.is_zipfile(temporary):
            raise RuntimeError(
                'Expected the CDS NetCDF stream package to be a ZIP archive: '
                f'{temporary}'
            )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        temporary.replace(target)
    except PermissionError as error:
        if target.exists() and zipfile.is_zipfile(target):
            if filecmp.cmp(temporary, target, shallow=False):
                temporary.unlink()
                print(
                    f'Archive already finalized by another process: {target}',
                    flush=True,
                )
                return
        raise PermissionError(
            f'Valid download remains at {temporary}; Windows could not rename '
            f'it to {target}. Close other processes using the target, let '
            'OneDrive finish syncing, then rerun to recover the .part file.'
        ) from error
    print(f'Finished {year}-{label}', flush=True)


def download_job(job):
    if not hasattr(CLIENT_LOCAL, 'client'):
        CLIENT_LOCAL.client = cdsapi.Client()
    year, months = job
    download_months(CLIENT_LOCAL.client, year, months)


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    jobs = []
    for year in range(START_YEAR, args.end_year + 1):
        start_month = START_MONTH if year == START_YEAR else 1
        last_month = args.end_month if year == args.end_year else 12
        for first_month in range(start_month, last_month + 1, PACKAGE_MONTHS):
            months = list(
                range(
                    first_month,
                    min(last_month + 1, first_month + PACKAGE_MONTHS),
                )
            )
            jobs.append((year, months))

    jobs_iter = iter(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: dict[Future, tuple[int, list[int]]] = {}
        for _ in range(args.workers):
            job = next(jobs_iter, None)
            if job is not None:
                pending[executor.submit(download_job, job)] = job

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job = pending.pop(future)
                try:
                    future.result()
                except BaseException:
                    for remaining in pending:
                        remaining.cancel()
                    raise
                next_job = next(jobs_iter, None)
                if next_job is not None:
                    pending[executor.submit(download_job, next_job)] = next_job


if __name__ == '__main__':
    main()
