import argparse
import calendar
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from datetime import date
import filecmp
from pathlib import Path
import re
from threading import local
import zipfile

import cdsapi
from era5_cache import WAVE_SUPPLEMENT_VARIABLES


START_YEAR = 2006
START_MONTH = 1
END_YEAR = 2026
PACKAGE_MONTHS = 1
OUTPUT_DIR = Path(__file__).resolve().parent / 'era5_monthly'
SUPPLEMENT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / 'era5_wave_supplement'
)
FILE_PATTERN = re.compile(
    r'pentland_era5_wind_waves_(?P<year>\d{4})_(?P<first>\d{2})'
    r'(?:-(?P<last>\d{2}))?\.(?P<extension>zip|nc)'
)

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
    'significant_height_of_total_swell',
    'peak_wave_period',
    'mean_wave_period',
    'wave_spectral_peakedness',
    'mean_wave_direction',
    'mean_direction_of_wind_waves',
    'mean_direction_of_total_swell',
    # Period and spectral descriptors used by the analysis
    'mean_zero_crossing_wave_period',
    'mean_period_of_wind_waves',
    'mean_period_of_total_swell',
    'wave_spectral_directional_width',
    'wave_spectral_skewness',
    'wave_spectral_kurtosis',
    'maximum_individual_wave_height',
    'period_of_maximum_individual_wave',
    'model_bathymetry',
]
TIMES = [f'{hour:02d}:00' for hour in range(24)]
CLIENT_LOCAL = local()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--workers', type=int, default=2,
        help='number of concurrent CDS downloads (default: 2)',
    )
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--start-month', type=int, default=START_MONTH)
    parser.add_argument('--end-year', type=int)
    parser.add_argument('--end-month', type=int)
    parser.add_argument(
        '--years-per-package', type=int, default=1,
        help=(
            'For wave supplements, group this many years into each '
            'calendar-month request (for example, 10).'
        ),
    )
    parser.add_argument(
        '--wave-supplement', action='store_true',
        help=(
            'Download wave-only supplementary variables into a separate '
            'directory.'
        ),
    )
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='list missing monthly packages without downloading them',
    )
    args = parser.parse_args()
    today = date.today()
    latest_year = today.year
    latest_month = today.month - 2
    if latest_month < 1:
        latest_year -= 1
        latest_month += 12
    if args.end_year is None:
        args.end_year = min(END_YEAR, latest_year)
    if args.end_month is None:
        args.end_month = (
            12 if args.end_year < latest_year else latest_month
        )
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if args.years_per_package < 1:
        parser.error('--years-per-package must be at least 1')
    if args.years_per_package > 1 and not args.wave_supplement:
        parser.error('--years-per-package requires --wave-supplement')
    if not 1 <= args.end_month <= 12:
        parser.error('--end-month must be between 1 and 12')
    if args.start_year < START_YEAR:
        parser.error('--start-year cannot be before the configured start year')
    if not 1 <= args.start_month <= 12:
        parser.error('--start-month must be between 1 and 12')
    if args.end_year < args.start_year:
        parser.error('--end-year cannot be before --start-year')
    if (args.start_year == args.end_year
            and args.start_month > args.end_month):
        parser.error('--start-month cannot be after --end-month')
    return args


def discover_existing_months(
    output_dir=OUTPUT_DIR,
    archive_prefix='pentland_era5_wind_waves',
):
    """Return months covered by valid archives or nonempty NetCDF files."""
    existing = set()
    if not output_dir.exists():
        return existing

    pattern = re.compile(
        rf'{re.escape(archive_prefix)}_(?P<year>\d{{4}})_'
        rf'(?P<first>\d{{2}})(?:-(?P<last>\d{{2}}))?\.(?P<extension>zip|nc)'
    )
    for path in output_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if not path.is_file() or match is None:
            continue
        first_month = int(match['first'])
        last_month = int(match['last'] or first_month)
        if not 1 <= first_month <= last_month <= 12:
            continue
        if match['extension'] == 'zip':
            if not zipfile.is_zipfile(path):
                continue
        elif path.stat().st_size == 0:
            continue

        year = int(match['year'])
        existing.update(
            (year, month) for month in range(first_month, last_month + 1)
        )
    return existing


def normalize_download(path, variables):
    if zipfile.is_zipfile(path):
        return
    import netCDF4

    wave_only = not any(
        'component_of_wind' in name or 'wind_gust' in name
        for name in variables
    )
    if not wave_only:
        raise RuntimeError(f'Expected separate wind/wave ZIP streams: {path}')
    try:
        with netCDF4.Dataset(path) as dataset:
            required = {'valid_time', 'latitude', 'longitude'}
            expected = {
                WAVE_SUPPLEMENT_VARIABLES[name] for name in variables
                if name in WAVE_SUPPLEMENT_VARIABLES
            }
            if not required.issubset(dataset.variables) or not expected:
                raise ValueError('Missing wave coordinates or variables')
            if not expected.issubset(dataset.variables):
                raise ValueError('Missing requested wave variables')
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Invalid wave NetCDF response: {path}') from error
    packaged = path.with_name(path.name + '.packaged')
    try:
        with zipfile.ZipFile(packaged, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(path, arcname='data_stream-wave_stepType-instant.nc')
        packaged.replace(path)
    finally:
        packaged.unlink(missing_ok=True)


def download_months(
    client, year, months, output_dir=OUTPUT_DIR, variables=VARIABLES,
    archive_prefix='pentland_era5_wind_waves', request_days=None,
    target_name=None,
):
    label = f'{months[0]:02d}'
    if len(months) > 1:
        label += f'-{months[-1]:02d}'
    target = output_dir / (
        target_name or f'{archive_prefix}_{year}_{label}.zip'
    )
    temporary = target.with_suffix('.zip.part')

    if target.exists():
        if not zipfile.is_zipfile(target):
            raise RuntimeError(f'Existing target is not a valid ZIP: {target}')
        print(f'Skipping existing {target}', flush=True)
        return

    if temporary.exists():
        if not zipfile.is_zipfile(temporary):
            try:
                normalize_download(temporary, variables)
            except RuntimeError:
                pass
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

    years = year if isinstance(year, list) else [year]
    days = request_days or list(range(
        1, max(calendar.monthrange(value, month)[1]
               for value in years for month in months) + 1
    ))
    request = {
        'product_type': ['reanalysis'],
        'variable': variables,
        'year': [str(value) for value in years],
        'month': [f'{month:02d}' for month in months],
        'day': [f'{day:02d}' for day in days],
        'time': TIMES,
        'area': [59.0, -4.0, 58.75, -3.75],
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }

    print(f'Downloading {target.name}...', flush=True)
    try:
        client.retrieve(
            'reanalysis-era5-single-levels', request, str(temporary)
        )
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError(f'Download produced no data: {temporary}')
        normalize_download(temporary, variables)
    except BaseException:
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
    print(f'Finished {target.name}', flush=True)


def download_job(job, output_dir, variables, archive_prefix):
    if not hasattr(CLIENT_LOCAL, 'client'):
        CLIENT_LOCAL.client = cdsapi.Client()
    year, months = job
    download_months(
        CLIENT_LOCAL.client, year, months, output_dir, variables,
        archive_prefix,
    )


def grouped_supplement_jobs(args, output_dir):
    jobs = []
    skipped = 0
    block_start = args.start_year
    while block_start <= args.end_year:
        block_end = min(
            block_start + args.years_per_package - 1, args.end_year
        )
        years = list(range(block_start, block_end + 1))
        for month in range(1, 13):
            if block_start == args.start_year and month < args.start_month:
                continue
            if block_end == args.end_year and month > args.end_month:
                continue
            valid_days = {
                year: calendar.monthrange(year, month)[1] for year in years
            }
            common_days = min(valid_days.values())
            day_groups = [(years, list(range(1, common_days + 1)))]
            for day in range(common_days + 1, max(valid_days.values()) + 1):
                eligible_years = [
                    year for year in years if valid_days[year] >= day
                ]
                if eligible_years:
                    day_groups.append((eligible_years, [day]))
            for request_years, days in day_groups:
                year_label = f'{request_years[0]}-{request_years[-1]}'
                day_label = f'{days[0]:02d}-{days[-1]:02d}'
                name = (
                    f'pentland_era5_wave_supplement_v2_{year_label}_'
                    f'm{month:02d}_d{day_label}.zip'
                )
                target = output_dir / name
                if target.is_file() and zipfile.is_zipfile(target):
                    skipped += 1
                    continue
                jobs.append({
                    'years': request_years,
                    'month': month,
                    'days': days,
                    'target_name': name,
                })
        block_start = block_end + 1
    return jobs, skipped


def download_grouped_supplement_job(job, output_dir, variables):
    if not hasattr(CLIENT_LOCAL, 'client'):
        CLIENT_LOCAL.client = cdsapi.Client()
    download_months(
        CLIENT_LOCAL.client, job['years'],
        job.get('months', [job['month']]), output_dir,
        variables, 'pentland_era5_wave_supplement_v2', job['days'],
        job['target_name'],
    )


def main():
    args = parse_args()
    output_dir = args.output_dir or (
        SUPPLEMENT_OUTPUT_DIR if args.wave_supplement else OUTPUT_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_prefix = (
        'pentland_era5_wave_supplement_v2' if args.wave_supplement
        else 'pentland_era5_wind_waves'
    )
    variables = (
        list(WAVE_SUPPLEMENT_VARIABLES)
        if args.wave_supplement else VARIABLES
    )
    if args.wave_supplement and args.years_per_package > 1:
        jobs, skipped = grouped_supplement_jobs(args, output_dir)
        print(
            f'Found {skipped} existing packages; {len(jobs)} packages '
            'are missing.',
            flush=True,
        )
        if args.dry_run:
            for job in jobs:
                print(f"Would download {job['target_name']}", flush=True)
            return
        jobs_iter = iter(jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            for _ in range(args.workers):
                job = next(jobs_iter, None)
                if job is not None:
                    pending[executor.submit(
                        download_grouped_supplement_job,
                        job, output_dir, variables,
                    )] = job
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    future.result()
                    job = next(jobs_iter, None)
                    if job is not None:
                        pending[executor.submit(
                            download_grouped_supplement_job,
                            job, output_dir, variables,
                        )] = job
        return

    existing_months = discover_existing_months(output_dir, archive_prefix)
    jobs = []
    skipped_months = 0
    for year in range(args.start_year, args.end_year + 1):
        start_month = args.start_month if year == args.start_year else 1
        last_month = args.end_month if year == args.end_year else 12
        for first_month in range(start_month, last_month + 1, PACKAGE_MONTHS):
            months = list(
                range(
                    first_month,
                    min(last_month + 1, first_month + PACKAGE_MONTHS),
                )
            )
            missing = [
                month for month in months
                if (year, month) not in existing_months
            ]
            if not missing:
                skipped_months += len(months)
            elif len(missing) == len(months):
                jobs.append((year, months))
            else:
                jobs.extend((year, [month]) for month in missing)

    print(
        f'Found {skipped_months} existing months; '
        f'{len(jobs)} monthly packages are missing.',
        flush=True,
    )
    if args.dry_run:
        for year, months in jobs:
            print(
                f'Would download {year}-{months[0]:02d} to {output_dir}',
                flush=True,
            )
        return

    jobs_iter = iter(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending: dict[Future, tuple[int, list[int]]] = {}
        for _ in range(args.workers):
            job = next(jobs_iter, None)
            if job is not None:
                pending[executor.submit(
                    download_job, job, output_dir, variables, archive_prefix
                )] = job

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
                    pending[executor.submit(
                        download_job, next_job, output_dir, variables,
                        archive_prefix,
                    )] = next_job


if __name__ == '__main__':
    main()
