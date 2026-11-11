"""Read and validate prepared ERA5 Parquet caches."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


METRIC_COLUMNS = (
    "shww", "swh", "shts", "pp1d", "wsp", "mwp", "mwd", "mdww", "mdts",
    "mp2", "mpww", "mpts", "wdw", "dwww", "dwps", "wss", "wsk",
    "hmax", "tmax", "wmb", "u10", "v10", "u100", "v100", "i10fg",
)
WAVE_SUPPLEMENT_VARIABLES = {
    "significant_height_of_total_swell": "shts",
    "wave_spectral_peakedness": "wsp",
    "mean_wave_period": "mwp",
    "mean_wave_direction": "mwd",
    "mean_direction_of_wind_waves": "mdww",
    "mean_direction_of_total_swell": "mdts",
    "mean_zero_crossing_wave_period": "mp2",
    "mean_period_of_wind_waves": "mpww",
    "mean_period_of_total_swell": "mpts",
    "wave_spectral_directional_width": "wdw",
    "wave_spectral_skewness": "wss",
    "wave_spectral_kurtosis": "wsk",
    "maximum_individual_wave_height": "hmax",
    "period_of_maximum_individual_wave": "tmax",
    "model_bathymetry": "wmb",
}
DEFAULT_CACHE = Path(__file__).resolve().parent / "era5_processed" / (
    "era5_point_2006_2023.parquet"
)
OPTIONAL_CACHE_COLUMNS = ("shts",)


def load_cache(path: Path, start_year: int, end_year: int):
    """Return the analysis input contract from a prepared Parquet cache."""
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"ERA5 Parquet cache not found: {path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"ERA5 cache metadata not found: {metadata_path}"
        )

    frame = pd.read_parquet(path)
    required = {
        "valid_time", *(set(METRIC_COLUMNS) - set(OPTIONAL_CACHE_COLUMNS))
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "ERA5 cache is missing columns: " + ", ".join(sorted(missing))
        )

    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="raise")
    frame = frame.loc[
        (frame["valid_time"].dt.year >= start_year)
        & (frame["valid_time"].dt.year <= end_year)
    ].sort_values("valid_time")
    times = frame["valid_time"].to_numpy(dtype="datetime64[ns]")
    if np.unique(times).size != times.size:
        raise ValueError("ERA5 cache contains duplicate timestamps")
    expected = np.arange(
        np.datetime64(f"{start_year}-01-01T00", "ns"),
        np.datetime64(f"{end_year + 1}-01-01T00", "ns"),
        np.timedelta64(1, "h"),
    )
    if not np.array_equal(times, expected):
        raise ValueError(
            "ERA5 cache does not contain a complete hourly record for the "
            f"requested period: found {times.size}, expected {expected.size}"
        )

    metrics = {
        name: (
            frame[name].to_numpy(dtype=float)
            if name in frame else np.full(times.size, np.nan)
        )
        for name in METRIC_COLUMNS
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    location = metadata.get("location", {})
    try:
        latitude = float(location["latitude"])
        longitude = float(location["longitude"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid selected location in cache metadata: {metadata_path}"
        ) from error
    units = metadata.get("units", {})
    for name in METRIC_COLUMNS:
        units.setdefault(name, "unknown")
    return times, metrics, latitude, longitude, units
