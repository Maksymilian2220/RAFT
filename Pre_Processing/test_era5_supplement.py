"""Tests for wave-only ERA5 cache supplementation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile
from argparse import Namespace

import numpy as np
import pandas as pd
import xarray as xr

from era5_cache import (
    METRIC_COLUMNS,
    OPTIONAL_CACHE_COLUMNS,
    WAVE_SUPPLEMENT_VARIABLES,
    load_cache,
)
from download_era5 import VARIABLES, grouped_supplement_jobs
from prepare_era5_cache import (
    discover_supplement_archives,
    merge_existing_cache,
    overlay_supplement_metrics,
)


class WaveSupplementTests(unittest.TestCase):
    def test_mars_ambiguous_spectral_widths_are_not_requested(self):
        invalid_variables = {
            "wind_wave_spectral_directional_width",
            "swell_wave_spectral_directional_width",
        }
        self.assertTrue(invalid_variables.isdisjoint(VARIABLES))
        self.assertTrue(
            invalid_variables.isdisjoint(WAVE_SUPPLEMENT_VARIABLES)
        )

    def test_swell_height_is_requested_and_mapped_to_cache(self):
        self.assertIn("significant_height_of_total_swell", VARIABLES)
        self.assertEqual(
            WAVE_SUPPLEMENT_VARIABLES["significant_height_of_total_swell"],
            "shts",
        )
        self.assertIn("shts", METRIC_COLUMNS)

    def test_legacy_cache_without_shts_remains_loadable(self):
        times = pd.date_range("2006-01-01", "2006-12-31 23:00", freq="h")
        data = {
            name: np.full(len(times), np.nan)
            for name in set(METRIC_COLUMNS) - set(OPTIONAL_CACHE_COLUMNS)
        }
        data["shww"] = np.full(len(times), 1.0)
        data["pp1d"] = np.full(len(times), 8.0)
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "legacy.parquet"
            pd.DataFrame({"valid_time": times, **data}).to_parquet(
                cache_path, index=False
            )
            cache_path.with_suffix(".json").write_text(json.dumps({
                "location": {"latitude": 59.0, "longitude": -4.0},
                "units": {},
            }), encoding="utf-8")

            _, metrics, _, _, _ = load_cache(cache_path, 2006, 2006)

        self.assertTrue(np.isnan(metrics["shts"]).all())

    def test_ten_year_blocks_plan_months_and_leap_days(self):
        args = Namespace(
            start_year=2006, end_year=2025, start_month=1,
            end_month=12, years_per_package=10,
        )
        with tempfile.TemporaryDirectory() as temporary:
            jobs, skipped = grouped_supplement_jobs(args, Path(temporary))

        self.assertEqual(skipped, 0)
        self.assertEqual(len(jobs), 26)
        february_leap_jobs = [
            job for job in jobs
            if job["month"] == 2 and job["days"] == [29]
        ]
        self.assertEqual(len(february_leap_jobs), 2)
        self.assertEqual(february_leap_jobs[0]["years"], [2008, 2012])
        self.assertEqual(february_leap_jobs[1]["years"], [2016, 2020, 2024])
        self.assertTrue(all(
            job["days"] == list(range(1, 29))
            for job in jobs if job["month"] == 2
            and job["days"] != [29]
        ))

    def test_grouped_archive_discovery_recognizes_leap_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / (
                "pentland_era5_wave_supplement_v2_2006-2015_"
                "m02_d29-29.zip"
            )
            with ZipFile(archive_path, "w"):
                pass

            archives = discover_supplement_archives(root, 2006, 2015)

        self.assertEqual(archives, [archive_path])

    def test_grouped_jobs_use_new_variable_set_version(self):
        args = Namespace(
            start_year=2006, end_year=2006, start_month=1,
            end_month=1, years_per_package=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            jobs, _ = grouped_supplement_jobs(args, Path(temporary))
        self.assertTrue(
            jobs[0]["target_name"].startswith(
                "pentland_era5_wave_supplement_v2_"
            )
        )

    def test_overlay_fills_missing_values_without_replacing_existing(self):
        times = np.array(
            ["2006-01-01T00", "2006-01-01T01", "2006-01-01T02"],
            dtype="datetime64[ns]",
        )
        base = {"mwd": np.array([np.nan, 45.0, np.nan])}
        supplement = {"mwd": np.array([350.0, 90.0, np.nan])}

        merged, filled = overlay_supplement_metrics(
            times, base, times, supplement
        )

        np.testing.assert_allclose(
            merged["mwd"], [350.0, 45.0, np.nan], equal_nan=True
        )
        self.assertEqual(filled["mwd"], 1)

    def test_overlay_rejects_misaligned_timestamps(self):
        base_times = np.array(
            ["2006-01-01T00", "2006-01-01T01"], dtype="datetime64[ns]"
        )
        supplement_times = np.array(
            ["2006-01-01T01", "2006-01-01T02"], dtype="datetime64[ns]"
        )
        with self.assertRaisesRegex(ValueError, "timestamps do not match"):
            overlay_supplement_metrics(
                base_times, {"mwd": np.array([np.nan, np.nan])},
                supplement_times, {"mwd": np.array([10.0, 20.0])},
            )

    def test_merge_existing_cache_reads_wave_only_archive(self):
        year = 2006
        times = np.arange(
            np.datetime64(f"{year}-01-01T00", "ns"),
            np.datetime64(f"{year + 1}-01-01T00", "ns"),
            np.timedelta64(1, "h"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_cache = root / "base.parquet"
            metadata_path = base_cache.with_suffix(".json")
            supplement_dir = root / "supplements"
            supplement_dir.mkdir()
            output = root / "with_waves.parquet"

            columns = {
                name: np.full(times.size, np.nan) for name in METRIC_COLUMNS
            }
            columns["shww"] = np.full(times.size, 1.0)
            columns["pp1d"] = np.full(times.size, 8.0)
            columns["mwd"][0] = 270.0
            pd.DataFrame({"valid_time": times, **columns}).to_parquet(
                base_cache, index=False
            )
            metadata_path.write_text(json.dumps({
                "location": {"latitude": 59.0, "longitude": -4.0},
                "units": {"mwd": "unavailable in source archive"},
            }), encoding="utf-8")

            direction = (np.arange(times.size) % 360).astype(float)
            dataset = xr.Dataset(
                {"mwd": (("valid_time", "latitude", "longitude"),
                         direction[:, None, None])},
                coords={
                    "valid_time": times,
                    "latitude": [59.0],
                    "longitude": [-4.0],
                },
            )
            netcdf_path = root / "wave_stepType-instant.nc"
            dataset.to_netcdf(netcdf_path, engine="netcdf4")
            archive_path = supplement_dir / (
                "pentland_era5_wave_supplement_2006_01-12.zip"
            )
            with ZipFile(archive_path, "w") as archive:
                archive.write(netcdf_path, "wave_stepType-instant.nc")

            merge_existing_cache(
                base_cache, supplement_dir, year, year, output
            )
            result = pd.read_parquet(output)
            original = pd.read_parquet(base_cache)
            result_metadata = json.loads(
                output.with_suffix(".json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.loc[0, "mwd"], 270.0)
        self.assertEqual(result.loc[1, "mwd"], 1.0)
        self.assertTrue(np.isnan(original.loc[1, "mwd"]))
        self.assertTrue(np.isnan(result.loc[0, "mdww"]))
        self.assertEqual(result_metadata["supplement_values_filled"]["mwd"],
                         times.size - 1)
        self.assertEqual(result_metadata["location"]["latitude"], 59.0)


if __name__ == "__main__":
    unittest.main()
