"""Offline regression tests for the historical swell-height backfill."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import netCDF4
import numpy as np
import pandas as pd

from backfill_era5_shts import (
    ARCHIVE_PREFIX, main, merge_backfill, missing_jobs, grouped_jobs,
    BULK_VARIABLES, merge_grouped_backfill,
    bundled_jobs,
)
from prepare_era5_cache import discover_supplement_archives


class BackfillTests(unittest.TestCase):
    def test_stages_plan_disjoint_years(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recent, _ = bundled_jobs(root, 1, 3, 2025, 2025)
            history, _ = bundled_jobs(root, 1, 3, 2006, 2024)
        self.assertEqual(len(recent), 6)
        self.assertEqual({year for job in recent for year in job["years"]},
                         {2025})
        self.assertEqual({year for job in history for year in job["years"]},
                         set(range(2006, 2025)))

    def test_bundles_cover_every_hour_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs, _ = bundled_jobs(root)
            self.assertEqual(len(jobs), 14)
            hours = []
            for job in jobs:
                for year in job["years"]:
                    for month in job["months"]:
                        for day in job["days"]:
                            start = pd.Timestamp(year, month, day)
                            hours.extend(pd.date_range(
                                start, periods=24, freq="h"
                            ))
            index = pd.DatetimeIndex(hours).sort_values()
            self.assertFalse(index.has_duplicates)
            pd.testing.assert_index_equal(index, pd.date_range(
                "2006-01-01", "2025-12-31 23:00", freq="h"
            ))
            target = root / jobs[0]["target_name"]
            with ZipFile(target, "w") as archive:
                archive.writestr("placeholder", "test")
            pending, skipped = bundled_jobs(root)
            self.assertEqual((len(pending), skipped), (13, 1))
            self.assertEqual(
                discover_supplement_archives(root, 2006, 2025), [target]
            )
            maximum, _ = bundled_jobs(Path(temporary) / "empty", 10, 12)
            self.assertEqual(len(maximum), 8)

    def test_grouped_merge_requires_all_six_fields_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, output = root / "base.parquet", root / "output.parquet"
            metrics = {name: np.array([1.0, 2.0])
                       for name, _ in BULK_VARIABLES.values()}
            units = dict(BULK_VARIABLES.values())
            units["mwd"] = "Degree true"
            contract = (None, {}, 59.0, -4.0, {})
            with (
                patch("backfill_era5_shts.load_cache",
                      return_value=contract),
                patch("backfill_era5_shts.read_supplement_archives",
                      return_value=(None, metrics, units, None, [])),
                patch("backfill_era5_shts.merge_existing_cache") as merge,
            ):
                merge_grouped_backfill(base, root, output)
                merge.assert_called_once_with(base, root, 2006, 2025, output)
                merge.reset_mock()
                metrics["mpww"][0] = np.nan
                with self.assertRaisesRegex(ValueError, "mpww"):
                    merge_grouped_backfill(base, root, output)
                merge.assert_not_called()

    def test_grouped_decades_include_all_hours_and_leap_days(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs, skipped = grouped_jobs(Path(temporary))
        self.assertEqual(len(jobs), 26)
        self.assertEqual(skipped, 0)
        hours = pd.DatetimeIndex([])
        for job in jobs:
            dates = [
                pd.Timestamp(year=year, month=job["month"], day=day)
                for year in job["years"] for day in job["days"]
            ]
            daily_hours = pd.DatetimeIndex([
                date + pd.Timedelta(hours=hour)
                for date in dates for hour in range(24)
            ])
            hours = hours.append(daily_hours)
        self.assertFalse(hours.has_duplicates)
        pd.testing.assert_index_equal(
            hours.sort_values(),
            pd.date_range("2006-01-01", "2025-12-31 23:00", freq="h"),
        )
        self.assertEqual(
            {metric for metric, _ in BULK_VARIABLES.values()},
            {"shts", "mpww", "mpts", "mwd", "mdww", "mdts"},
        )

    def test_grouped_resume_and_smaller_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs, _ = grouped_jobs(root)
            with ZipFile(root / jobs[0]["target_name"], "w") as archive:
                archive.writestr("placeholder", "test")
            pending, skipped = grouped_jobs(root)
            self.assertEqual((len(pending), skipped), (25, 1))
            yearly, _ = grouped_jobs(root, 1)
            self.assertEqual(len(yearly), 240)

    def test_exact_requested_month_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = missing_jobs(Path(temporary))
        self.assertEqual(len(jobs), 234)
        self.assertEqual(jobs[0], (2006, [1]))
        self.assertEqual(jobs[-1], (2025, [6]))

    def test_resume_skips_completed_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / f"{ARCHIVE_PREFIX}_2006_01.zip"
            with ZipFile(target, "w") as archive:
                archive.writestr("placeholder", "test")
            jobs = missing_jobs(root)
        self.assertEqual(len(jobs), 233)
        self.assertNotIn((2006, [1]), jobs)

    def test_dry_run_never_downloads_or_creates_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "absent"
            with patch("backfill_era5_shts.download_job") as download:
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["--download-dir", str(root), "--dry-run"])
                download.assert_not_called()
            self.assertFalse(root.exists())

    def test_merge_refuses_base_overwrite(self):
        base = Path("base.parquet")
        with self.assertRaisesRegex(ValueError, "differ"):
            merge_backfill(base, Path("archives"), base)

    def test_merge_validates_archives_and_preserves_base(self):
        cases = (
            ("valid", 59.0, [0, 1], [1.0, 2.0], None),
            ("wrong_grid", 58.5, [0, 1], [1.0, 2.0], "grid point"),
            ("missing_hour", 59.0, [0], [1.0], "timestamps"),
            ("negative_height", 59.0, [0, 1], [-1.0, 2.0], "Invalid"),
            ("conflict", 59.0, [0, 1], [1.0, 3.0], "Conflicting"),
        )
        for label, latitude, hours, values, error in cases:
            with (
                self.subTest(case=label),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                base, output = root / "base.parquet", root / "merged.parquet"
                times = np.array([
                    "2006-01-01T00", "2006-01-01T01", "2025-07-01T00",
                ], dtype="datetime64[ns]")
                frame = pd.DataFrame({
                    "valid_time": times, "shts": [np.nan, 2.0, 4.0],
                    "shww": [0.5, 0.6, 0.7],
                })
                frame.to_parquet(base, index=False)
                base.with_suffix(".json").write_text(json.dumps({
                    "location": {"latitude": 59.0, "longitude": -4.0},
                    "variables": ["shww", "shts"], "units": {},
                }), encoding="utf-8")
                source = root / "wave_stepType-instant.nc"
                with netCDF4.Dataset(source, "w") as dataset:
                    dimensions = (
                        ("valid_time", len(hours)),
                        ("latitude", 1), ("longitude", 1),
                    )
                    for dimension, size in dimensions:
                        dataset.createDimension(dimension, size)
                    time_variable = dataset.createVariable(
                        "valid_time", "i8", ("valid_time",)
                    )
                    time_variable.units = "hours since 2006-01-01 00:00:00"
                    time_variable[:] = hours
                    dataset.createVariable(
                        "latitude", "f8", ("latitude",)
                    )[:] = [latitude]
                    dataset.createVariable(
                        "longitude", "f8", ("longitude",)
                    )[:] = [-4.0]
                    height = dataset.createVariable(
                        "shts", "f8", ("valid_time", "latitude", "longitude")
                    )
                    height.units = "m"
                    height[:] = np.asarray(values).reshape(-1, 1, 1)
                archive_path = root / f"{ARCHIVE_PREFIX}_2006_01.zip"
                with ZipFile(archive_path, "w") as archive:
                    archive.write(source, arcname=source.name)
                contract = (
                    times, {"shts": frame.shts.to_numpy()}, 59.0, -4.0, {}
                )
                with (
                    patch("backfill_era5_shts.load_cache",
                          return_value=contract),
                    patch("backfill_era5_shts.requested_months",
                          return_value=pd.period_range(
                              "2006-01", "2006-01", freq="M"
                          )),
                ):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            merge_backfill(base, root, output)
                        self.assertFalse(output.exists())
                    else:
                        with contextlib.redirect_stdout(io.StringIO()):
                            merge_backfill(base, root, output)
                        result = pd.read_parquet(output)
                        np.testing.assert_array_equal(
                            result.shts, [1.0, 2.0, 4.0]
                        )
                        pd.testing.assert_series_equal(result.shww, frame.shww)
                        metadata = json.loads(
                            output.with_suffix(".json").read_text()
                        )
                        self.assertEqual(
                            metadata["shts_backfill"]["filled_hours"], 1
                        )
                        with self.assertRaises(FileExistsError):
                            merge_backfill(base, root, output)
                pd.testing.assert_frame_equal(pd.read_parquet(base), frame)


if __name__ == "__main__":
    unittest.main()
