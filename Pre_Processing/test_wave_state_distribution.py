import tempfile
import unittest
from pathlib import Path

import numpy as np

import wave_state_distribution as analysis


class ServiceabilityTests(unittest.TestCase):
    def test_thresholds_overlap_and_missing_wind(self):
        valid, retained, summary = analysis.serviceability_filters(
            np.array([1., 1., 2.5, 3., 1., np.nan]),
            np.ones(6), np.array([0., 12., 0., 20., np.nan, 0.]),
            np.array([0., 16., 0., 0., 0., 0.]), 20., 2.5,
        )
        self.assertEqual(int(valid.sum()), 5)
        self.assertEqual(retained.tolist(),
                         [True, False, False, False, False, False])
        self.assertEqual(summary["wind_constraint"]["hours"], 2)
        self.assertEqual(summary["height_constraint"]["hours"], 2)
        self.assertEqual(summary["both_constraints"]["hours"], 1)
        self.assertEqual(summary["either_constraint"]["hours"], 3)
        self.assertEqual(summary["missing_wind"]["hours"], 1)
        self.assertEqual(summary["retained"]["percentage_of_valid"], 20.)

    def test_signed_wind_is_converted_to_magnitude(self):
        _, retained, _ = analysis.serviceability_filters(
            np.ones(2), np.ones(2), np.array([-20., -19.]),
            np.zeros(2), 20., 2.5,
        )
        self.assertEqual(retained.tolist(), [False, True])

    def test_archive_ownership_and_missing_months(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "pentland_era5_wind_waves_2006_01-12.nc"
            bundle.touch()
            monthly = root / "pentland_era5_wind_waves_2006_01.zip"
            monthly.touch()
            owners = analysis.discover_archives(root, 2006, 2006)
            self.assertEqual(len(owners), 12)
            self.assertEqual(owners["2006-01"], monthly)
            self.assertEqual(owners["2006-02"], bundle)
            with self.assertRaisesRegex(ValueError, "Missing ERA5 months"):
                analysis.discover_archives(root, 2006, 2007)


if __name__ == "__main__":
    unittest.main()
