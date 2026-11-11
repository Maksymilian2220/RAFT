"""Synthetic checks for constructed, not fitted, site spectra."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jonswap_site_spectra import (
    BASE_DIR, construct_spectrum, export_constructed, iec_gamma,
    raft_density, select_states, summarize_hours,
)


class ConstructedSpectrumTests(unittest.TestCase):
    def test_iec_boundaries_and_calm(self):
        np.testing.assert_allclose(iec_gamma([1, 1, 1], [3.6, 5, 4]),
                                   [5, 1, np.exp(1.15)])
        self.assertTrue(np.isnan(iec_gamma(0, 8)))
        self.assertGreater(float(iec_gamma(1, 3.600001)), 5)

    def test_invalid_inputs(self):
        for height, period in [(1, 0), (1, -1), (-1, 8),
                               (np.nan, 8), (1, np.inf)]:
            with self.assertRaises(ValueError):
                construct_spectrum(height, period, 3.3)
        for gamma in [0, np.nan, np.inf, 11]:
            with self.assertRaises(ValueError):
                construct_spectrum(1, 8, gamma)
        with self.assertRaises(ValueError):
            summarize_hours([np.nan], [8])

    def test_normalization_and_positive_grid(self):
        for height, period in [(0, 8), (0.1, 1), (2, 8), (12, 20)]:
            for gamma in [1, 3.3, 5]:
                frequency, density, checks = construct_spectrum(
                    height, period, gamma)
                self.assertTrue(
                    np.all(
                        np.isfinite(frequency) & (
                            frequency > 0)))
                self.assertTrue(np.all(np.diff(frequency) > 0))
                self.assertTrue(np.all(np.isfinite(density) & (density >= 0)))
                self.assertAlmostEqual(
                    np.trapezoid(
                        density,
                        frequency),
                    height**2 / 16,
                    places=12)
                self.assertAlmostEqual(
                    checks["reconstructed_hs_m"], height, places=12)
                self.assertLess(
                    checks["omitted_raw_energy_relative_bound"], 1e-5)
                self.assertLess(checks["grid_refinement_relative_error"], 1e-5)
                self.assertLessEqual(checks["peak_step_over_fp"], .002)

    def test_helper_formula_and_units(self):
        tree = ast.parse((BASE_DIR.parent / "raft/helpers.py").read_text())
        function = next(node for node in tree.body if isinstance(
            node, ast.FunctionDef) and node.name == "JONSWAP")
        namespace = {"np": np}
        exec(
            compile(
                ast.Module(
                    body=[function],
                    type_ignores=[]),
                "helper_reference",
                "exec"),
            namespace)
        frequency = np.geomspace(.001, 10, 20000)
        for gamma in [1, 3.3, 5]:
            actual = raft_density(frequency, 2, 8, gamma)
            expected = namespace["JONSWAP"](2 * np.pi * frequency, 2, 8, gamma)
            np.testing.assert_allclose(actual, expected, rtol=1e-13)
        self.assertAlmostEqual(np.trapezoid(raft_density(
            frequency, 2, 8, 1), frequency), .25 / (2 * np.pi), places=8)

    def test_pair_counts_and_deterministic_selection(self):
        height, period, counts = summarize_hours(
            [0, 1, 1, 2, np.nan, -1, 2], [8, 6, 6, 9, 8, 0, np.inf])
        self.assertEqual(counts["valid_pair_hours"], 4)
        self.assertEqual(counts["excluded_invalid_pair_hours"], 3)
        self.assertEqual(counts["zero_hs_valid_pair_hours"], 1)
        states, _ = select_states(height, period)
        reverse, _ = select_states(height[::-1], period[::-1])
        self.assertEqual(states, reverse)
        self.assertEqual(states[0]["count"], 2)
        self.assertEqual(states[0]["probability"], .5)
        ties, _ = select_states(np.array([1., 1., 2.]), np.array([7., 6., 8.]))
        self.assertEqual(ties[0]["tp_s"], 6.125)

    def test_hourly_summary_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            metadata = export_constructed([1, 1, 1, 1, 0, np.nan], [
                                          3, 3, 3, 6, 8, 8], directory, {})
            self.assertEqual(metadata["hourly_iec_gamma"]["mean"], 4)
            self.assertEqual(metadata["hourly_iec_gamma"]["median"], 5)
            self.assertEqual(metadata["hourly_iec_gamma"]["hours"], 4)
            self.assertEqual(len(metadata["spectrum_checks"]), 16)
            self.assertEqual(len(list(directory.glob("jonswap_*"))), 4)
            json.loads(
                (directory / "jonswap_constructed_metadata.json").read_text())

    def test_optional_export_to_plot(self):
        from plot_wave_results import plot_jonswap_constructed

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plot_jonswap_constructed(directory, directory)
            self.assertEqual(list(directory.glob("*.png")), [])
            export_constructed([1, 1, 2, 3], [6, 6, 8, 10], directory, {})
            plot_jonswap_constructed(directory, directory)
            self.assertEqual(
                {path.name for path in directory.glob("*.png")},
                {"jonswap_constructed_spectra.png",
                 "jonswap_iec_gamma_distribution.png", })
            self.assertTrue(
                all(path.stat().st_size > 1000
                    for path in directory.glob("*.png"))
            )


if __name__ == "__main__":
    unittest.main()
