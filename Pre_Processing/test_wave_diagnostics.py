"""Regression tests for wave diagnostics and optional diagnostic plots."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from era5_cache import METRIC_COLUMNS
from plot_wave_results import (
    plot_directional_probability,
    plot_height_cdf,
    plot_jonswap_fits,
    plot_partition_diagnostics,
    plot_variable_coverage,
    plot_wave_height_comparison,
)
from wave_diagnostics import (
    empirical_distribution,
    export_diagnostics,
    export_spectral_fits,
    fit_jonswap_gamma,
    height_diagnostics,
)


class WaveDiagnosticTests(unittest.TestCase):
    def test_cdf_counts_duplicates_and_rejects_missing_negative(self):
        heights, cdf, counts = empirical_distribution(
            np.array([1., 1., 2., np.nan, -1.])
        )
        np.testing.assert_array_equal(heights, [1., 2.])
        np.testing.assert_array_equal(counts, [2, 1])
        np.testing.assert_allclose(cdf, [2 / 3, 1])

    def test_partition_closure_and_impossible_height_order(self):
        times = np.arange(3).astype("datetime64[h]")
        metrics = {name: np.full(3, np.nan) for name in METRIC_COLUMNS}
        metrics.update({
            "swh": np.array([5., 1., 2.]),
            "shww": np.array([3., 2., 1.]),
            "shts": np.array([4., np.nan, np.nan]),
        })
        summary, paired, partitioned = height_diagnostics(times, metrics)
        self.assertEqual(paired.sum(), 3)
        self.assertEqual(partitioned.sum(), 1)
        self.assertEqual(summary["energy_residual_m2_rmse"], 0.)
        self.assertEqual(
            summary["combined_below_wind_by_more_than_0_01m_hours"], 1
        )

    def test_gamma_recovered_from_synthetic_spectrum(self):
        frequency = np.linspace(.03, .5, 160)
        peak_frequency = .12
        sigma = np.where(frequency <= peak_frequency, .07, .09)
        enhancement = np.exp(
            -.5 * ((frequency / peak_frequency - 1) / sigma) ** 2
        )
        density = frequency ** -5
        density *= np.exp(-1.25 * (peak_frequency / frequency) ** 4)
        density *= 3.3 ** enhancement
        fit = fit_jonswap_gamma(frequency, density)
        self.assertAlmostEqual(fit["gamma"], 3.3, delta=.15)
        self.assertAlmostEqual(fit["tp_s"], 1 / peak_frequency, delta=.1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "spectra.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "spectrum_id", "frequency_hz", "psd_m2_per_hz",
                ])
                writer.writerows(
                    ["synthetic", value, sample]
                    for value, sample in zip(frequency, density)
                )
            export_spectral_fits(source, root)
            plot_jonswap_fits(root, root)
            self.assertTrue((root / "jonswap_fitting.png").is_file())
        with self.assertRaises(ValueError):
            fit_jonswap_gamma(frequency, -density)

    def test_synthetic_exports_and_all_diagnostic_plots(self):
        times = np.array([
            "2006-01-01T00", "2006-01-01T01", "2006-02-01T00",
        ], dtype="datetime64[ns]")
        metrics = {name: np.full(3, np.nan) for name in METRIC_COLUMNS}
        metrics.update({
            "swh": np.array([5., 2., 1.]),
            "shww": np.array([3., 1., .5]),
            "shts": np.array([4., np.nan, .8]),
            "pp1d": np.array([10., 8., 7.]),
            "mwd": np.array([359., 1., 270.]),
            "mdww": np.array([359., 1., 270.]),
            "u10": np.array([5., 3., 1.]),
            "v10": np.zeros(3),
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("wave_diagnostics.load_cache", return_value=(
                times, metrics, 59., -4., {},
            )):
                export_diagnostics(Path("synthetic.parquet"), root, 2006, 2006)
            with (root / "directional_wave_probability.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for source in ("swh", "shww"):
                self.assertAlmostEqual(sum(
                    float(row["probability"]) for row in rows
                    if row["height_source"] == source
                ), 1.)
            summary = json.loads(
                (root / "wave_quality_summary.json").read_text()
            )
            self.assertEqual(summary["three_partition_hours"], 2)
            for plot in (
                plot_height_cdf, plot_partition_diagnostics,
                plot_directional_probability, plot_variable_coverage,
                plot_wave_height_comparison,
            ):
                plot(root, root)
            self.assertEqual(len(list(root.glob("*.png"))), 5)
            self.assertTrue(all(
                path.stat().st_size > 1000 for path in root.glob("*.png")
            ))


if __name__ == "__main__":
    unittest.main()
