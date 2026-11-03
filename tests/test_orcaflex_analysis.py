"""Focused tests for the OrcaFlex comparison, independent of RAFT solver imports."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("orcaflex_analysis", ROOT / "OrcaFlex_Results/analyze_orcaflex.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class TestOrcaFlexAnalysis(unittest.TestCase):
    def test_harmonic_amplitude_and_acceleration(self):
        time = np.arange(6001) * 0.1
        frequency = 1 / 9.875
        amplitudes = np.arange(1, 7) * 0.01
        values = 2 + np.cos(2 * np.pi * frequency * time[:, None] + 0.7) * amplitudes
        fitted, residual = analysis.harmonic_fit(time, values, frequency)
        np.testing.assert_allclose(fitted, amplitudes, rtol=1e-12)
        np.testing.assert_allclose(residual, 0, atol=1e-13)
        angular_frequency = 2 * np.pi * frequency
        fitted_acceleration, _ = analysis.harmonic_fit(time, -(values - 2) * angular_frequency ** 2, frequency)
        np.testing.assert_allclose(fitted_acceleration, amplitudes * angular_frequency ** 2, rtol=1e-12)

    def test_psd_energy_and_units(self):
        time = np.arange(6000) * 0.1
        values = np.tile(np.cos(2 * np.pi * 0.1 * time[:, None]), (1, 6))
        frequency, density, _, _ = analysis.spectra(time, values, 200)
        np.testing.assert_allclose(density.sum(axis=0) * np.diff(frequency)[0], 0.5, rtol=1e-10)
        omega = frequency * 2 * np.pi
        density_per_radian = density / (2 * np.pi)
        np.testing.assert_allclose(density_per_radian.sum(axis=0) * np.diff(omega)[0], 0.5, rtol=1e-10)

    def test_actual_exports(self):
        time, channels = analysis.read_time_history(ROOT / "OrcaFlex_Results/time_response.csv")
        self.assertAlmostEqual(time[-1] - time[0], 600)
        self.assertEqual(len(time), 6001)
        self.assertIn("GX acceleration", channels)
        sections = analysis.read_raos(ROOT / "OrcaFlex_Results/RAO_response.csv")
        motion = analysis.interpolate_rao(sections["motion"], 1 / 9.875)
        acceleration = analysis.interpolate_rao(sections["acceleration"], 1 / 9.875)
        np.testing.assert_allclose(np.abs(acceleration), np.abs(motion) * (2 * np.pi / 9.875) ** 2, rtol=1e-9)
        with self.assertRaises(ValueError):
            analysis.interpolate_rao(sections["motion"], 100)


if __name__ == "__main__":
    unittest.main()