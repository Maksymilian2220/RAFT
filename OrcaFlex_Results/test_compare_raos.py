import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compare_raos


class ComparisonPlotTests(unittest.TestCase):
    def test_plot_uses_original_frequency_grids(self):
        raft_frequencies = np.array([0.01, 0.02, 0.03])
        raft_values = {
            dof: np.array([1 + 0j, 4 + 0j, 1 + 0j])
            for dof in compare_raos.RAFT_DOFS
        }
        orcaflex = {
            dof: {
                "frequencies_hz": np.array([0.015, 0.025]),
                "amplitudes": np.array([2.0, 3.0]),
                "amplitude_units": compare_raos.DOF_UNITS[dof],
            }
            for dof in compare_raos.RAFT_DOFS
        }
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(compare_raos.plt, "close"):
                compare_raos._plot_comparison(
                    Path(directory) / "comparison.png", orcaflex,
                    raft_frequencies, raft_values,
                )
            figure = compare_raos.plt.gcf()
            try:
                for axis, dof in zip(figure.axes, compare_raos.RAFT_DOFS):
                    np.testing.assert_array_equal(
                        axis.lines[0].get_xdata(), raft_frequencies
                    )
                    np.testing.assert_array_equal(
                        axis.lines[0].get_ydata(), np.abs(raft_values[dof])
                    )
                    np.testing.assert_array_equal(
                        axis.lines[1].get_xdata(),
                        orcaflex[dof]["frequencies_hz"],
                    )
            finally:
                compare_raos.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
