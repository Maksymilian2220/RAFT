"""Offline tests for staged raw ERA5 spectral retrieval."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from download_era5_spectra import retrieve_spectra, spectral_request


class SpectraTests(unittest.TestCase):
    def test_request_contains_all_bins_and_correct_calendar(self):
        request = spectral_request(2024, 2)
        self.assertEqual(request["date"], "2024-02-01/to/2024-02-29")
        self.assertEqual(request["param"], "140251")
        self.assertEqual(request["frequency"], "1/to/30")
        self.assertEqual(request["direction"], "1/to/24")
        self.assertEqual(len(request["time"].split("/")), 24)
        self.assertEqual(request["area"], "59/-4/59/-4")
        self.assertTrue(spectral_request(2025, 2)["date"].endswith("02-28"))

    def test_dry_run_never_creates_client_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            with patch("download_era5_spectra.cdsapi.Client") as client:
                with contextlib.redirect_stdout(io.StringIO()) as text:
                    retrieve_spectra(2025, 2025, output, dry_run=True)
                self.assertEqual(text.getvalue().count("Would retrieve"), 12)
                client.assert_not_called()
            self.assertFalse(output.exists())

    def test_retrieve_and_resume_with_mock_grib_envelopes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            client = Mock()
            client.retrieve.side_effect = (
                lambda dataset, request, target:
                Path(target).write_bytes(b"GRIBpayload7777")
            )
            with patch("download_era5_spectra.cdsapi.Client",
                       return_value=client):
                with contextlib.redirect_stdout(io.StringIO()):
                    retrieve_spectra(2025, 2025, output)
                    retrieve_spectra(2025, 2025, output)
            self.assertEqual(client.retrieve.call_count, 12)
            metadata = json.loads(
                (output / "era5_spectra_2025_01.request.json").read_text()
            )
            self.assertEqual(metadata["dataset"], "reanalysis-era5-complete")


if __name__ == "__main__":
    unittest.main()
