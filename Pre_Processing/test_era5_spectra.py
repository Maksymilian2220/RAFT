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
        self.assertEqual(spectral_request(2024, 2, 29)["date"], "2024-02-29")
        with self.assertRaises(ValueError):
            spectral_request(2025, 2, 29)

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
            self.assertEqual(client.retrieve.call_count, 365)
            requests = [
                call.args[1] for call in client.retrieve.call_args_list
            ]
            self.assertEqual(
                len({request["date"] for request in requests}), 365
            )
            self.assertTrue(all(
                "/to/" not in request["date"] for request in requests
            ))
            self.assertEqual(
                (output / "era5_spectra_2025_01.grib").read_bytes(),
                b"GRIBpayload7777" * 31,
            )
            metadata = json.loads(
                (output / "era5_spectra_2025_01.request.json").read_text()
            )
            self.assertEqual(metadata["dataset"], "reanalysis-era5-complete")

    def test_resume_after_daily_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            client = Mock()

            def retrieve(dataset, request, target):
                if request["date"] == "2025-01-03":
                    raise RuntimeError("Download interrupted")
                Path(target).write_bytes(b"GRIBpayload7777")

            client.retrieve.side_effect = retrieve
            with patch("download_era5_spectra.cdsapi.Client",
                       return_value=client):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        retrieve_spectra(2025, 2025, output)
                    self.assertFalse(
                        (output / "era5_spectra_2025_01.grib").exists()
                    )
                    client.reset_mock()
                    client.retrieve.side_effect = (
                        lambda dataset, request, target:
                        Path(target).write_bytes(b"GRIBpayload7777")
                    )
                    retrieve_spectra(2025, 2025, output)
            self.assertEqual(client.retrieve.call_count, 363)
            self.assertEqual(
                client.retrieve.call_args_list[0].args[1]["date"], "2025-01-03"
            )


if __name__ == "__main__":
    unittest.main()
