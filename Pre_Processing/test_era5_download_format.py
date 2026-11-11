"""Offline tests for single-stream CDS download responses."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from zipfile import ZipFile

import netCDF4

from download_era5 import download_months, normalize_download


VARIABLES = ['significant_height_of_total_swell']


def write_wave(path):
    with netCDF4.Dataset(path, 'w') as dataset:
        dataset.createDimension('valid_time', 1)
        for name in ('valid_time', 'latitude', 'longitude', 'shts'):
            dataset.createVariable(name, 'f8', ('valid_time',))[:] = [1]


class DownloadFormatTests(unittest.TestCase):
    def test_single_netcdf_is_packaged_and_existing_zip_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'response.part'
            write_wave(path)
            normalize_download(path, VARIABLES)
            with ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), [
                    'data_stream-wave_stepType-instant.nc'
                ])
            previous = path.read_bytes()
            normalize_download(path, VARIABLES)
            self.assertEqual(path.read_bytes(), previous)

    def test_completed_netcdf_partial_is_recovered_without_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_wave(root / 'result.zip.part')
            client = Mock()
            download_months(client, 2006, [1], root, VARIABLES,
                            target_name='result.zip')
            client.retrieve.assert_not_called()
            self.assertTrue((root / 'result.zip').exists())

    def test_new_single_netcdf_download_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = Mock()
            client.retrieve.side_effect = (
                lambda dataset, request, target: write_wave(target)
            )
            download_months(client, 2006, [1], root, VARIABLES,
                            target_name='result.zip')
            with ZipFile(root / 'result.zip') as archive:
                self.assertEqual(len(archive.namelist()), 1)

    def test_invalid_response_is_not_destroyed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'response.part'
            path.write_bytes(b'not netcdf')
            with self.assertRaises(RuntimeError):
                normalize_download(path, VARIABLES)
            self.assertEqual(path.read_bytes(), b'not netcdf')


if __name__ == '__main__':
    unittest.main()
