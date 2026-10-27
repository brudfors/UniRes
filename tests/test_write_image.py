"""Regression tests for the output dtype requested by _write_image."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import nibabel as nib
import numpy as np
import torch
from nitorch import io

from unires._core import _write_data
from unires._util import _read_label, _write_image
from unires.struct import _input, _output, settings


class TestWriteImage(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.data = torch.tensor(
            [0.25, 1.5, 32767, 32768, 40000, 50000, -32769, -40000],
            dtype=torch.float32,
        ).reshape(2, 2, 2)
        self.affine = np.diag([1.5, 2.0, 2.5, 1.0])
        self.affine[:3, 3] = [10.0, -20.0, 30.0]

    def template(self, dtype):
        path = self.directory / 'template.nii.gz'
        image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=dtype),
                               self.affine)
        nib.save(image, str(path))
        return str(path)

    def check_round_trip(self, expected_dtype, data=None, **kwargs):
        if data is None:
            data = self.data
        path = self.directory / 'sub-01_T1w.nii.gz'
        _write_image(data, str(path), mat=torch.from_numpy(self.affine),
                     **kwargs)
        if kwargs.get('bids'):
            path = self.directory / 'sub-01_space-unires_T1w.nii.gz'
        image = nib.load(str(path))
        self.assertEqual(image.get_data_dtype(), np.dtype(expected_dtype))
        np.testing.assert_array_equal(image.get_fdata(), data.numpy())
        np.testing.assert_array_equal(image.affine, self.affine)

    def test_default_dtype_without_template(self):
        self.check_round_trip('float32', data=self.data.double())

    def test_explicit_dtype_without_template(self):
        for dtype in ('float64', torch.float64):
            with self.subTest(dtype=dtype):
                self.check_round_trip('float64', dtype=dtype)

    def test_default_dtype_with_template(self):
        for dtype in ('int16', 'float64'):
            with self.subTest(template_dtype=dtype):
                self.check_round_trip('float32', file=self.template(dtype))

    def test_explicit_dtype_with_template(self):
        for mapped in (False, True):
            for dtype in ('float32', 'float64'):
                with self.subTest(mapped=mapped, dtype=dtype):
                    template = self.template('int16')
                    if mapped:
                        template = io.map(template)
                    self.check_round_trip(dtype, file=template, dtype=dtype)

    def test_explicit_integer_dtype_for_labels(self):
        labels = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        self.check_round_trip('int16', data=labels,
                              file=self.template('float32'), dtype='int16')

    def test_write_data_preserves_label_dtype(self):
        labels = np.arange(8, dtype=np.int16).reshape(2, 2, 2)
        template = self.directory / 'labels.nii.gz'
        nib.save(nib.Nifti1Image(labels, self.affine), str(template))
        sett = settings()
        sett.device = 'cpu'
        sett.dir_out = str(self.directory)
        sett.do_print = 0
        image = _input()
        image.dat = self.data
        image.dim = self.data.shape
        image.file = io.map(str(template))
        _read_label(image, str(template), sett)
        output = _output()
        output.dat = self.data.clone()
        output.mat = torch.from_numpy(self.affine)
        output.label = image.label[0]

        _, paths, _, label_path = _write_data([[image]], [output], sett)

        written_label = nib.load(label_path)
        self.assertEqual(written_label.get_data_dtype(), labels.dtype)
        np.testing.assert_array_equal(written_label.get_fdata(), labels)
        np.testing.assert_array_equal(written_label.affine, self.affine)
        written_image = nib.load(paths[0])
        self.assertEqual(written_image.get_data_dtype(), np.dtype('float32'))
        np.testing.assert_array_equal(written_image.get_fdata(), self.data.numpy())

    def test_bids_filename_with_requested_dtype(self):
        self.check_round_trip('float64', file=self.template('int16'),
                              dtype='float64', bids=True)


if __name__ == '__main__':
    unittest.main()
