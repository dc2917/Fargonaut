"""Tests for velocity module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.velocity import Velocity

TEMPDIR = tempfile.gettempdir()
GASVX1_FILE_NAME = TEMPDIR + "/gasvx1.dat"
GASVX1 = rand(12)
GASVY1_FILE_NAME = TEMPDIR + "/gasvy1.dat"
GASVY1 = rand(12)


class TestVelocity(unittest.TestCase):
    """Tests for Velocity class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and a temporary gas velocity file."""
        output = unittest.mock.Mock()
        output._directory = Path(TEMPDIR)
        output._xdomain = array([-3.14, -1.57, 0.0, 1.57, 3.14])
        output._ydomain = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        output._zdomain = array([0.0, 0.0])
        output.nx = 4
        output.ny = 3
        output.nz = 1
        output.nghx = 0
        output.nghy = 3
        output.nghz = 0

        cls.gasvx1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        GASVX1.tofile(cls.gasvx1_file)
        cls.gasvx1_file.close()
        os.rename(cls.gasvx1_file.name, GASVX1_FILE_NAME)

        cls.gasvy1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        GASVY1.tofile(cls.gasvy1_file)
        cls.gasvy1_file.close()
        os.rename(cls.gasvy1_file.name, GASVY1_FILE_NAME)
        cls.output = output

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(GASVX1_FILE_NAME)
        os.remove(GASVY1_FILE_NAME)

    def setUp(self) -> None:
        """Create velocity field fixture."""
        self.velocity_x = Velocity(self.output, "x", 1)
        self.velocity_y = Velocity(self.output, "y", 1)

    def tearDown(self) -> None:
        """Destroy velocity field fixture."""
        del self.velocity_x
        del self.velocity_y

    def test_init(self) -> None:
        """Test Velocity's __init__ method."""
        self.assertEqual(self.velocity_x._output, self.output)
        assert_array_equal(self.velocity_x._raw, GASVX1)
        self.assertEqual(self.velocity_y._output, self.output)
        assert_array_equal(self.velocity_y._raw, GASVY1)

    def test_process_domains(self) -> None:
        """Test Velocity's _process_domains method."""
        xdata_expected_x = self.output._xdomain[:-1]
        ydata_expected_x = (
            0.5
            * (self.output._ydomain[1:] + self.output._ydomain[:-1])[
                self.output.nghy : -self.output.nghy
            ]
        )
        xdata_expected_y = 0.5 * (self.output._xdomain[1:] + self.output._xdomain[:-1])
        ydata_expected_y = self.output._ydomain[:-1][
            self.output.nghy : -self.output.nghy
        ]
        zdata_expected = 0.5 * (self.output._zdomain[1:] + self.output._zdomain[:-1])

        assert_array_equal(self.velocity_x._xdata, xdata_expected_x)
        assert_array_equal(self.velocity_x._ydata, ydata_expected_x)
        assert_array_equal(self.velocity_x._zdata, zdata_expected)
        assert_array_equal(self.velocity_y._xdata, xdata_expected_y)
        assert_array_equal(self.velocity_y._ydata, ydata_expected_y)
        assert_array_equal(self.velocity_y._zdata, zdata_expected)

    def test_process_data(self) -> None:
        """Test Velocity's _process_data method."""
        self.assertTupleEqual(self.velocity_x._data.shape, (3, 4))
        self.assertTupleEqual(self.velocity_y._data.shape, (3, 4))
