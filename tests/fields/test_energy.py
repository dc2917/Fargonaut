"""Tests for energy module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.energy import Energy

TEMPDIR = tempfile.gettempdir()
GASENERGY1_FILE_NAME = TEMPDIR + "/gasenergy1.dat"
GASENERGY1 = rand(12)


class TestEnergy(unittest.TestCase):
    """Tests for Energy class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and a temporary gas energy file."""
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

        cls.gasenergy1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        GASENERGY1.tofile(cls.gasenergy1_file)
        cls.gasenergy1_file.close()
        os.rename(cls.gasenergy1_file.name, GASENERGY1_FILE_NAME)
        cls.output = output

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(GASENERGY1_FILE_NAME)

    def setUp(self) -> None:
        """Create energy field fixture."""
        self.energy = Energy(self.output, 1)

    def tearDown(self) -> None:
        """Destroy energy field fixture."""
        del self.energy

    def test_init(self) -> None:
        """Test Energy's __init__ method."""
        self.assertEqual(self.energy._output, self.output)
        assert_array_equal(self.energy._raw, GASENERGY1)

    def test_process_domains(self) -> None:
        """Test Energy's _process_domains method."""
        xdata_expected = 0.5 * (self.output._xdomain[1:] + self.output._xdomain[:-1])
        ydata_expected = (
            0.5
            * (self.output._ydomain[1:] + self.output._ydomain[:-1])[
                self.output.nghy : -self.output.nghy
            ]
        )
        zdata_expected = 0.5 * (self.output._zdomain[1:] + self.output._zdomain[:-1])

        assert_array_equal(self.energy._xdata, xdata_expected)
        assert_array_equal(self.energy._ydata, ydata_expected)
        assert_array_equal(self.energy._zdata, zdata_expected)

    def test_process_data(self) -> None:
        """Test Energy's _process_data method."""
        self.assertTupleEqual(self.energy._data.shape, (3, 4))
