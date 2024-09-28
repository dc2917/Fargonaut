"""Tests for field module."""

import unittest
import unittest.mock

from numpy import array
from numpy.testing import assert_array_equal

from fargonaut.field import Field


class TestField(unittest.TestCase):
    """Tests for Field class."""

    def setUp(self) -> None:
        """Create field fixture."""
        self.field = unittest.mock.Mock()
        self.field._output.nx = 1
        self.field._output.ny = 2
        self.field._output.nz = 3
        self.field._xdata = array([0.1, 0.2, 0.3])
        self.field._ydata = array([0.4, 0.5, 0.6])
        self.field._zdata = array([0.7, 0.8, 0.9])
        self.field._raw = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.field._process_data = Field._process_data

    def tearDown(self) -> None:
        """Destroy field fixture."""
        del self.field

    def test_process_data(self) -> None:
        """Test Field's _process_data method."""
        self.field._process_data(self.field)
        self.assertTupleEqual(self.field._data.shape, (1, 2, 3))
        assert_array_equal(
            self.field._data,
            array(
                [
                    [
                        [0.1, 0.3, 0.5],
                        [0.2, 0.4, 0.6],
                    ]
                ]
            ),
        )
