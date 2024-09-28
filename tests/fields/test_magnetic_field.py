"""Tests for magnetic_field module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array, cos, sin
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.magnetic_field import MagneticField

TEMPDIR = tempfile.gettempdir()
BX1_FILE_NAME = TEMPDIR + "/bx1.dat"
BX1 = rand(6)
BY1_FILE_NAME = TEMPDIR + "/by1.dat"
BY1 = rand(6)
BZ1_FILE_NAME = TEMPDIR + "/bz1.dat"
BZ1 = rand(6)


class TestMagneticField(unittest.TestCase):
    """Tests for MagneticField class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and temporary magnetic field files."""
        output = unittest.mock.Mock()
        output._directory = Path(TEMPDIR)
        output._xdomain = array([-3.14, -1.57, 0.0, 1.57, 3.14])
        output._ydomain = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        output._zdomain = array([-0.75, -0.25, 0.25, 0.75])
        output.nx = 2
        output.ny = 3
        output.nz = 1
        output.nghx = 1
        output.nghy = 3
        output.nghz = 1

        cls.bx1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        BX1.tofile(cls.bx1_file)
        cls.bx1_file.close()
        os.rename(cls.bx1_file.name, BX1_FILE_NAME)

        cls.by1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        BY1.tofile(cls.by1_file)
        cls.by1_file.close()
        os.rename(cls.by1_file.name, BY1_FILE_NAME)

        cls.bz1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        BZ1.tofile(cls.bz1_file)
        cls.bz1_file.close()
        os.rename(cls.bz1_file.name, BZ1_FILE_NAME)

        cls.output = output

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(BX1_FILE_NAME)
        os.remove(BY1_FILE_NAME)
        os.remove(BZ1_FILE_NAME)

    def setUp(self) -> None:
        """Create magnetic field fixture."""
        self.b_x = MagneticField(self.output, "x", 1)
        self.b_y = MagneticField(self.output, "y", 1)
        self.b_z = MagneticField(self.output, "z", 1)

    def tearDown(self) -> None:
        """Destroy magnetic field fixture."""
        del self.b_x
        del self.b_y
        del self.b_z

    def test_init(self) -> None:
        """Test MagneticField's __init__ method."""
        self.assertEqual(self.b_x._output, self.output)
        assert_array_equal(self.b_x._raw, BX1)
        self.assertEqual(self.b_y._output, self.output)
        assert_array_equal(self.b_y._raw, BY1)
        self.assertEqual(self.b_z._output, self.output)
        assert_array_equal(self.b_z._raw, BZ1)

    def test_process_domains(self) -> None:
        """Test MagneticField's _process_domains method."""
        xdata_expected_x = self.output._xdomain[:-1][
            self.output.nghx : -self.output.nghx
        ]
        ydata_expected_x = (
            0.5
            * (self.output._ydomain[1:] + self.output._ydomain[:-1])[
                self.output.nghy : -self.output.nghy
            ]
        )
        zdata_expected_x = (
            0.5
            * (self.output._zdomain[1:] + self.output._zdomain[:-1])[
                self.output.nghz : -self.output.nghz
            ]
        )

        xdata_expected_y = (
            0.5
            * (self.output._xdomain[1:] + self.output._xdomain[:-1])[
                self.output.nghx : -self.output.nghx
            ]
        )
        ydata_expected_y = self.output._ydomain[:-1][
            self.output.nghy : -self.output.nghy
        ]
        zdata_expected_y = (
            0.5
            * (self.output._zdomain[1:] + self.output._zdomain[:-1])[
                self.output.nghz : -self.output.nghz
            ]
        )

        xdata_expected_z = (
            0.5
            * (self.output._xdomain[1:] + self.output._xdomain[:-1])[
                self.output.nghx : -self.output.nghx
            ]
        )
        ydata_expected_z = (
            0.5
            * (self.output._ydomain[1:] + self.output._ydomain[:-1])[
                self.output.nghy : -self.output.nghy
            ]
        )
        zdata_expected_z = self.output._zdomain[:-1][
            self.output.nghz : -self.output.nghz
        ]

        assert_array_equal(self.b_x._xdata, xdata_expected_x)
        assert_array_equal(self.b_x._ydata, ydata_expected_x)
        assert_array_equal(self.b_x._zdata, zdata_expected_x)
        assert_array_equal(self.b_y._xdata, xdata_expected_y)
        assert_array_equal(self.b_y._ydata, ydata_expected_y)
        assert_array_equal(self.b_y._zdata, zdata_expected_y)
        assert_array_equal(self.b_z._xdata, xdata_expected_z)
        assert_array_equal(self.b_z._ydata, ydata_expected_z)
        assert_array_equal(self.b_z._zdata, zdata_expected_z)

    def test_get_2D_cartesian_plot_data(self) -> None:
        """Test MagneticField's _get_2D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.b_x._get_2D_cartesian_plot_data("polar", "xy", 0)
        with self.assertRaises(ValueError):
            self.b_x._get_2D_cartesian_plot_data("invalid", "xy", 0)

        X_expected = array(
            [
                [-1.57, -1.57, -1.57, -1.57],
                [0.0, 0.0, 0.0, 0.0],
                [1.57, 1.57, 1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        C_expected = self.b_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$B_$x$$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cartesian_plot_data(
            "cartesian", "xy", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array(
            [
                [-1.57, -1.57],
                [0.0, 0.0],
                [1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        C_expected = self.b_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cartesian_plot_data(
            "cartesian", "xz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        Y_expected = array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]])
        C_expected = self.b_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cartesian_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_cylindrical_plot_data(self) -> None:
        """Test MagneticField's _get_2D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.b_x._get_2D_cylindrical_plot_data("invalid", "xy", 0)

        X_expected = array(
            [
                [-1.57, -1.57, -1.57, -1.57],
                [0.0, 0.0, 0.0, 0.0],
                [1.57, 1.57, 1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        C_expected = self.b_x.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$B_\phi$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "polar", "xy", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array(
            [
                [-1.57, -1.57],
                [0.0, 0.0],
                [1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        C_expected = self.b_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "polar", "xz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        Y_expected = array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]])
        C_expected = self.b_x.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "polar", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57, -1.57, -1.57],
                [0.0, 0.0, 0.0, 0.0],
                [1.57, 1.57, 1.57, 1.57],
            ],
        )
        r_expected = array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        X_expected = r_expected * cos(phi_expected)
        Y_expected = r_expected * sin(phi_expected)
        C_expected = self.b_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = r"$B_$x$$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "cartesian", "xy", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57],
                [0.0, 0.0],
                [1.57, 1.57],
            ],
        )
        r_expected = array(
            [
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ],
        )
        z_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        X_expected = r_expected * cos(phi_expected)
        Y_expected = z_expected
        C_expected = self.b_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "cartesian", "xz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57],
                [-1.57, -1.57],
                [-1.57, -1.57],
                [-1.57, -1.57],
            ],
        )
        r_expected = array(
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
        )
        z_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        X_expected = r_expected * sin(phi_expected)
        Y_expected = z_expected
        C_expected = self.b_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_cylindrical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_spherical_plot_data(self) -> None:
        """Test MagneticField's _get_2D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.b_x._get_2D_spherical_plot_data("invalid", "xy", 0)

        X_expected = array(
            [
                [-1.57, -1.57, -1.57, -1.57],
                [0.0, 0.0, 0.0, 0.0],
                [1.57, 1.57, 1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        C_expected = self.b_x.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$B_\phi$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "polar", "xy", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array(
            [
                [-1.57, -1.57],
                [0.0, 0.0],
                [1.57, 1.57],
            ],
        )
        Y_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        C_expected = self.b_x.data[:, 0, :]
        ylabel_expected = r"$\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "polar", "xz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        X_expected = array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])
        Y_expected = array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25]])
        C_expected = self.b_x.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "polar", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        clabel_expected = r"$B_\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.b_z._get_2D_spherical_plot_data(
            "polar", "xy", 0
        )
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57, -1.57, -1.57],
                [0.0, 0.0, 0.0, 0.0],
                [1.57, 1.57, 1.57, 1.57],
            ],
        )
        r_expected = array(
            [
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
        )
        theta_expected = array(
            [
                [-0.25, -0.25, -0.25, -0.25],
                [-0.25, -0.25, -0.25, -0.25],
                [-0.25, -0.25, -0.25, -0.25],
            ],
        )
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = r_expected * sin(phi_expected) * sin(theta_expected)
        C_expected = self.b_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$B_$x$$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "cartesian", "xy", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57],
                [0.0, 0.0],
                [1.57, 1.57],
            ],
        )
        r_expected = array(
            [
                [4.0, 4.0],
                [4.0, 4.0],
                [4.0, 4.0],
            ],
        )
        theta_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = r_expected * cos(theta_expected)
        C_expected = self.b_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "cartesian", "xz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        phi_expected = array(
            [
                [-1.57, -1.57],
                [-1.57, -1.57],
                [-1.57, -1.57],
                [-1.57, -1.57],
            ],
        )
        r_expected = array(
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
        )
        theta_expected = array(
            [
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
                [-0.25, 0.25],
            ],
        )
        X_expected = r_expected * sin(phi_expected) * sin(theta_expected)
        Y_expected = r_expected * cos(theta_expected)
        C_expected = self.b_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.b_x._get_2D_spherical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_1D_cartesian_plot_data(self) -> None:
        """Test MagneticField's _get_1D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.b_x._get_1D_cartesian_plot_data("polar", "x", (0, 0))
        with self.assertRaises(ValueError):
            self.b_x._get_1D_cartesian_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0])
        Y_expected = self.b_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = r"$B_$x$$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cartesian_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.b_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cartesian_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.b_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cartesian_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_cylindrical_plot_data(self) -> None:
        """Test MagneticField's _get_1D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.b_x._get_1D_cylindrical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0.0])
        Y_expected = self.b_x.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$B_\phi$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.b_x.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.b_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-1.57, 0.0])
        r_expected = array([4.5, 4.5])
        X_expected = r_expected * cos(phi_expected)
        Y_expected = self.b_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = r"$B_$x$$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-1.57, -1.57, -1.57])
        r_expected = array([4.5, 5.5, 6.5])
        z_expected = array([0.0, 0.0, 0.0])
        X_expected = r_expected * sin(phi_expected)
        Y_expected = self.b_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785])
        r_expected = array([4.5])
        z_expected = array([0])
        X_expected = z_expected
        Y_expected = self.b_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_cylindrical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_spherical_plot_data(self) -> None:
        """Test MagneticField's _get_1D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.b_x._get_1D_spherical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0.0])
        Y_expected = self.b_x.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$B_\phi$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.b_x.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.b_x.data[0, 0, :]
        xlabel_expected = r"$\theta$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        ylabel_expected = r"$B_\theta$"
        X, Y, xlabel, ylabel = self.b_z._get_1D_spherical_plot_data(
            "polar", "z", (0, 0)
        )
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-1.57, 0])
        r_expected = array([4.5, 4.5])
        theta_expected = array([0.0, 0.0])
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = self.b_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = r"$B_$x$$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785, -0.785, -0.785])
        r_expected = array([4.5, 5.5, 6.5])
        theta_expected = array([0.0, 0.0, 0.0])
        X_expected = r_expected * sin(phi_expected) * sin(theta_expected)
        Y_expected = self.b_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785])
        r_expected = array([4.5])
        theta_expected = array([0])
        X_expected = r_expected * cos(theta_expected)
        Y_expected = self.b_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.b_x._get_1D_spherical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_x(self) -> None:
        """Test Field's x property."""
        assert_array_equal(self.b_x.x, array([-1.57, 0.0]))
        assert_array_equal(self.b_y.x, array([-0.785, 0.785]))

    def test_y(self) -> None:
        """Test Field's y property."""
        assert_array_equal(self.b_x.y, array([4.5, 5.5, 6.5]))
        assert_array_equal(self.b_y.y, array([4.0, 5.0, 6.0]))

    def test_z(self) -> None:
        """Test Field's z property."""
        assert_array_equal(self.b_x.z, array([0.0]))
        assert_array_equal(self.b_z.z, array([-0.25]))

    def test_raw(self) -> None:
        """Test Field's raw property."""
        assert_array_equal(self.b_x.raw, BX1)
        assert_array_equal(self.b_y.raw, BY1)
        assert_array_equal(self.b_z.raw, BZ1)

    def test_data(self) -> None:
        """Test Field's data proper.ty."""
        expected_x = BX1.reshape((2, 3, 1), order="F")
        assert_array_equal(self.b_x.data, expected_x)
