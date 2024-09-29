"""Tests for velocity module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array, cos, sin
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.velocity import Velocity

TEMPDIR = tempfile.gettempdir()
GASVX1_FILE_NAME = TEMPDIR + "/gasvx1.dat"
GASVX1 = rand(6)
GASVY1_FILE_NAME = TEMPDIR + "/gasvy1.dat"
GASVY1 = rand(6)
GASVZ1_FILE_NAME = TEMPDIR + "/gasvz1.dat"
GASVZ1 = rand(6)


class TestVelocity(unittest.TestCase):
    """Tests for Velocity class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and temporary gas velocity files."""
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

        cls.gasvz1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        GASVZ1.tofile(cls.gasvz1_file)
        cls.gasvz1_file.close()
        os.rename(cls.gasvz1_file.name, GASVZ1_FILE_NAME)

        cls.output = output

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(GASVX1_FILE_NAME)
        os.remove(GASVY1_FILE_NAME)
        os.remove(GASVZ1_FILE_NAME)

    def setUp(self) -> None:
        """Create velocity field fixture."""
        self.velocity_x = Velocity(self.output, "x", 1)
        self.velocity_y = Velocity(self.output, "y", 1)
        self.velocity_z = Velocity(self.output, "z", 1)

    def tearDown(self) -> None:
        """Destroy velocity field fixture."""
        del self.velocity_x
        del self.velocity_y
        del self.velocity_z

    def test_init(self) -> None:
        """Test Velocity's __init__ method."""
        self.assertEqual(self.velocity_x._output, self.output)
        assert_array_equal(self.velocity_x._raw, GASVX1)
        self.assertEqual(self.velocity_y._output, self.output)
        assert_array_equal(self.velocity_y._raw, GASVY1)
        self.assertEqual(self.velocity_z._output, self.output)
        assert_array_equal(self.velocity_z._raw, GASVZ1)

    def test_process_domains(self) -> None:
        """Test Velocity's _process_domains method."""
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

        assert_array_equal(self.velocity_x._xdata, xdata_expected_x)
        assert_array_equal(self.velocity_x._ydata, ydata_expected_x)
        assert_array_equal(self.velocity_x._zdata, zdata_expected_x)
        assert_array_equal(self.velocity_y._xdata, xdata_expected_y)
        assert_array_equal(self.velocity_y._ydata, ydata_expected_y)
        assert_array_equal(self.velocity_y._zdata, zdata_expected_y)
        assert_array_equal(self.velocity_z._xdata, xdata_expected_z)
        assert_array_equal(self.velocity_z._ydata, ydata_expected_z)
        assert_array_equal(self.velocity_z._zdata, zdata_expected_z)

    def test_get_2D_cartesian_plot_data(self) -> None:
        """Test Velocity's _get_2D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.velocity_x._get_2D_cartesian_plot_data("polar", "xy", 0)
        with self.assertRaises(ValueError):
            self.velocity_x._get_2D_cartesian_plot_data("invalid", "xy", 0)

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
        C_expected = self.velocity_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$v_x$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cartesian_plot_data(
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
        C_expected = self.velocity_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cartesian_plot_data(
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
        C_expected = self.velocity_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cartesian_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_cylindrical_plot_data(self) -> None:
        """Test Velocity's _get_2D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.velocity_x._get_2D_cylindrical_plot_data("invalid", "xy", 0)

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
        C_expected = self.velocity_x.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$v_\phi$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
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
        C_expected = self.velocity_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
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
        C_expected = self.velocity_x.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
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
        C_expected = self.velocity_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$v_x$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
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
        C_expected = self.velocity_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
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
        C_expected = self.velocity_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_cylindrical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_spherical_plot_data(self) -> None:
        """Test Velocity's _get_2D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.velocity_x._get_2D_spherical_plot_data("invalid", "xy", 0)

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
        C_expected = self.velocity_x.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$v_\phi$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
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
        C_expected = self.velocity_x.data[:, 0, :]
        ylabel_expected = r"$\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
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
        C_expected = self.velocity_x.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
            "polar", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

        clabel_expected = r"$v_\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_z._get_2D_spherical_plot_data(
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
        C_expected = self.velocity_x.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$v_x$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
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
        C_expected = self.velocity_x.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
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
        C_expected = self.velocity_x.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.velocity_x._get_2D_spherical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_1D_cartesian_plot_data(self) -> None:
        """Test Velocity's _get_1D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.velocity_x._get_1D_cartesian_plot_data("polar", "x", (0, 0))
        with self.assertRaises(ValueError):
            self.velocity_x._get_1D_cartesian_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0])
        Y_expected = self.velocity_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$v_x$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cartesian_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.velocity_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cartesian_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.velocity_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cartesian_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_cylindrical_plot_data(self) -> None:
        """Test Velocity's _get_1D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.velocity_x._get_1D_cylindrical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0.0])
        Y_expected = self.velocity_x.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$v_\phi$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.velocity_x.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.velocity_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-1.57, 0.0])
        r_expected = array([4.5, 4.5])
        X_expected = r_expected * cos(phi_expected)
        Y_expected = self.velocity_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$v_x$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
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
        Y_expected = self.velocity_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
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
        Y_expected = self.velocity_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_cylindrical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_spherical_plot_data(self) -> None:
        """Test Velocity's _get_1D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.velocity_x._get_1D_spherical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-1.57, 0.0])
        Y_expected = self.velocity_x.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$v_\phi$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.velocity_x.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.velocity_x.data[0, 0, :]
        xlabel_expected = r"$\theta$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        ylabel_expected = r"$v_\theta$"
        X, Y, xlabel, ylabel = self.velocity_z._get_1D_spherical_plot_data(
            "polar", "z", (0, 0)
        )
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-1.57, 0])
        r_expected = array([4.5, 4.5])
        theta_expected = array([0.0, 0.0])
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = self.velocity_x.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$v_x$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
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
        Y_expected = self.velocity_x.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
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
        Y_expected = self.velocity_x.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.velocity_x._get_1D_spherical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_x(self) -> None:
        """Test Field's x property."""
        assert_array_equal(self.velocity_x.x, array([-1.57, 0.0]))
        assert_array_equal(self.velocity_y.x, array([-0.785, 0.785]))

    def test_y(self) -> None:
        """Test Field's y property."""
        assert_array_equal(self.velocity_x.y, array([4.5, 5.5, 6.5]))
        assert_array_equal(self.velocity_y.y, array([4.0, 5.0, 6.0]))

    def test_z(self) -> None:
        """Test Field's z property."""
        assert_array_equal(self.velocity_x.z, array([0.0]))
        assert_array_equal(self.velocity_z.z, array([-0.25]))

    def test_raw(self) -> None:
        """Test Field's raw property."""
        assert_array_equal(self.velocity_x.raw, GASVX1)
        assert_array_equal(self.velocity_y.raw, GASVY1)
        assert_array_equal(self.velocity_z.raw, GASVZ1)

    def test_data(self) -> None:
        """Test Field's data property."""
        expected_x = GASVX1.reshape((2, 3, 1), order="F")
        assert_array_equal(self.velocity_x.data, expected_x)
