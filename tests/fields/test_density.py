"""Tests for density module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array, cos, sin
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.density import Density

TEMPDIR = tempfile.gettempdir()
GASDENS1_FILE_NAME = TEMPDIR + "/gasdens1.dat"
GASDENS1 = rand(6)


class TestDensity(unittest.TestCase):
    """Tests for Density class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and a temporary gas density file."""
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

        cls.gasdens1_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w+b", suffix=".dat"
        )
        GASDENS1.tofile(cls.gasdens1_file)
        cls.gasdens1_file.close()
        os.rename(cls.gasdens1_file.name, GASDENS1_FILE_NAME)
        cls.output = output

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(GASDENS1_FILE_NAME)

    def setUp(self) -> None:
        """Create density field fixture."""
        self.density = Density(self.output, 1)

    def tearDown(self) -> None:
        """Destroy density field fixture."""
        del self.density

    def test_init(self) -> None:
        """Test Density's __init__ method."""
        self.assertEqual(self.density._output, self.output)
        assert_array_equal(self.density._raw, GASDENS1)

    def test_process_domains(self) -> None:
        """Test Density's _process_domains method."""
        xdata_expected = (
            0.5
            * (self.output._xdomain[1:] + self.output._xdomain[:-1])[
                self.output.nghx : -self.output.nghx
            ]
        )
        ydata_expected = (
            0.5
            * (self.output._ydomain[1:] + self.output._ydomain[:-1])[
                self.output.nghy : -self.output.nghy
            ]
        )
        zdata_expected = (
            0.5
            * (self.output._zdomain[1:] + self.output._zdomain[:-1])[
                self.output.nghz : -self.output.nghz
            ]
        )

        assert_array_equal(self.density._xdata, xdata_expected)
        assert_array_equal(self.density._ydata, ydata_expected)
        assert_array_equal(self.density._zdata, zdata_expected)

    def test_get_2D_cartesian_plot_data(self) -> None:
        """Test Density's _get_2D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.density._get_2D_cartesian_plot_data("polar", "xy", 0)
        with self.assertRaises(ValueError):
            self.density._get_2D_cartesian_plot_data("invalid", "xy", 0)

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
        C_expected = self.density.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cartesian_plot_data(
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
        C_expected = self.density.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cartesian_plot_data(
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
        C_expected = self.density.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cartesian_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_cylindrical_plot_data(self) -> None:
        """Test Density's _get_2D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.density._get_2D_cylindrical_plot_data("invalid", "xy", 0)

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
        C_expected = self.density.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
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
        C_expected = self.density.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
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
        C_expected = self.density.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
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
        C_expected = self.density.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
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
        C_expected = self.density.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
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
        C_expected = self.density.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_cylindrical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_spherical_plot_data(self) -> None:
        """Test Density's _get_2D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.density._get_2D_spherical_plot_data("invalid", "xy", 0)

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
        C_expected = self.density.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
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
        C_expected = self.density.data[:, 0, :]
        ylabel_expected = r"$\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
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
        C_expected = self.density.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
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
        theta_expected = array(
            [
                [-0.25, -0.25, -0.25, -0.25],
                [-0.25, -0.25, -0.25, -0.25],
                [-0.25, -0.25, -0.25, -0.25],
            ],
        )
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = r_expected * sin(phi_expected) * sin(theta_expected)
        C_expected = self.density.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
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
        C_expected = self.density.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
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
        C_expected = self.density.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.density._get_2D_spherical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_1D_cartesian_plot_data(self) -> None:
        """Test Density's _get_1D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.density._get_1D_cartesian_plot_data("polar", "x", (0, 0))
        with self.assertRaises(ValueError):
            self.density._get_1D_cartesian_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.density.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, xlabel, ylabel = self.density._get_1D_cartesian_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.density.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.density._get_1D_cartesian_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.density.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.density._get_1D_cartesian_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_cylindrical_plot_data(self) -> None:
        """Test Density's _get_1D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.density._get_1D_cylindrical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.density.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.density.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.density.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785, 0.785])
        r_expected = array([4.5, 4.5])
        X_expected = r_expected * cos(phi_expected)
        Y_expected = self.density.data[:, 0, 0]
        xlabel_expected = "$x$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785, -0.785, -0.785])
        r_expected = array([4.5, 5.5, 6.5])
        z_expected = array([0.0, 0.0, 0.0])
        X_expected = r_expected * sin(phi_expected)
        Y_expected = self.density.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
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
        Y_expected = self.density.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.density._get_1D_cylindrical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_spherical_plot_data(self) -> None:
        """Test Density's _get_1D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.density._get_1D_spherical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.density.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = r"$\mathit{\Sigma}_\mathrm{g}$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.density.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.density.data[0, 0, :]
        xlabel_expected = r"$\theta$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785, 0.785])
        r_expected = array([4.5, 4.5])
        theta_expected = array([0.0, 0.0])
        X_expected = r_expected * cos(phi_expected) * sin(theta_expected)
        Y_expected = self.density.data[:, 0, 0]
        xlabel_expected = "$x$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
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
        Y_expected = self.density.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
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
        Y_expected = self.density.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.density._get_1D_spherical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    @unittest.mock.patch("fargonaut.fields.density.Density._get_1D_spherical_plot_data")
    @unittest.mock.patch("fargonaut.fields.density.Density._get_1D_cartesian_plot_data")
    @unittest.mock.patch(
        "fargonaut.fields.density.Density._get_1D_cylindrical_plot_data"
    )
    @unittest.mock.patch("fargonaut.fields.density.Density._get_2D_spherical_plot_data")
    @unittest.mock.patch("fargonaut.fields.density.Density._get_2D_cartesian_plot_data")
    @unittest.mock.patch(
        "fargonaut.fields.density.Density._get_2D_cylindrical_plot_data"
    )
    def test_plot(
        self,
        cylindrical_2D_mock,
        cartesian_2D_mock,
        spherical_2D_mock,
        cylindrical_1D_mock,
        cartesian_1D_mock,
        spherical_1D_mock,
    ) -> None:
        """Test Field's plot method."""
        cylindrical_2D_mock.return_value = ("X", "Y", "C", "xlabel", "ylabel", "clabel")
        self.density._output.coordinate_system = "cylindrical"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.pcolormesh"),
            unittest.mock.patch("matplotlib.pyplot.colorbar"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="xy", idx=0)
        cylindrical_2D_mock.assert_called_once_with("polar", "xy", 0)

        cartesian_2D_mock.return_value = ("X", "Y", "C", "xlabel", "ylabel", "clabel")
        self.density._output.coordinate_system = "cartesian"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.pcolormesh"),
            unittest.mock.patch("matplotlib.pyplot.colorbar"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="xy", idx=0)
        cartesian_2D_mock.assert_called_once_with("polar", "xy", 0)

        spherical_2D_mock.return_value = ("X", "Y", "C", "xlabel", "ylabel", "clabel")
        self.density._output.coordinate_system = "spherical"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.pcolormesh"),
            unittest.mock.patch("matplotlib.pyplot.colorbar"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="xy", idx=0)
        spherical_2D_mock.assert_called_once_with("polar", "xy", 0)

        cylindrical_1D_mock.return_value = ("X", "Y", "xlabel", "ylabel")
        self.density._output.coordinate_system = "cylindrical"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.plot"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="x", idx=(0, 0))
        cylindrical_1D_mock.assert_called_once_with("polar", "x", (0, 0))

        cartesian_1D_mock.return_value = ("X", "Y", "xlabel", "ylabel")
        self.density._output.coordinate_system = "cartesian"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.plot"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="x", idx=(0, 0))
        cartesian_1D_mock.assert_called_once_with("polar", "x", (0, 0))

        spherical_1D_mock.return_value = ("X", "Y", "xlabel", "ylabel")
        self.density._output.coordinate_system = "spherical"
        with (
            unittest.mock.patch("matplotlib.pyplot.figure"),
            unittest.mock.patch("matplotlib.pyplot.plot"),
            unittest.mock.patch("matplotlib.pyplot.show"),
        ):
            self.density.plot(csys="polar", dims="x", idx=(0, 0))
        spherical_1D_mock.assert_called_once_with("polar", "x", (0, 0))

        self.density._output.coordinate_system = "something else"
        with self.assertRaises(NotImplementedError):
            self.density.plot(csys="polar", dims="ij", idx=0)

        with self.assertRaises(NotImplementedError):
            self.density.plot(csys="polar", dims="k", idx=(0, 0))

    def test_x(self) -> None:
        """Test Field's x property."""
        assert_array_equal(self.density.x, array([-0.785, 0.785]))

    def test_y(self) -> None:
        """Test Field's y property."""
        assert_array_equal(self.density.y, array([4.5, 5.5, 6.5]))

    def test_z(self) -> None:
        """Test Field's z property."""
        assert_array_equal(self.density.z, array([0.0]))

    def test_raw(self) -> None:
        """Test Field's raw property."""
        assert_array_equal(self.density.raw, GASDENS1)

    def test_data(self) -> None:
        """Test Field's data property."""
        expected = GASDENS1.reshape((2, 3, 1), order="F")
        assert_array_equal(self.density.data, expected)

    def test_check_valid_for_arithmetic(self) -> None:
        """Test Field's _check_valid_for_arithmetic method."""
        field2 = unittest.mock.Mock(spec=self.density)
        field2._xdata = self.density._xdata
        field2._ydata = self.density._ydata
        field2._zdata = self.density._zdata
        self.assertEqual(
            None,
            self.density._check_valid_for_arithmetic(field2, "arithmetic_operation"),
        )
        with self.assertRaises(Exception):
            self.density._check_valid_for_arithmetic(
                "not_a_Field", "arithmetic_operation"
            )

        field2._xdata = self.density._xdata - 3.14 / 4
        with self.assertRaises(Exception):
            self.density._check_valid_for_arithmetic(field2, "arithmetic_operation")

    def test_add(self) -> None:
        """Test Field's __add__ method."""
        field2 = unittest.mock.Mock(spec=self.density)
        field2._xdata = self.density._xdata
        field2._ydata = self.density._ydata
        field2._zdata = self.density._zdata
        field2._raw = rand(6)
        field2._data = field2._raw.reshape((2, 3, 1), order="F")
        result = self.density + field2
        assert_array_equal(self.density._raw + field2._raw, result._raw)

    def test_sub(self) -> None:
        """Test Field's __sub__ method."""
        field2 = unittest.mock.Mock(spec=self.density)
        field2._xdata = self.density._xdata
        field2._ydata = self.density._ydata
        field2._zdata = self.density._zdata
        field2._raw = rand(6)
        field2._data = field2._raw.reshape((2, 3, 1), order="F")
        result = self.density - field2
        assert_array_equal(self.density._raw - field2._raw, result._raw)

    def test_mul(self) -> None:
        """Test Field's __mul__ method."""
        field2 = unittest.mock.Mock(spec=self.density)
        field2._xdata = self.density._xdata
        field2._ydata = self.density._ydata
        field2._zdata = self.density._zdata
        field2._raw = rand(6)
        field2._data = field2._raw.reshape((2, 3, 1), order="F")
        result = self.density * field2
        assert_array_equal(self.density._raw * field2._raw, result._raw)

    def test_truediv(self) -> None:
        """Test Field's __truediv__ method."""
        field2 = unittest.mock.Mock(spec=self.density)
        field2._xdata = self.density._xdata
        field2._ydata = self.density._ydata
        field2._zdata = self.density._zdata
        field2._raw = rand(6)
        field2._data = field2._raw.reshape((2, 3, 1), order="F")
        result = self.density / field2
        assert_array_equal(self.density._raw / field2._raw, result._raw)

    def test_pow(self) -> None:
        """Test Field's __pow__ method."""
        result = self.density**2
        assert_array_equal(self.density._raw**2, result._raw)
