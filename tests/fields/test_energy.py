"""Tests for energy module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array, cos, sin
from numpy.random import rand
from numpy.testing import assert_array_equal

from fargonaut.fields.energy import Energy

TEMPDIR = tempfile.gettempdir()
GASENERGY1_FILE_NAME = TEMPDIR + "/gasenergy1.dat"
GASENERGY1 = rand(6)


class TestEnergy(unittest.TestCase):
    """Tests for Energy class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock output fixture and a temporary gas energy file."""
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

        assert_array_equal(self.energy._xdata, xdata_expected)
        assert_array_equal(self.energy._ydata, ydata_expected)
        assert_array_equal(self.energy._zdata, zdata_expected)

    def test_get_2D_cartesian_plot_data(self) -> None:
        """Test Energy's _get_2D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.energy._get_2D_cartesian_plot_data("polar", "xy", 0)
        with self.assertRaises(ValueError):
            self.energy._get_2D_cartesian_plot_data("invalid", "xy", 0)

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
        C_expected = self.energy.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$e$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cartesian_plot_data(
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
        C_expected = self.energy.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cartesian_plot_data(
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
        C_expected = self.energy.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cartesian_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_cylindrical_plot_data(self) -> None:
        """Test Energy's _get_2D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.energy._get_2D_cylindrical_plot_data("invalid", "xy", 0)

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
        C_expected = self.energy.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = "$e$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
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
        C_expected = self.energy.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
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
        C_expected = self.energy.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
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
        C_expected = self.energy.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$e$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
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
        C_expected = self.energy.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
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
        C_expected = self.energy.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_cylindrical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_2D_spherical_plot_data(self) -> None:
        """Test Energy's _get_2D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.energy._get_2D_spherical_plot_data("invalid", "xy", 0)

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
        C_expected = self.energy.data[:, :, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$r$"
        clabel_expected = "$e$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
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
        C_expected = self.energy.data[:, 0, :]
        ylabel_expected = r"$\theta$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
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
        C_expected = self.energy.data[0, :, :]
        xlabel_expected = "$r$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
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
        C_expected = self.energy.data[:, :, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$y$"
        clabel_expected = "$e$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
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
        C_expected = self.energy.data[:, 0, :]
        ylabel_expected = "$z$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
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
        C_expected = self.energy.data[0, :, :]
        xlabel_expected = "$y$"
        X, Y, C, xlabel, ylabel, clabel = self.energy._get_2D_spherical_plot_data(
            "cartesian", "yz", 0
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        assert_array_equal(C, C_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)
        self.assertEqual(clabel, clabel_expected)

    def test_get_1D_cartesian_plot_data(self) -> None:
        """Test Energy's _get_1D_cartesian_plot_data method."""
        with self.assertRaises(NotImplementedError):
            self.energy._get_1D_cartesian_plot_data("polar", "x", (0, 0))
        with self.assertRaises(ValueError):
            self.energy._get_1D_cartesian_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.energy.data[:, 0, 0]
        xlabel_expected = "$x$"
        ylabel_expected = "$e$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cartesian_plot_data(
            "cartesian", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.energy.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cartesian_plot_data(
            "cartesian", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.energy.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cartesian_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_cylindrical_plot_data(self) -> None:
        """Test Energy's _get_1D_cylindrical_plot_data method."""
        with self.assertRaises(ValueError):
            self.energy._get_1D_cylindrical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.energy.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$e$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.energy.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.energy.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
            "polar", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        phi_expected = array([-0.785, 0.785])
        r_expected = array([4.5, 4.5])
        X_expected = r_expected * cos(phi_expected)
        Y_expected = self.energy.data[:, 0, 0]
        xlabel_expected = "$x$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
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
        Y_expected = self.energy.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
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
        Y_expected = self.energy.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.energy._get_1D_cylindrical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_get_1D_spherical_plot_data(self) -> None:
        """Test Energy's _get_1D_spherical_plot_data method."""
        with self.assertRaises(ValueError):
            self.energy._get_1D_spherical_plot_data("invalid", "x", (0, 0))

        X_expected = array([-0.785, 0.785])
        Y_expected = self.energy.data[:, 0, 0]
        xlabel_expected = r"$\phi$"
        ylabel_expected = "$e$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
            "polar", "x", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([4.5, 5.5, 6.5])
        Y_expected = self.energy.data[0, :, 0]
        xlabel_expected = "$r$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
            "polar", "y", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

        X_expected = array([0.0])
        Y_expected = self.energy.data[0, 0, :]
        xlabel_expected = r"$\theta$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
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
        Y_expected = self.energy.data[:, 0, 0]
        xlabel_expected = "$x$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
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
        Y_expected = self.energy.data[0, :, 0]
        xlabel_expected = "$y$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
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
        Y_expected = self.energy.data[0, 0, :]
        xlabel_expected = "$z$"
        X, Y, xlabel, ylabel = self.energy._get_1D_spherical_plot_data(
            "cartesian", "z", (0, 0)
        )
        assert_array_equal(X, X_expected)
        assert_array_equal(Y, Y_expected)
        self.assertEqual(xlabel, xlabel_expected)
        self.assertEqual(ylabel, ylabel_expected)

    def test_x(self) -> None:
        """Test Field's x property."""
        assert_array_equal(self.energy.x, array([-0.785, 0.785]))

    def test_y(self) -> None:
        """Test Field's y property."""
        assert_array_equal(self.energy.y, array([4.5, 5.5, 6.5]))

    def test_z(self) -> None:
        """Test Field's z property."""
        assert_array_equal(self.energy.z, array([0.0]))

    def test_raw(self) -> None:
        """Test Field's raw property."""
        assert_array_equal(self.energy.raw, GASENERGY1)

    def test_data(self) -> None:
        """Test Field's data property."""
        expected = GASENERGY1.reshape((2, 3, 1), order="F")
        assert_array_equal(self.energy.data, expected)
