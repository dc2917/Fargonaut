"""Tests for output module."""

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from numpy import array
from numpy.testing import assert_array_equal

from fargonaut.output import Output

TEMPDIR = tempfile.gettempdir()

DOMAIN_X_FILE_NAME = TEMPDIR + "/domain_x.dat"
DOMAIN_Y_FILE_NAME = TEMPDIR + "/domain_y.dat"
DOMAIN_Z_FILE_NAME = TEMPDIR + "/domain_z.dat"
SUMMARY0_FILE_NAME = TEMPDIR + "/summary0.dat"
VARIABLES_FILE_NAME = TEMPDIR + "/variables.par"

DOMAIN_X = "-3.14\n-1.57\n0.0\n1.57\n3.14\n"
DOMAIN_Y = "1.0\n2.0\n3.0\n"
DOMAIN_Z = "-1.0\n0.0\n1.0\n"
SUMMARY0 = (
    "stuff\n==\nCOMPILATION OPTION SECTION:\n==\n"
    "-DX -DY -DISOTHERMAL -DCYLINDRICAL\nmore stuff\n"
)
VARIABLES = "VAR1\tVAL1\nCOORDINATES\tcylindrical\nNX\t5\nNY\t3\nNZ\t3\nVARN\tVALN\n"


class TestOutput(unittest.TestCase):
    """Tests for Output class."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create temporary files for reading."""
        cls.domain_x_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.domain_x_file.write(DOMAIN_X)
        cls.domain_x_file.close()
        os.rename(cls.domain_x_file.name, DOMAIN_X_FILE_NAME)

        cls.domain_y_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.domain_y_file.write(DOMAIN_Y)
        cls.domain_y_file.close()
        os.rename(cls.domain_y_file.name, DOMAIN_Y_FILE_NAME)

        cls.domain_z_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.domain_z_file.write(DOMAIN_Z)
        cls.domain_z_file.close()
        os.rename(cls.domain_z_file.name, DOMAIN_Z_FILE_NAME)

        cls.summary0_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.summary0_file.write(SUMMARY0)
        cls.summary0_file.close()
        os.rename(cls.summary0_file.name, SUMMARY0_FILE_NAME)

        cls.variables_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".par"
        )
        cls.variables_file.write(VARIABLES)
        cls.variables_file.close()
        os.rename(cls.variables_file.name, VARIABLES_FILE_NAME)

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(DOMAIN_X_FILE_NAME)
        os.remove(DOMAIN_Y_FILE_NAME)
        os.remove(DOMAIN_Z_FILE_NAME)
        os.remove(SUMMARY0_FILE_NAME)
        os.remove(VARIABLES_FILE_NAME)

    def setUp(self) -> None:
        """Create output fixture."""
        self.output = Output(TEMPDIR)

    def tearDown(self) -> None:
        """Destroy output fixture."""
        del self.output

    def test_init(self) -> None:
        """Test Output's __init__ method."""
        self.assertEqual(self.output._directory, Path(TEMPDIR))

    def test_read_domains(self) -> None:
        """Test Output's _read_domains method."""
        xdomain = array([-3.14, -1.57, 0.0, 1.57, 3.14])
        ydomain = array([1.0, 2.0, 3.0])
        zdomain = array([-1.0, 0.0, 1.0])

        assert_array_equal(self.output._xdomain, xdomain)
        assert_array_equal(self.output._ydomain, ydomain)
        assert_array_equal(self.output._zdomain, zdomain)

    def test_read_opts(self) -> None:
        """Test Output's _read_opts method."""
        opts = ("X", "Y", "ISOTHERMAL", "CYLINDRICAL")
        self.assertTupleEqual(self.output._opts, opts)

    def test_read_vars(self) -> None:
        """Test Output's _read_vars method."""
        variables = {
            "VAR1": "VAL1",
            "COORDINATES": "cylindrical",
            "NX": "5",
            "NY": "3",
            "NZ": "3",
            "VARN": "VALN",
        }
        self.assertDictEqual(self.output._vars, variables)

    def test_read_units(self) -> None:
        """Test Output's _read_units method."""
        with self.assertRaises(NotImplementedError):
            self.output._read_units()

    def test_get_var(self) -> None:
        """Test Output's get_var method."""
        self.assertEqual(self.output.get_var("VAR1"), "VAL1")

    def test_get_opt(self) -> None:
        """Test Output's get_opt method."""
        self.assertEqual(self.output.get_opt("ISOTHERMAL"), True)
        self.assertEqual(self.output.get_opt("PARALLEL"), False)

        del self.output._opts
        with self.assertRaises(AttributeError):
            self.output.get_opt("PARALLEL")

    @unittest.mock.patch("fargonaut.output.Velocity")
    @unittest.mock.patch("fargonaut.output.MagneticField")
    @unittest.mock.patch("fargonaut.output.Energy")
    @unittest.mock.patch("fargonaut.output.Density")
    def test_get_field(
        self, density_mock, energy_mock, magnetic_field_mock, velocity_mock
    ) -> None:
        """Test Output's get_field method."""
        self.output.get_field("gasdens", 2)
        density_mock.assert_called_once_with(self.output, 2)
        self.output.get_field("gasenergy", 3)
        energy_mock.assert_called_once_with(self.output, 3)
        self.output.get_field("gasvx", 4)
        velocity_mock.assert_called_with(self.output, "x", 4)
        self.output.get_field("gasvy", 5)
        velocity_mock.assert_called_with(self.output, "y", 5)
        self.output.get_field("gasvz", 6)
        velocity_mock.assert_called_with(self.output, "z", 6)
        self.assertEqual(velocity_mock.call_count, 3)
        self.output.get_field("bx", 7)
        magnetic_field_mock.assert_called_with(self.output, "x", 7)
        self.output.get_field("by", 8)
        magnetic_field_mock.assert_called_with(self.output, "y", 8)
        self.output.get_field("bz", 9)
        magnetic_field_mock.assert_called_with(self.output, "z", 9)
        self.assertEqual(magnetic_field_mock.call_count, 3)

        with self.assertRaises(NotImplementedError):
            self.output.get_field("undefinedfield", 25)

    def test_coordinate_system(self) -> None:
        """Test Output's coordinate_system property."""
        self.assertEqual(self.output.coordinate_system, "cylindrical")

        del self.output._vars
        self.output._vars = {"COORDINATES": "cartesian"}
        self.assertEqual(self.output.coordinate_system, "cartesian")

        del self.output._vars
        self.output._vars = {"COORDINATES": "spherical"}
        self.assertEqual(self.output.coordinate_system, "spherical")

        del self.output._vars
        with self.assertRaises(Exception):
            self.output.coordinate_system

    def test_includes_ghosts(self) -> None:
        """Test Output's includes_ghosts property."""
        self.assertEqual(self.output.includes_ghosts, False)

        del self.output._opts
        with self.assertRaises(Exception):
            self.output.includes_ghosts

    def test_xdomain(self) -> None:
        """Test Output's xdomain property."""
        xdomain = array([-3.14, -1.57, 0.0, 1.57, 3.14])
        assert_array_equal(self.output.xdomain, xdomain)

        del self.output._xdomain
        with self.assertRaises(Exception):
            self.output.xdomain

    def test_ydomain(self) -> None:
        """Test Output's ydomain property."""
        ydomain = array([1.0, 2.0, 3.0])
        assert_array_equal(self.output.ydomain, ydomain)

        del self.output._ydomain
        with self.assertRaises(Exception):
            self.output.ydomain

    def test_zdomain(self) -> None:
        """Test Output's zdomain property."""
        zdomain = array([-1.0, 0.0, 1.0])
        assert_array_equal(self.output.zdomain, zdomain)

        del self.output._zdomain
        with self.assertRaises(Exception):
            self.output.zdomain

    def test_nx(self) -> None:
        """Test Output's nx property."""
        nx = 5
        self.assertEqual(self.output.nx, nx)

        del self.output._vars
        with self.assertRaises(Exception):
            self.output.nx

    def test_ny(self) -> None:
        """Test Output's ny property."""
        ny = 3
        self.assertEqual(self.output.ny, ny)

        del self.output._vars
        with self.assertRaises(Exception):
            self.output.ny

    def test_nz(self) -> None:
        """Test Output's nz property."""
        nz = 3
        self.assertEqual(self.output.nz, nz)

        del self.output._vars
        with self.assertRaises(Exception):
            self.output.nz

    def test_nghx(self) -> None:
        """Test Output's nghx property."""
        nghx = 0
        self.assertEqual(self.output.nghx, nghx)

        del self.output._xdomain
        with self.assertRaises(Exception):
            self.output.nghx

    def test_nghy(self) -> None:
        """Test Output's nghy property."""
        nghy = 0
        self.assertEqual(self.output.nghy, nghy)

        del self.output._ydomain
        with self.assertRaises(Exception):
            self.output.nghy

    def test_nghz(self) -> None:
        """Test Output's nghz property."""
        nghz = 0
        self.assertEqual(self.output.nghz, nghz)

        del self.output._zdomain
        with self.assertRaises(Exception):
            self.output.nghz
