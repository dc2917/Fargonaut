"""Tests for output module."""

import os
import tempfile
import unittest
from pathlib import Path

from numpy import array
from numpy.testing import assert_array_equal

from fargonaut.output import Output

tempdir = tempfile.gettempdir()

domain_x_file_name = tempdir + "/domain_x.dat"
domain_y_file_name = tempdir + "/domain_y.dat"
domain_z_file_name = tempdir + "/domain_z.dat"
summary0_file_name = tempdir + "/summary0.dat"
variables_file_name = tempdir + "/variables.par"

DOMAIN_X = "-3.14\n-1.57\n0.0\n1.57\n3.14\n"
DOMAIN_Y = "1.0\n2.0\n3.0\n"
DOMAIN_Z = "-1.0\n0.0\n1.0\n"
SUMMARY0 = (
    "stuff\n==\nCOMPILATION OPTION SECTION:\n==\n"
    "-DX -DY -DISOTHERMAL -DCYLINDRICAL\nmore stuff\n"
)
VARIABLES = "VAR1\tVAL1\nVAR2\tVAL2\nNX\t5\nNY\t3\nNZ\t3\nVARN\tVALN\n"


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
        os.rename(cls.domain_x_file.name, domain_x_file_name)

        cls.domain_y_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.domain_y_file.write(DOMAIN_Y)
        cls.domain_y_file.close()
        os.rename(cls.domain_y_file.name, domain_y_file_name)

        cls.domain_z_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.domain_z_file.write(DOMAIN_Z)
        cls.domain_z_file.close()
        os.rename(cls.domain_z_file.name, domain_z_file_name)

        cls.summary0_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".dat"
        )
        cls.summary0_file.write(SUMMARY0)
        cls.summary0_file.close()
        os.rename(cls.summary0_file.name, summary0_file_name)

        cls.variables_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", newline="\n", suffix=".par"
        )
        cls.variables_file.write(VARIABLES)
        cls.variables_file.close()
        os.rename(cls.variables_file.name, variables_file_name)

    @classmethod
    def tearDownClass(cls) -> None:
        """Delete temporary output files."""
        os.remove(domain_x_file_name)
        os.remove(domain_y_file_name)
        os.remove(domain_z_file_name)
        os.remove(summary0_file_name)
        os.remove(variables_file_name)

    def setUp(self) -> None:
        """Create output fixture."""
        self.output = Output(tempdir)

    def tearDown(self) -> None:
        """Destroy output fixture."""
        del self.output

    def test_init(self) -> None:
        """Test Output's __init__ method."""
        self.assertEqual(self.output._directory, Path(tempdir))

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
            "VAR2": "VAL2",
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

    def test_get_field(self) -> None:
        """Test Output's get_field method."""
        with self.assertRaises(NotImplementedError):
            self.output.get_field("gasvphi", 25)
