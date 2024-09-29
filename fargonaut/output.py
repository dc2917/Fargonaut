"""A FARGO3D simulation output reader."""

from pathlib import Path

from numpy import array, float64
from numpy.typing import NDArray

from fargonaut.field import Field
from fargonaut.fields.density import Density
from fargonaut.fields.energy import Energy
from fargonaut.fields.magnetic_field import MagneticField
from fargonaut.fields.velocity import Velocity


class Output:
    """A FARGO3D simulation output.

    Attributes:
        directory: The path to the directory containing the output files
        domain_x: The x domain over which the output data are defined
        domain_y: The y domain over which the output data are defined
        domain_z: The z domain over which the output data are defined
        nghx: The number of ghost cells in the x dimension
        nghy: The number of ghost cells in the y dimension
        nghz: The number of ghost cells in the z dimension
        opts: The options used in the simulation
        vars: The variables defined for the simulation
    """

    def __init__(self, directory: str) -> None:
        """Read a FARGO3D output.

        Args:
            directory (str): The path to the directory containing the output
                             files
        """
        self._directory = Path(directory)
        self._read_opts()
        self._read_vars()
        self._read_domains()

    def _read_domains(self) -> None:
        """Read and store the contents of the output's dimensions files."""
        fid = open(f"{self._directory / 'domain_x.dat'}")
        xdata = fid.read().split("\n")
        fid.close()
        xdata.pop()
        self._xdomain = array([float(i) for i in xdata])

        fid = open(f"{self._directory / 'domain_y.dat'}")
        ydata = fid.read().split("\n")
        fid.close()
        ydata.pop()
        self._ydomain = array([float(i) for i in ydata])

        fid = open(f"{self._directory / 'domain_z.dat'}")
        zdata = fid.read().split("\n")
        fid.close()
        zdata.pop()
        self._zdomain = array([float(i) for i in zdata])

    def _read_opts(self) -> None:
        """Read and store the options from the output's summary0 file."""
        fid = open(f"{self._directory / 'summary0.dat'}")
        line = fid.readline()
        target = "COMPILATION OPTION SECTION:\n"
        while not line == target:
            line = fid.readline()
        line = fid.readline()
        opts = fid.readline().split()
        fid.close()
        self._opts = tuple(opt.lstrip("-D") for opt in opts)

    def _read_vars(self) -> None:
        """Read and store the contents of the output's variables file."""
        fid = open(f"{self._directory / 'variables.par'}")
        variables = {}
        for line in fid:
            (key, val) = line.split()
            variables[key] = val
        fid.close()
        self._vars = variables

    def _read_units(self) -> None:
        """Read the contents of the output's variables file."""
        raise NotImplementedError

    def get_var(self, var_name: str) -> str:
        """Get the value of a variable.

        Args:
            var_name (str): The name of the variable to query

        Returns:
            str: The value of the variable
        """
        return self._vars[var_name]

    def get_opt(self, opt_name: str) -> bool:
        """Get whether an option was set.

        Args:
            opt_name (str): The name of the option to query

        Returns:
            bool: Whether the option was used
        """
        return opt_name in self._opts

    def get_field(self, name: str, num: int) -> Field:
        """Load the field at a given output time.

        Args:
            name (str): The name of the field to get
            num (str): The number of the field output time to get

        Returns:
            Field: The field

        Raises:
            NotImplementedError: An invalid field was requested
        """
        if name == "gasdens":
            return Density(self, num)
        elif name == "gasenergy":
            return Energy(self, num)
        elif name == "bx":
            return MagneticField(self, "x", num)
        elif name == "by":
            return MagneticField(self, "y", num)
        elif name == "bz":
            return MagneticField(self, "z", num)
        elif name == "gasvx":
            return Velocity(self, "x", num)
        elif name == "gasvy":
            return Velocity(self, "y", num)
        elif name == "gasvz":
            return Velocity(self, "z", num)
        else:
            raise NotImplementedError

    @property
    def coordinate_system(self) -> str:
        """The coordinate system used in the simulation.

        Returns:
            str: The coordinate system used in the simulation

        Raises:
            Exception: If _read_opts has not been executed
        """
        try:
            return self._vars["COORDINATES"]
        except AttributeError:
            raise Exception("Output variables have not been read.")

    @property
    def includes_ghosts(self) -> bool:
        """Whether field outputs contain ghost cell values.

        Returns:
            bool: Whether ghost cells are included in the field data files

        Raises:
            Exception: If _read_opts has not been executed
        """
        try:
            return "WRITEGHOSTS" in self._opts
        except AttributeError:
            raise Exception("Simulation options have not been read.")

    @property
    def xdomain(self) -> NDArray[float64]:
        """Domain of the output in the x dimension.

        Returns:
            NDArray: A numpy array containing the x-coordinates

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return self._xdomain
        except AttributeError:
            raise Exception("Output domains have not been read.")

    @property
    def ydomain(self) -> NDArray[float64]:
        """Domain of the output in the y dimension.

        Returns:
            NDArray: A numpy array containing the y-coordinates

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return self._ydomain
        except AttributeError:
            raise Exception("Output domains have not been read.")

    @property
    def zdomain(self) -> NDArray[float64]:
        """Domain of the output in the z dimension.

        Returns:
            NDArray: A numpy array containing the z-coordinates

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return self._zdomain
        except AttributeError:
            raise Exception("Output domains have not been read.")

    @property
    def nx(self) -> int:
        """Number of cells used in the x dimension.

        Returns:
            int: The number of cells used in the x dimension.

        Raises:
            Exception: If _read_vars has not been executed
        """
        try:
            return int(self._vars["NX"])
        except AttributeError:
            raise Exception("Output variables have not been read.")

    @property
    def ny(self) -> int:
        """Number of cells used in the y dimension.

        Returns:
            int: The number of cells used in the y dimension.

        Raises:
            Exception: If _read_vars has not been executed
        """
        try:
            return int(self._vars["NY"])
        except AttributeError:
            raise Exception("Output variables have not been read.")

    @property
    def nz(self) -> int:
        """Number of cells used in the z dimension.

        Returns:
            int: The number of cells used in the z dimension.

        Raises:
            Exception: If _read_vars has not been executed
        """
        try:
            return int(self._vars["NZ"])
        except AttributeError:
            raise Exception("Output variables have not been read.")

    @property
    def nghx(self) -> int:
        """Number of ghost cells used in the x dimension.

        Returns:
            int: The number of ghost cells used in the x dimension.

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return int((len(self._xdomain) - self.nx - 1) / 2)
        except AttributeError:
            raise Exception("Output domains have not been read.")

    @property
    def nghy(self) -> int:
        """Number of ghost cells used in the y dimension.

        Returns:
            int: The number of ghost cells used in the y dimension.

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return int((len(self._ydomain) - self.ny - 1) / 2)
        except AttributeError:
            raise Exception("Output domains have not been read.")

    @property
    def nghz(self) -> int:
        """Number of ghost cells used in the z dimension.

        Returns:
            int: The number of ghost cells used in the z dimension.

        Raises:
            Exception: If _read_domains has not been executed
        """
        try:
            return int((len(self._zdomain) - self.nz - 1) / 2)
        except AttributeError:
            raise Exception("Output domains have not been read.")
