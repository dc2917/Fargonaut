"""A velocity field handler."""

from matplotlib.pyplot import axis, colorbar, figure, pcolormesh, show, subplot
from numpy import cos, float64, fromfile, meshgrid, reshape, sin
from numpy.typing import NDArray

from fargonaut.field import Field


class Velocity(Field):
    """A FARGO3D gas velocity field.

    Attributes:
        directory: The path to the directory containing the output files
        x: The x-coordinates at which the data are defined
        y: The y-coordinates at which the data are defined
        z: The z-coordinates at which the data are defined
        raw: The raw, unshaped field data
        data: The field velocity data mapped to the coordinates
    """

    def __init__(self, output, dimension: str, num: int) -> None:
        """Read a velocity field.

        Args:
            output: The FARGO3D simulation output
            dimension (str): The axis corresponding to the velocity direction
            num (int): The number of the field output time to load
        """
        self._output = output
        self._dimension = dimension
        self._raw = self._load(num)
        self._process_domains()
        self._process_data()

    def _load(self, num: int) -> NDArray[float64]:
        """Load the velocity field data from file.

        Args:
            num (int): The number of the field output time to load

        Returns:
            NDArray: The velocity field data
        """
        return fromfile(
            f"{self._output._directory / 'gasv'}{self._dimension}{num}{'.dat'}"
        )

    def _process_domains(self) -> None:
        """Generate the coordinates at which the field data are defined."""
        if self._dimension == "x":
            self._xdata = self._output._xdomain[:-1]
            self._ydata = 0.5 * (self._output._ydomain[:-1] + self._output._ydomain[1:])
            self._zdata = 0.5 * (self._output._zdomain[:-1] + self._output._zdomain[1:])
        elif self._dimension == "y":
            self._xdata = 0.5 * (self._output._xdomain[:-1] + self._output._xdomain[1:])
            self._ydata = self._output._ydomain[:-1]
            self._zdata = 0.5 * (self._output._zdomain[:-1] + self._output._zdomain[1:])
        else:
            self._xdata = 0.5 * (self._output._xdomain[:-1] + self._output._xdomain[1:])
            self._ydata = 0.5 * (self._output._ydomain[:-1] + self._output._ydomain[1:])
            self._zdata = self._output._zdomain[:-1]

        if self._output.nghx:
            self._xdata = self._xdata[self._output.nghx : -self._output.nghx]
        if self._output.nghy:
            self._ydata = self._ydata[self._output.nghy : -self._output.nghy]
        if self._output.nghz:
            self._zdata = self._zdata[self._output.nghz : -self._output.nghz]

    def _process_data(self) -> None:
        """Reshape the data to the domain."""
        self._data = reshape(self._raw, (self._output.ny, self._output.nx))

    def plot(self, csys: str = "polar") -> tuple[figure, axis, colorbar]:
        """Plot the velocity field.

        Args:
            csys (str): The coordinate system on which to plot the field

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
            colorbar: The colorbar for the velocity field
        """
        # assume x, y = phi, r
        coord_map = {"x": "r", "y": r"\phi", "z": "z"}
        phidata = self._output._xdomain
        rdata = self._output._ydomain[3:-3]
        rgrid, phigrid = meshgrid(rdata, phidata, indexing="ij")
        fig = figure()
        axs = subplot(111)
        clabel = f"$v_{coord_map[self._dimension]}$"
        if csys == "polar":
            xlabel = r"$\phi$"
            ylabel = "$r$"
            pcolormesh(phigrid, rgrid, self._data, shading="flat")
        elif csys == "cartesian":
            xlabel = "$x$"
            ylabel = "$y$"
            xgrid = rgrid * cos(phigrid)
            ygrid = rgrid * sin(phigrid)
            pcolormesh(xgrid, ygrid, self._data, shading="flat")
        else:
            raise ValueError(f"Unknown coordinate system {csys}")
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        cb = colorbar()
        cb.set_label(clabel)
        show()
        return fig, axs, cb

    @property
    def x(self) -> NDArray[float64]:
        """The x-coordinates at which the velocity is defined.

        Returns:
            NDArray: A numpy array containing the x-coordinates
        """
        return self._xdata

    @property
    def y(self) -> NDArray[float64]:
        """The y-coordinates at which the velocity is defined.

        Returns:
            NDArray: A numpy array containing the y-coordinates
        """
        return self._ydata

    @property
    def z(self) -> NDArray[float64]:
        """The z-coordinates at which the velocity is defined.

        Returns:
            NDArray: A numpy array containing the z-coordinates
        """
        return self._zdata

    @property
    def raw(self) -> NDArray[float64]:
        """The velocity values.

        Returns:
            NDArray: A 1D numpy array containing the velocity values
        """
        return self._raw

    @property
    def data(self) -> NDArray[float64]:
        """The velocity values.

        Returns:
            NDArray: A shaped numpy array containing the velocity values
        """
        return self._data
