"""A magnetic field handler."""

from matplotlib.pyplot import axis, colorbar, figure
from numpy import cos, float64, fromfile, meshgrid, sin
from numpy.typing import NDArray

from fargonaut.field import Field


class MagneticField(Field):
    """A FARGO3D magnetic field.

    Attributes:
        directory: The path to the directory containing the output files
        x: The x-coordinates at which the data are defined
        y: The y-coordinates at which the data are defined
        z: The z-coordinates at which the data are defined
        raw: The raw, unshaped field data
        data: The field data mapped to the coordinates
    """

    def __init__(self, output, dimension: str, num: int) -> None:
        """Read a magnetic field.

        Args:
            output: The FARGO3D simulation output
            dimension (str): The axis corresponding to the magnetic field direction
            num (int): The number of the field output time to load
        """
        self.symbol = "B"
        self._output = output
        self._dimension = dimension
        self._raw = self._load(num)
        self._process_domains()
        self._process_data()

    def _load(self, num: int) -> NDArray[float64]:
        """Load the magnetic field data from file.

        Args:
            num (int): The number of the field output time to load

        Returns:
            NDArray: The magnetic field data
        """
        return fromfile(
            f"{self._output._directory / 'b'}{self._dimension}{num}{'.dat'}"
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

    def _get_2D_cartesian_plot_data(
        self, csys: str, dims: str, idx: int
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 2D slice of the cartesian field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (int): The index of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
            colorbar: The colorbar for the field
        """
        xdata = self._output._xdomain
        ydata = self._output._ydomain
        zdata = self._output._zdomain

        if self._output.nghx:
            xdata = xdata[self._output.nghx : -self._output.nghx]
        if self._output.nghy:
            ydata = ydata[self._output.nghy : -self._output.nghy]
        if self._output.nghz:
            zdata = zdata[self._output.nghz : -self._output.nghz]

        xgrid, ygrid, zgrid = meshgrid(xdata, ydata, zdata, indexing="ij")

        if csys == "polar":
            raise NotImplementedError
        elif csys == "cartesian":
            coord_map = {
                "x": [xgrid, "$x$"],
                "y": [ygrid, "$y$"],
                "z": [zgrid, "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims[0]][0]
        ygrid = coord_map[dims[1]][0]
        xlabel = coord_map[dims[0]][1]
        ylabel = coord_map[dims[1]][1]

        if dims == "xy":
            X = xgrid[:, :, idx]
            Y = ygrid[:, :, idx]
            C = self._data[:, :, idx]
        elif dims == "xz":
            X = xgrid[:, idx, :]
            Y = ygrid[:, idx, :]
            C = self._data[:, idx, :]
        elif dims == "yz":
            X = xgrid[idx, :, :]
            Y = ygrid[idx, :, :]
            C = self._data[idx, :, :]

        clabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, C, xlabel, ylabel, clabel

    def _get_2D_cylindrical_plot_data(
        self, csys: str, dims: str, idx: int
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 2D slice of the cylindrical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (int): The index of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
            colorbar: The colorbar for the field
        """
        phidata = self._output._xdomain
        rdata = self._output._ydomain
        zdata = self._output._zdomain

        if self._output.nghx:
            phidata = phidata[self._output.nghx : -self._output.nghx]
        if self._output.nghy:
            rdata = rdata[self._output.nghy : -self._output.nghy]
        if self._output.nghz:
            zdata = zdata[self._output.nghz : -self._output.nghz]

        phigrid, rgrid, zgrid = meshgrid(phidata, rdata, zdata, indexing="ij")

        if csys == "polar":
            coord_map = {
                "x": [phigrid, r"$\phi$"],
                "y": [rgrid, "$r$"],
                "z": [zgrid, "$z$"],
            }
        elif csys == "cartesian":
            coord_map = {
                "x": [rgrid * cos(phigrid), "$x$"],
                "y": [rgrid * sin(phigrid), "$y$"],
                "z": [zgrid, "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims[0]][0]
        ygrid = coord_map[dims[1]][0]
        xlabel = coord_map[dims[0]][1]
        ylabel = coord_map[dims[1]][1]

        if dims == "xy":
            X = xgrid[:, :, idx]
            Y = ygrid[:, :, idx]
            C = self._data[:, :, idx]
        elif dims == "xz":
            X = xgrid[:, idx, :]
            Y = ygrid[:, idx, :]
            C = self._data[:, idx, :]
        elif dims == "yz":
            X = xgrid[idx, :, :]
            Y = ygrid[idx, :, :]
            C = self._data[idx, :, :]

        clabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, C, xlabel, ylabel, clabel

    def _get_2D_spherical_plot_data(
        self, csys: str, dims: str, idx: int
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 2D slice of the spherical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (int): The index of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
            colorbar: The colorbar for the field
        """
        phidata = self._output._xdomain
        rdata = self._output._ydomain
        thetadata = self._output._zdomain

        if self._output.nghx:
            phidata = phidata[self._output.nghx : -self._output.nghx]
        if self._output.nghy:
            rdata = rdata[self._output.nghy : -self._output.nghy]
        if self._output.nghz:
            thetadata = thetadata[self._output.nghz : -self._output.nghz]

        phigrid, rgrid, thetagrid = meshgrid(phidata, rdata, thetadata, indexing="ij")

        if csys == "polar":
            coord_map = {
                "x": [phigrid, r"$\phi$"],
                "y": [rgrid, "$r$"],
                "z": [thetagrid, r"$\theta$"],
            }
        elif csys == "cartesian":
            coord_map = {
                "x": [rgrid * cos(phigrid) * sin(thetagrid), "$x$"],
                "y": [rgrid * sin(phigrid) * sin(thetagrid), "$y$"],
                "z": [rgrid * cos(thetagrid), "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims[0]][0]
        ygrid = coord_map[dims[1]][0]
        xlabel = coord_map[dims[0]][1]
        ylabel = coord_map[dims[1]][1]

        if dims == "xy":
            X = xgrid[:, :, idx]
            Y = ygrid[:, :, idx]
            C = self._data[:, :, idx]
        elif dims == "xz":
            X = xgrid[:, idx, :]
            Y = ygrid[:, idx, :]
            C = self._data[:, idx, :]
        elif dims == "yz":
            X = xgrid[idx, :, :]
            Y = ygrid[idx, :, :]
            C = self._data[idx, :, :]

        clabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, C, xlabel, ylabel, clabel

    def _get_1D_cartesian_plot_data(
        self, csys: str, dims: str, idx: tuple[int, int]
    ) -> tuple[figure, axis]:
        """Plot a 1D slice of the cartesian field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """
        xdata = self._xdata
        ydata = self._ydata
        zdata = self._zdata

        xgrid, ygrid, zgrid = meshgrid(xdata, ydata, zdata, indexing="ij")

        if csys == "polar":
            raise NotImplementedError
        elif csys == "cartesian":
            coord_map = {
                "x": [xgrid, "$x$"],
                "y": [ygrid, "$y$"],
                "z": [zgrid, "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims][0]
        xlabel = coord_map[dims][1]

        if dims == "x":
            X = xgrid[:, idx[0], idx[1]]
            Y = self._data[:, idx[0], idx[1]]
        elif dims == "y":
            X = xgrid[idx[0], :, idx[1]]
            Y = self._data[idx[0], :, idx[1]]
        elif dims == "z":
            X = xgrid[idx[0], idx[1], :]
            Y = self._data[idx[0], idx[1], :]

        ylabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, xlabel, ylabel

    def _get_1D_cylindrical_plot_data(
        self, csys: str, dims: str, idx: tuple[int, int]
    ) -> tuple[figure, axis]:
        """Plot a 1D slice of the cylindrical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """
        phidata = self._xdata
        rdata = self._ydata
        zdata = self._zdata

        phigrid, rgrid, zgrid = meshgrid(phidata, rdata, zdata, indexing="ij")

        if csys == "polar":
            coord_map = {
                "x": [phigrid, r"$\phi$"],
                "y": [rgrid, "$r$"],
                "z": [zgrid, "$z$"],
            }
        elif csys == "cartesian":
            coord_map = {
                "x": [rgrid * cos(phigrid), "$x$"],
                "y": [rgrid * sin(phigrid), "$y$"],
                "z": [zgrid, "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims][0]
        xlabel = coord_map[dims][1]

        if dims == "x":
            X = xgrid[:, idx[0], idx[1]]
            Y = self._data[:, idx[0], idx[1]]
        elif dims == "y":
            X = xgrid[idx[0], :, idx[1]]
            Y = self._data[idx[0], :, idx[1]]
        elif dims == "z":
            X = xgrid[idx[0], idx[1], :]
            Y = self._data[idx[0], idx[1], :]

        ylabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, xlabel, ylabel

    def _get_1D_spherical_plot_data(
        self, csys: str, dims: str, idx: tuple[int, int]
    ) -> tuple[figure, axis]:
        """Plot a 1D slice of the spherical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """
        phidata = self._xdata
        rdata = self._ydata
        thetadata = self._zdata

        phigrid, rgrid, thetagrid = meshgrid(phidata, rdata, thetadata, indexing="ij")

        if csys == "polar":
            coord_map = {
                "x": [phigrid, r"$\phi$"],
                "y": [rgrid, "$r$"],
                "z": [thetagrid, r"$\theta$"],
            }
        elif csys == "cartesian":
            coord_map = {
                "x": [rgrid * cos(phigrid) * sin(thetagrid), "$x$"],
                "y": [rgrid * sin(phigrid) * sin(thetagrid), "$y$"],
                "z": [rgrid * cos(thetagrid), "$z$"],
            }
        else:
            raise ValueError(f"Unknown coordinate system {csys}")

        xgrid = coord_map[dims][0]
        xlabel = coord_map[dims][1]

        if dims == "x":
            X = xgrid[:, idx[0], idx[1]]
            Y = self._data[:, idx[0], idx[1]]
        elif dims == "y":
            X = xgrid[idx[0], :, idx[1]]
            Y = self._data[idx[0], :, idx[1]]
        elif dims == "z":
            X = xgrid[idx[0], idx[1], :]
            Y = self._data[idx[0], idx[1], :]

        ylabel = f"${self.symbol}_{coord_map[self._dimension][1].strip('$')}$"

        return X, Y, xlabel, ylabel
