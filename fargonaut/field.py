"""A field handler."""

from abc import ABC, abstractmethod

from matplotlib.pyplot import axis, colorbar, figure, pcolormesh, plot, show, subplot
from numpy import float64, reshape
from numpy.typing import NDArray


class Field(ABC):
    """An abstract base field."""

    @abstractmethod
    def _load(self, num: int) -> None:
        """Load the field data from file.

        Args:
            num (int): The number of the field output time to load
        """

    @abstractmethod
    def _process_domains(self) -> None:
        """Generate the coordinates the field data are defined at."""

    def _process_data(self) -> None:
        """Reshape the field data to the domain."""
        self._data = reshape(
            self._raw, (self._output.nx, self._output.ny, self._output.nz), order="F"
        )

    @abstractmethod
    def _get_2D_cartesian_plot_data(
        self, csys: str = "cartesian", dims: str = "xy", idx: int = 0
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

    @abstractmethod
    def _get_2D_cylindrical_plot_data(
        self, csys: str = "polar", dims: str = "xy", idx: int = 0
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

    @abstractmethod
    def _get_2D_spherical_plot_data(
        self, csys: str = "polar", dims: str = "xy", idx: int = 0
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

    @abstractmethod
    def _get_1D_cartesian_plot_data(
        self, csys: str = "cartesian", dims: str = "x", idx: tuple[int, int] = (0, 0)
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 1D slice of the cartesian field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """

    @abstractmethod
    def _get_1D_cylindrical_plot_data(
        self, csys: str = "polar", dims: str = "x", idx: tuple[int, int] = (0, 0)
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 1D slice of the cylindrical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """

    @abstractmethod
    def _get_1D_spherical_plot_data(
        self, csys: str = "polar", dims: str = "x", idx: tuple[int, int] = (0, 0)
    ) -> tuple[figure, axis, colorbar]:
        """Plot a 1D slice of the spherical field.

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (tuple[int, int]): The indices of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
        """

    def plot(
        self, csys: str = "polar", dims: str = "xy", idx: int = 0
    ) -> tuple[figure, axis, colorbar] | tuple[figure, axis]:
        """Plot the field.

        dims can be "xy", "xz", "yz", "yx", "zx", "zy", taking a 2D slice of 3D data
        idx is the index at which to slice in the third dimension

        Args:
            csys (str): The coordinate system on which to plot the field
            dims (str): The dimensions of the field to plot
            idx (int): The index of the slice to plot

        Returns:
            figure: The figure containing the plot
            axis: The axes containing the plot
            colorbar: The colorbar for the field (conditional)

        Raises:
            NotImplementedError: If unknown coordinate system requested
        """
        if len(dims) == 2:
            if self._output.coordinate_system == "cartesian":
                X, Y, C, xlabel, ylabel, clabel = self._get_2D_cartesian_plot_data(
                    csys, dims, idx
                )
            elif self._output.coordinate_system == "cylindrical":
                X, Y, C, xlabel, ylabel, clabel = self._get_2D_cylindrical_plot_data(
                    csys, dims, idx
                )
            elif self._output.coordinate_system == "spherical":
                X, Y, C, xlabel, ylabel, clabel = self._get_2D_spherical_plot_data(
                    csys, dims, idx
                )
            else:
                raise NotImplementedError(f"Unable to plot on coordinate system {csys}")
            fig = figure()
            axs = subplot(111)
            pcolormesh(X, Y, C, shading="flat")
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            cb = colorbar()
            cb.set_label(clabel)
            show()
            return fig, axs, cb

        elif len(dims) == 1:
            if self._output.coordinate_system == "cartesian":
                X, Y, xlabel, ylabel = self._get_1D_cartesian_plot_data(csys, dims, idx)
            elif self._output.coordinate_system == "cylindrical":
                X, Y, xlabel, ylabel = self._get_1D_cylindrical_plot_data(
                    csys, dims, idx
                )
            elif self._output.coordinate_system == "spherical":
                X, Y, xlabel, ylabel = self._get_1D_spherical_plot_data(csys, dims, idx)
            else:
                raise NotImplementedError(f"Unable to plot on coordinate system {csys}")
            fig = figure()
            axs = subplot(111)
            plot(X, Y)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            show()
            return fig, axs

    @property
    def x(self) -> NDArray[float64]:
        """The x-coordinates at which the field is defined.

        Returns:
            NDArray: A numpy array containing the x-coordinates
        """
        return self._xdata

    @property
    def y(self) -> NDArray[float64]:
        """The y-coordinates at which the field is defined.

        Returns:
            NDArray: A numpy array containing the y-coordinates
        """
        return self._ydata

    @property
    def z(self) -> NDArray[float64]:
        """The z-coordinates at which the field is defined.

        Returns:
            NDArray: A numpy array containing the z-coordinates
        """
        return self._zdata

    @property
    def raw(self) -> NDArray[float64]:
        """The field values.

        Returns:
            NDArray: A 1D numpy array containing the field values
        """
        return self._raw

    @property
    def data(self) -> NDArray[float64]:
        """The field values.

        Returns:
            NDArray: A shaped numpy array containing the field values
        """
        return self._data
