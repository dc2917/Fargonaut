"""A field handler."""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from numpy import float64, reshape
from numpy.typing import NDArray


class Field(ABC):
    """An abstract base field."""

    def _check_valid_for_arithmetic(self, other: "Field", operation: "str") -> None:
        """Check if two fields are suitable for arithmetic operations.

        Fields must be defined at the same coordinates in order to be valid.

        Args:
            other (Field): The field on the RHS of the operator.
            operation (str): The name of the operation.

        Raises:
            Exception: If other is not a Field.
            Exception: If the fields are incompatible.
        """
        if not isinstance(other, Field) and not isinstance(other, DerivedField):
            raise Exception(f"{other} is of an invalid class.")
        if (
            (self._xdata != other._xdata).all()
            or (self._ydata != other._ydata).all()
            or (self._zdata != other._zdata).all()
        ):
            raise Exception(
                f"Cannot {operation} fields defined at different coordinates."
            )

    def __add__(self, other: "Field") -> "DerivedField":
        """Add one field to another.

        Args:
            other (Field): The field to be added.

        Returns:
            DerivedField: The sum of the two fields.
        """
        self._check_valid_for_arithmetic(other, "add")
        result = DerivedField(self)
        result.symbol = f"{self.symbol} + {other.symbol}"
        result._raw = self._raw + other._raw
        result._data = self._data + other._data
        return result

    def __sub__(self, other) -> "DerivedField":
        """Subtract one field from another.

        Args:
            other (Field): The field to be subtracted.

        Returns:
            DerivedField: The difference of the two fields.
        """
        self._check_valid_for_arithmetic(other, "subtract")
        result = DerivedField(self)
        result.symbol = f"{self.symbol} - {other.symbol}"
        result._raw = self._raw - other._raw
        result._data = self._data - other._data
        return result

    def __mul__(self, other) -> "DerivedField":
        """Multiply one field by another.

        Args:
            other (Field): The field to be multiplied by.

        Returns:
            DerivedField: The product of the two fields.
        """
        self._check_valid_for_arithmetic(other, "multiply")
        result = DerivedField(self)
        result.symbol = rf"{self.symbol} \times {other.symbol}"
        result._raw = self._raw * other._raw
        result._data = self._data * other._data
        return result

    def __truediv__(self, other) -> "DerivedField":
        """Divide one field by another.

        Args:
            other (Field): The field to be divided by.

        Returns:
            DerivedField: The ratio of the two fields.
        """
        self._check_valid_for_arithmetic(other, "divide")
        result = DerivedField(self)
        result.symbol = f"{self.symbol} / {other.symbol}"
        result._raw = self._raw / other._raw
        result._data = self._data / other._data
        return result

    def __pow__(self, power: float | int) -> "DerivedField":
        """Raise the field data to a power.

        Args:
            power (float | int): The power to raise the field to.

        Returns:
            DerivedField: The field with data raised to the requested power.
        """
        result = DerivedField(self)
        result.symbol = f"{self.symbol}^{power}"
        result._raw = self._raw**power
        result._data = self._data**power
        return result

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
    ) -> tuple[plt.figure, plt.subplot, plt.colorbar]:
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
    ) -> tuple[plt.figure, plt.subplot, plt.colorbar]:
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
    ) -> tuple[plt.figure, plt.subplot, plt.colorbar]:
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
    ) -> tuple[plt.figure, plt.subplot]:
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
    ) -> tuple[plt.figure, plt.subplot]:
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
    ) -> tuple[plt.figure, plt.subplot]:
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
    ) -> tuple[plt.figure, plt.subplot, plt.colorbar] | tuple[plt.figure, plt.subplot]:
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
            fig = plt.figure()
            axs = plt.subplot(111)
            plt.pcolormesh(X, Y, C, shading="flat")
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            cb = plt.colorbar()
            cb.set_label(clabel)
            plt.show()
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
            fig = plt.figure()
            axs = plt.subplot(111)
            plt.plot(X, Y)
            axs.set_xlabel(xlabel)
            axs.set_ylabel(ylabel)
            plt.show()
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


class DerivedField:
    """A derived field."""

    def __new__(cls, base: Field) -> "DerivedField":
        """Define a new field, derived from another.

        Args:
            base (Field): The field to derive from.

        Returns:
            DerivedField: A new field, derived from another.
        """
        cls._get_2D_cartesian_plot_data = base.__class__._get_2D_cartesian_plot_data
        cls._get_2D_cylindrical_plot_data = base.__class__._get_2D_cylindrical_plot_data
        cls._get_2D_spherical_plot_data = base.__class__._get_2D_spherical_plot_data
        cls._get_1D_cartesian_plot_data = base.__class__._get_1D_cartesian_plot_data
        cls._get_1D_cylindrical_plot_data = base.__class__._get_1D_cylindrical_plot_data
        cls._get_1D_spherical_plot_data = base.__class__._get_1D_spherical_plot_data
        cls.x = base.__class__.x
        cls.y = base.__class__.y
        cls.z = base.__class__.z
        cls.raw = base.__class__.raw
        cls.data = base.__class__.data
        cls.plot = Field.plot
        return super().__new__(cls)

    def __init__(self, base: Field) -> None:
        """Create a derived field.

        Args:
            base (Field): The field to derive from.
        """
        self.symbol = base.symbol
        self._output = base._output
        self._xdata = base._xdata
        self._ydata = base._ydata
        self._zdata = base._zdata

    def set_symbol(self, symbol: str) -> None:
        """Set the symbol representing the field's quantity.

        Args:
            symbol (str): The symbol to represent the quantity.
        """
        self.symbol = symbol
