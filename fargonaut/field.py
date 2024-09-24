"""A field handler."""

from abc import ABC, abstractmethod


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

    @abstractmethod
    def _process_data(self) -> None:
        """Reshape the field data to the domain."""
