"""JSON-able datatype interface."""

import abc
from typing import Self

from ._json import JSON


class JSONable(abc.ABC):
    """Class requiring a JSON conversion interface."""

    @classmethod
    def from_json(cls, json: JSON) -> Self:
        """Initialize object from JSON data.

        Parameters
        ----------
        json : JSON
            Data from which to initialize object.

        Returns
        -------
        Self
        """
        try:
            return cls._from_json(json)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {cls} from JSON {json}"
            ) from e

    @classmethod
    @abc.abstractmethod
    def _from_json(cls, json: JSON) -> Self:
        """Perform JSON self-initialization."""

    @abc.abstractmethod
    def to_json(self) -> JSON:
        """Convert object to JSON data.

        Returns
        -------
        JSON
        """
