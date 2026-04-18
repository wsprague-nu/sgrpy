"""Generic types."""

import abc
from collections.abc import Hashable
from typing import Protocol


class Comparable(Protocol):
    @abc.abstractmethod
    def __lt__[_T](self: _T, __value: _T) -> bool: ...


class CompareHash(Hashable, Comparable, Protocol): ...


class AsString(Protocol):
    @abc.abstractmethod
    def as_str(self) -> str: ...
