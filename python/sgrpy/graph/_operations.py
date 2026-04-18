"""Simple datastructures representing operations on CGraph objects."""

import dataclasses
from collections.abc import Iterable

from ._permutations import IndexMapping


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AddNode:
    """Addition of a colored node to a `sgrpy.graph.CGraph`.

    Parameters
    ----------
    connections : tuple[int, ...]
        Tuple of indices in the current graph to which the new node shall be
        joined.
    color : int
        Color of the new node.
    """

    connections: tuple[int, ...]
    """
    Tuple of indices in the current graph to which the new node shall be
    joined.
    """
    color: int
    """Color of the new node."""


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class Extension[_E]:
    """Extension of an edge from a node, with specific type.

    Parameters
    ----------
    idx : int
        Originating node index.
    typ : _E
        Edge type.
    """

    idx: int
    """Originating node index."""
    typ: _E
    """Edge type."""


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AddNodeT[_N, _E]:
    """Addition of a labeled node to a generic graph."""

    _ext: tuple[Extension[_E], ...]
    _typ: _N

    @classmethod
    def new(cls, ext: Iterable[Extension[_E]], typ: _N) -> "AddNodeT[_N, _E]":
        return AddNodeT(_ext=tuple(ext), _typ=typ)

    @classmethod
    def new_sort(
        cls, ext: Iterable[Extension[_E]], typ: _N
    ) -> "AddNodeT[_N, _E]":
        return AddNodeT(_ext=tuple(sorted(ext)), _typ=typ)

    @property
    def ext(self) -> tuple[Extension[_E], ...]:
        """tuple[Extension[_E]]: Tuple of extensions to the current graph."""
        return self._ext

    @property
    def typ(self) -> _N:
        """_N: Label of the new node."""
        return self._typ

    def sort(self) -> "AddNodeT[_N, _E]":
        return AddNodeT.new_sort(self._ext, self._typ)

    def permute(self, mapping: IndexMapping) -> "AddNodeT[_N, _E]":
        """Permute operation indices according to mapping."""
        map_tup = mapping.inv().as_tuple_int()
        new_ext = tuple(
            dataclasses.replace(ext, idx=map_tup[ext.idx]) for ext in self._ext
        )
        new_op = dataclasses.replace(self, _ext=new_ext)
        return new_op


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class SubNode:
    """Dataclass representing the subtraction of a node from a generic graph.

    Parameters
    ----------
    index : int
        Index of the node to be removed.
    """

    index: int
    """Index of the node to be removed."""
