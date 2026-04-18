"""Permutation operations and classes."""

import dataclasses
import logging
from collections.abc import Sequence
from typing import TypeVar

import numpy
import numpy.typing

_T = TypeVar("_T")


def _convert_seq(
    seq: Sequence[int | None],
) -> numpy.typing.NDArray[numpy.uintp]:
    new_list = [0] + [0 if x is None else x + 1 for x in seq]
    new_map: numpy.typing.NDArray[numpy.uintp] = numpy.array(
        new_list, dtype=numpy.uintp
    )
    return new_map


def _invert_map(
    mapping: numpy.typing.NDArray[numpy.uintp],
) -> numpy.typing.NDArray[numpy.uintp]:
    new_map = numpy.zeros(mapping.max() + numpy.uintp(1), dtype=numpy.uintp)
    map_set: set[int] = set()
    for i, x in enumerate(mapping):
        x_int = int(x)
        if x_int in map_set:
            if x_int != 0:
                logging.debug(
                    f"Imperfect inverse; multiple mappings to {x_int - 1}"
                )
            continue
        new_map[x] = i
        map_set.add(x_int)
    return new_map


@dataclasses.dataclass(frozen=True, slots=True)
class IndexMapping:
    """Mapping of nodes from one set of indices to another.

    Mappings are not guaranteed to be permutations, and may map to "None".
    When mapping indices from A to B, the indices of the mapping represent the
    positions in domain B and the values of the mapping represent their
    positions in domain A. In order to permute a sequence from domain A to
    domain B, apply the `permute` or `permute_not_none` methods.
    """

    _nodes_map: numpy.typing.NDArray[numpy.uintp]

    @classmethod
    def from_seq(cls, seq: Sequence[None | int]) -> "IndexMapping":
        """Initialize mapping from a sequence.

        Parameters
        ----------
        seq : Sequence[None | int]
            Index is index in domain B, value is index in domain A.

        Returns
        -------
        IndexMapping
        """
        return IndexMapping(_nodes_map=_convert_seq(seq))

    @classmethod
    def from_seq_inv(cls, seq: Sequence[None | int]) -> "IndexMapping":
        """Initialize mapping from a sequence.

        Parameters
        ----------
        seq : Sequence[None | int]
            Index is index in domain A, value is index in domain B.

        Returns
        -------
        IndexMapping
        """
        new_map = _invert_map(_convert_seq(seq))
        return IndexMapping(_nodes_map=new_map)

    @classmethod
    def identity(cls, n: int) -> "IndexMapping":
        """Initialize identity map A -> A.

        Parameters
        ----------
        n : int
            Size of the domain of A.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> sgrpy.graph.IndexMapping.identity(4).permute((3,1,4,5))
        (3, 1, 4, 5)
        """
        # new identity mapping of size n
        new_map = numpy.arange(0, stop=n + 1, dtype=numpy.uintp)
        return IndexMapping(_nodes_map=new_map)

    @classmethod
    def identity_minus(cls, n: int, i: int) -> "IndexMapping":
        """Initialize mapping which pops an element from domain A.

        Parameters
        ----------
        n : int
            Size of domain A.
        i : int
            Index of A which is dropped from the map to domain B.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> sgrpy.graph.IndexMapping.identity_minus(4,2).permute((3,1,4,5))
        (3, 1, 5)
        """
        # maps sequence of size n to sequence without entry i
        new_map_left = numpy.arange(0, stop=i + 1, dtype=numpy.uintp)
        new_map_right = numpy.arange(i + 2, stop=n + 1, dtype=numpy.uintp)
        return IndexMapping(
            _nodes_map=numpy.concatenate([new_map_left, new_map_right])
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexMapping):
            return False
        return numpy.array_equal(self._nodes_map, other._nodes_map)

    def __hash__(self) -> int:
        return hash(str(self._nodes_map))

    def __len__(self) -> int:
        return self._nodes_map.size - 1

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_seq("
            f"{
                repr(
                    [
                        int(x - numpy.uintp(1)) if x > 0 else None
                        for x in self._nodes_map[1:]
                    ]
                )
            })"
        )

    def append(self, i: int | None) -> "IndexMapping":
        """Create a new mapping with an additional index appended to the end.

        Parameters
        ----------
        i : int | None
            Index in domain A which corresponds to the last position in domain
            B.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> sgrpy.graph.IndexMapping.identity(3).append(0).permute((3,1,4,5))
        (3, 1, 4, 3)
        """
        # append value to end of mapping
        new_val = numpy.uintp(0) if i is None else numpy.uintp(i + 1)
        new_map = numpy.append(self._nodes_map, new_val)
        return IndexMapping(_nodes_map=new_map)

    def subst(self, i: int, m: int | None) -> "IndexMapping":
        """Create a new mapping with a single substitution.

        Parameters
        ----------
        i : int
            Index in domain B to change mapping.
        m : int | None
            Index in domain A which will now correspond to index `i` in domain
            B.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> sgrpy.graph.IndexMapping.identity(4).subst(0,3).permute((3,1,4,5))
        (5, 1, 4, 5)
        """
        # substitute map at position i with value m
        new_map = self._nodes_map.copy()
        new_map[i + 1] = numpy.uintp(0) if m is None else numpy.uintp(m + 1)
        return IndexMapping(_nodes_map=new_map)

    def inv(self) -> "IndexMapping":
        """Invert mapping (convert to map from B to A).

        The resulting mapping will have a size equal to the maximum index in
        domain A which is a component of the map. Positions in domain A which do
        not have a map will be assigned `None`. If the original mapping is
        one-to-many, the lowest corresponding index in domain B will be
        assigned.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> IndexMapping.from_seq((4, 3, 1, 4, None)).inv().as_tuple()
        (None, 2, None, 1, 0)
        """
        # inversion of current map
        return IndexMapping(_nodes_map=_invert_map(self._nodes_map))

    def compose(self, other_map: "IndexMapping") -> "IndexMapping":
        """Composes two mappings.

        Assuming `other_map` maps A -> B, and `self` maps B -> C, produces the
        map which maps A -> C.

        Parameters
        ----------
        other_map : IndexMapping
            Map to be composed with `self`.

        Returns
        -------
        IndexMapping

        Examples
        --------
        >>> mapAB = sgrpy.graph.IndexMapping.from_seq((4, 3, 1, 4, None))
        >>> mapBC = sgrpy.graph.IndexMapping.from_seq((0, 5, 2, 8))
        >>> mapBC.compose(mapAB).as_tuple()
        (4, None, 1, None)
        """
        # produces the map which implements self.permute(other.permute(x))
        other_array = other_map._nodes_map
        self_array = self._nodes_map
        other_len = other_array.size
        self_array_trunc = self_array.copy()
        self_array_trunc[self_array_trunc >= other_len] = numpy.uintp(0)
        new_map = other_array[self_array_trunc]
        return IndexMapping(_nodes_map=new_map)

    def permute(self, seq: Sequence[_T]) -> Sequence[None | _T]:
        """Apply mapping to change index positions of arbitrary sequence.

        When `self` maps domain A to domain B, an item with index i_B in the
        returned sequence will be the one with the corresponding index i_A in
        the input sequence. Will fail if `seq` is not long enough to support
        the mapping.

        Parameters
        ----------
        seq: Sequence[_T]
            Sequence to be mapped.

        Returns
        -------
        Sequence[None | _T]

        Examples
        --------
        >>> tup = (3, 1, 4, 5)
        >>> mapAB = sgrpy.graph.IndexMapping.from_seq((0, 3, 1, 0))
        >>> tuple(mapAB.permute(tup))
        (3, 5, 1, 3)
        """
        # apply map to sequence
        self_map = self._nodes_map
        return tuple(
            seq[x - numpy.uintp(1)] if x != 0 else None for x in self_map[1:]
        )

    def permute_not_none(self, seq: Sequence[_T]) -> Sequence[_T]:
        """Apply same operation as `self.permute()`, but error if `None`.

        Parameters
        ----------
        seq : Sequence[_T]
            Generic sequence of items.

        Returns
        -------
        Sequence[_T]
        """
        # apply map to sequence
        self_map = self._nodes_map[1:]
        if any(int(x) == 0 for x in self_map):
            raise ValueError("Failed 'Not None' condition")
        return tuple(seq[x - numpy.uintp(1)] for x in self_map)

    def as_tuple(self) -> tuple[None | int, ...]:
        """Convert mapping to a tuple.

        Returns
        -------
        tuple[None | int, ...]
        """
        return tuple(
            int(x) - 1 if x != 0 else None for x in self._nodes_map[1:]
        )

    def as_tuple_int(self) -> tuple[int, ...]:
        """Convert mapping to a tuple, raising error if `None`.

        Returns
        -------
        tuple[int, ...]
        """
        t = self.as_tuple()
        tc = tuple(x for x in t if x is not None)
        if len(tc) < len(t):
            raise ValueError("Map domain is not covered!")
        return tc

    def is_permutation(self) -> bool:
        """Check if mapping is a perfect permutation.

        A mapping is a perfect permutation if the domains of both A and B are
        of the same size and range from 0 to N.

        Returns
        -------
        bool
        """
        ideal_range = numpy.arange(self._nodes_map.size, dtype=numpy.uintp)
        return numpy.array_equal(numpy.sort(self._nodes_map), ideal_range)
