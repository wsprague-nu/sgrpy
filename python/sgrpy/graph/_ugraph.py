"""Uncolored graph type definitions."""

import dataclasses
import itertools
import json
from collections.abc import Iterable, Sequence
from typing import final

import igraph

from ._bond import Bond


@final
@dataclasses.dataclass(frozen=True, slots=True)
class UGraph:
    """String-labeled graph (graph form).

    Complementary to `sgrpy.graph.SUGraph`. This form is not directly hashable
    or comparable, though methods are implemented via `sgrpy.graph.SUGraph`. The
    strength of the graph form is that the graph is stored in native `igraph`
    form and therefore graph methods can be used to transform it.
    """

    _graph: igraph.Graph
    _labels: tuple[str, ...]

    @classmethod
    def from_tuples(
        cls,
        bonds: Iterable[tuple[int, int]],
        labels: Iterable[str],
    ) -> "UGraph":
        """Initialize a string-labeled graph from a series of bonds and labels.

        The number of nodes is the minimum of the length of `labels` and the
        maximum index referred to by `bonds`. All indices should be expected to
        change.
        Bonds are not given directionality, so source and target may be
        interchanged.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int]]
            Iterable of edges in the intended graph.
        labels : Iterable[str]
            Labels of the nodes in the graph.

        Returns
        -------
        UGraph
        """
        return UGraph.from_bonds(
            (Bond.new(src=src, trg=trg) for src, trg in bonds), labels=labels
        )

    @classmethod
    def from_bonds(
        cls,
        bonds: Iterable[Bond],
        labels: Iterable[str],
    ) -> "UGraph":
        """Initialize a string-labeled graph from a series of bonds and labels.

        The number of nodes is the minimum of the length of `labels` and the
        maximum index referred to by `bonds`. All indices should be expected to
        change.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int]]
            Iterable of edges in the intended graph.
        labels : Iterable[str]
            Labels of the nodes in the graph.

        Returns
        -------
        UGraph
        """
        labels_tuple = tuple(labels)
        num_vertices = len(labels_tuple)

        # create graph of correct size
        graph = igraph.Graph()
        graph.add_vertices(num_vertices)

        # enumerate bonds
        bonds_list = list(bonds)
        if len(bonds_list) == 0:
            return UGraph(_graph=graph, _labels=labels_tuple)
        if max(max(b.as_tuple()) for b in bonds_list) >= num_vertices:
            raise ValueError("Unlabelled vertex present in bond list.")
        graph.add_edges(b.as_tuple() for b in bonds_list)
        return UGraph(_graph=graph, _labels=labels_tuple)

    @classmethod
    def from_sugraph(cls, sugraph: "SUGraph") -> "UGraph":
        """Convert a string-labeled graph from string to graph form.

        Parameters
        ----------
        sugraph: SUGraph
            String-form ugraph to convert.

        Returns
        -------
        UGraph
        """
        # WARNING: this method assumes SUGraph has been correctly constructed
        bonds, labels = sugraph.to_split()

        # create graph of correct size
        graph = igraph.Graph()
        graph.add_vertices(len(labels))

        # enumerate bonds
        graph.add_edges(bonds)

        return UGraph(_graph=graph, _labels=labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UGraph):
            return False
        return self.to_sugraph() == other.to_sugraph()

    def __hash__(self) -> int:
        return hash(self.to_sugraph())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_bonds("
            f"bonds=({','.join(repr(b) for b in self.get_bonds())}),"
            f"labels={self._labels})"
        )

    def nof_components(self) -> int:
        """Calculate number of components of graph."""
        return len(self._graph.components())

    def nof_nodes(self) -> int:
        """Get number of vertices in graph."""
        return len(self._graph.vs)

    def to_sugraph(self) -> "SUGraph":
        """Convert from graph to string form.

        Returns
        -------
        SUGraph
        """
        return SUGraph.from_ugraph(self)

    def get_bonds(self) -> Iterable[Bond]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[Bond]
        """
        return (Bond.new(b.source, b.target) for b in self._graph.es)

    def get_labels(self) -> Sequence[str]:
        """Get graph node labels.

        Returns
        -------
        Sequence[str]
        """
        return self._labels


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class SUGraph:
    """String-labeled graph (string form).

    Complementary to `sgrpy.graph.UGraph`. This form is comparable, hashable,
    and (ideally) uses less memory than the graph form. It can also be easily
    encoded to a string.
    """

    _graph_s: str

    @classmethod
    def from_ugraph(cls, ugraph: UGraph) -> "SUGraph":
        """Convert string-labeled graph from graph to string form.

        Parameters
        ----------
        ugraph: UGraph
            Graph-form UGraph to convert.

        Returns
        -------
        SUGraph
        """
        if ugraph.nof_nodes() == 0:
            return SUGraph(_graph_s="")
        bond_str = ",".join(f"{b.src},{b.trg}" for b in ugraph.get_bonds())
        labels = list(ugraph.get_labels())
        label_str = json.dumps(labels, separators=(",", ":"))
        total_str = ";".join((bond_str, label_str))
        return SUGraph(_graph_s=total_str)

    @classmethod
    def from_str(cls, string: str) -> "SUGraph":
        """Initialize colored graph from a `str`.

        Parameters
        ----------
        string : str
            String from which to initialize the string-labeled graph.

        Returns
        -------
        SUGraph
        """
        return SUGraph(_graph_s=string)

    def as_str(self) -> str:
        """Convert to string."""
        return self._graph_s

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_ugraph({repr(self.to_ugraph())})"
        )

    def to_split(self) -> tuple[tuple[tuple[int, int], ...], tuple[str, ...]]:
        """Convert to tuples, as suited for a `from_tuples` initializer.

        Returns
        -------
        tuple[tuple[tuple[int, int], ...], tuple[int, ...]]
        """
        if self.as_str() == "":
            return (), ()
        bond_str, labels_str = self.as_str().split(";", maxsplit=1)
        bonds = (
            tuple(
                (int(v1), int(v2))
                for v1, v2 in itertools.batched(bond_str.split(","), 2)
            )
            if len(bond_str) > 0
            else tuple()
        )
        labels: tuple[str, ...] = tuple(json.loads(labels_str))
        return bonds, labels

    def to_ugraph(self) -> "UGraph":
        """Convert to graph form.

        Returns
        -------
        UGraph
        """
        return UGraph.from_sugraph(self)
