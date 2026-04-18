"""Canonical graph type definitions."""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import final

from ._bond import Bond
from ._canon_types import CanonType
from ._cgraph import CGraph, SCGraph
from ._operations import AddNode, SubNode
from ._permutations import IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CanonGraph:
    """Canonized colored graph (graph form).

    Complementary to `SCanonGraph`. This form is not directly
    hashable  or comparable, though methods are implemented via
    `SCanonGraph`. The strength of this method is that the graph is
    stored in native `igraph` form and therefore graph methods can be used to
    transform it.
    """

    _cgraph: CGraph
    _method: CanonType

    @classmethod
    def empty(cls, method: CanonType) -> tuple["CanonGraph", IndexMapping]:
        """Initialize empty canonical graph (no vertices).

        Returns both the canonical graph and a blank index mapping.

        Parameters
        ----------
        method : CanonType
            Type of canonization to associate with this canonical graph.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        return CanonGraph.from_cgraph(CGraph.empty(), method=method)

    @classmethod
    def singlet(
        cls, color: int, method: CanonType
    ) -> tuple["CanonGraph", IndexMapping]:
        """Initialize a canonical graph with a single node.

        Returns both the canonical graph and an index mapping.

        Parameters
        ----------
        color : int
            Color of the single node in the graph.
        method : CanonType
            Type of canonization to associate with this canonical graph.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        return CanonGraph.from_cgraph(CGraph.singlet(color), method=method)

    @classmethod
    def from_bonds(
        cls,
        bonds: Iterable[Bond],
        colors: Iterable[int],
        method: CanonType,
    ) -> tuple["CanonGraph", IndexMapping]:
        """Initialize a canonical graph from a series of bonds and colors.

        The number of nodes is the minimum of the length of `colors` and the
        maximum index referred to by `bonds`. Since the resulting
        graph will be canonical, all indices should be expected to change. The
        mapping representing the index changes is also returned.

        Parameters
        ----------
        bonds: Iterable[Bond]
            Iterable of edges in the intended graph.
        colors : Iterable[int]
            Colors of the nodes in the graph.
        method : CanonType
            Type of canonization to associate with this canonical graph.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        cgraph = CGraph.from_bonds(bonds=bonds, colors=colors)
        cgraph_canon, indexmap = cgraph.canonize(method=method)
        return CanonGraph(_cgraph=cgraph_canon, _method=method), indexmap

    @classmethod
    def from_cgraph(
        cls, cgraph: CGraph, method: CanonType
    ) -> tuple["CanonGraph", IndexMapping]:
        """Initialize a canonical graph from a colored graph.

        Indices are changed to canonize `cgraph` according to the value of
        `method`. The mapping representing the index changes is also returned.

        Parameters
        ----------
        cgraph: CGraph
            Colored graph to canonize.
        method : CanonType
            Type of canonization to use to form this canonical graph.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        cgraph_canon, indexmap = cgraph.canonize(method=method)
        return CanonGraph(_cgraph=cgraph_canon, _method=method), indexmap

    @classmethod
    def from_scanongraph(cls, scanongraph: "SCanonGraph") -> "CanonGraph":
        """Initialize a canonical graph from its string form.

        Parameters
        ----------
        scanongraph: SCanonGraph
            String-form CanonGraph to convert.

        Returns
        -------
        CanonGraph
        """
        cgraph = CGraph.from_scgraph(scanongraph.scgraph)
        method = scanongraph.method
        canongraph = CanonGraph(_cgraph=cgraph, _method=method)
        return canongraph

    @property
    def cgraph(self) -> "CGraph":
        """CGraph : Colored graph in graph form.

        Returns
        -------
        CGraph
        """
        return self._cgraph

    @property
    def method(self) -> CanonType:
        """CanonType : Method used to canonize the graph.

        Returns
        -------
        sgrpy.graph.CanonType
        """
        return self._method

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CanonGraph):
            return False
        return self.to_scanongraph() == other.to_scanongraph()

    def __hash__(self) -> int:
        return hash(self.to_scanongraph())

    def __len__(self) -> int:
        return len(self.cgraph)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_bonds("
            f"bonds=({','.join(repr(b) for b in self.cgraph.get_bonds())}),"
            f"colors={self.cgraph._colors},"
            f"method={self.method})"
        )

    def nof_nodes(self) -> int:
        """Get number of nodes in graph."""
        return self.cgraph.nof_nodes()

    def nof_edges(self) -> int:
        """Get number of edges/bonds in graph."""
        return self.cgraph.nof_edges()

    def to_scanongraph(self) -> "SCanonGraph":
        """Convert to string form.

        Returns
        -------
        SCanonGraph
        """
        return SCanonGraph.from_canongraph(self)

    def get_bonds(self) -> Iterable[Bond]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[Bond]
        """
        return self.cgraph.get_bonds()

    def get_colors(self) -> Sequence[int]:
        """Get graph node colors.

        Returns
        -------
        Sequence[int]
        """
        return self.cgraph._colors

    def summary(self) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Get summary tuple of canonized graph."""
        return self.to_scanongraph().to_split()

    def nof_automorphisms(self) -> int:
        """Calculate number of automorphisms of graph."""
        return self._cgraph.nof_automorphisms(method=self._method)

    def add_node(self, op: AddNode) -> tuple["CanonGraph", IndexMapping]:
        """Add node to CanonGraph, returning new CanonGraph and mapping.

        Parameters
        ----------
        op : AddNode
            Node addition operation.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        new_cgraph, new_cmap = self._cgraph.add_node(op)
        new_cangraph, canon_map = CanonGraph.from_cgraph(
            cgraph=new_cgraph, method=self._method
        )
        total_map = canon_map.compose(new_cmap)
        return new_cangraph, total_map

    def sub_node(self, op: SubNode) -> tuple["CanonGraph", IndexMapping]:
        """Subtract node from CanonGraph, returning new CanonGraph and mapping.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        new_cgraph, new_cmap = self._cgraph.sub_node(op)
        new_cangraph, canon_map = CanonGraph.from_cgraph(
            cgraph=new_cgraph, method=self._method
        )
        total_map = canon_map.compose(new_cmap)
        return new_cangraph, total_map

    def get_parents(
        self,
    ) -> Iterable[tuple["CanonGraph", IndexMapping]]:
        """Get all direct "parents" of CanonGraph.

        Graph A is a "direct parent" of graph B if it has all the same nodes,
        colors, and connectivity as graph B except that it has one fewer nodes.

        Returns
        -------
        Iterable[tuple[CanonGraph, IndexMapping]]
        """
        for sub_i in range(len(self._cgraph)):
            sub_op = SubNode(sub_i)
            yield self.sub_node(sub_op)

    def verify(self) -> bool:
        """Verify that graph is properly canonized.

        Should always be true. If this is False, it is likely because the graph
        was loaded from an incorrectly formatted string form. Try making sure
        you are using the same version of `sgrpy` as when your data was
        generated.

        Returns
        -------
        bool
        """
        return self == CanonGraph.from_cgraph(self._cgraph, self._method)

    def subset(
        self, indices: Iterable[int]
    ) -> tuple["CanonGraph", IndexMapping]:
        """Return a subset of `self` consisting of the indicated nodes.

        Also returns the mapping from `self` to the returned CanonGraph.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the nodes composing the subset of the graph.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        new_graph1, new_map1 = self._cgraph.subset(indices=indices)
        new_graph2, new_map2 = new_graph1.canonize(method=self._method)
        total_map = new_map2.compose(new_map1)

        new_canongraph = CanonGraph(_cgraph=new_graph2, _method=self._method)

        return new_canongraph, total_map


@final
@dataclasses.dataclass(eq=True, frozen=True, order=True, slots=True)
class SCanonGraph:
    """Canonized colored graph (string form).

    Complementary to `SCanonGraph`. This form is comparable,
    hashable, and (ideally) uses less memory than the graph form. It can also be
    easily encoded to a string.
    """

    _graph_sc: SCGraph
    _method: CanonType

    @classmethod
    def from_canongraph(cls, canongraph: CanonGraph) -> "SCanonGraph":
        """Convert canonical graph from graph to string form.

        Parameters
        ----------
        canongraph: CanonGraph
            Graph-form CanonGraph to convert.

        Returns
        -------
        SCanonGraph
        """
        scgraph = SCGraph.from_cgraph(canongraph.cgraph)
        return SCanonGraph(_graph_sc=scgraph, _method=canongraph.method)

    @classmethod
    def from_str(cls, string: str, checked: bool) -> "SCanonGraph":
        """Initialize canonical graph from a `str`.

        Parameters
        ----------
        string : str
            String from which to initialize the canonical graph.
        checked : bool
            If True, perform check to ensure that string is well-formed. This
            takes extra computing time but is worth it on unverified data since
            all methods of `SCanonGraph` and
            `CanonGraph` assume proper canonicity.

        Returns
        -------
        SCanonGraph
        """
        method_str, scgraph_str = string.split(";", maxsplit=1)
        method = CanonType(method_str)
        scgraph = SCGraph.from_str(scgraph_str)
        scanongraph = SCanonGraph(_graph_sc=scgraph, _method=method)
        if checked and not scanongraph.verify():
            raise ValueError(f"Invalid string {string}")
        return scanongraph

    @property
    def method(self) -> CanonType:
        """CanonType : Method used to canonize the graph.

        Returns
        -------
        CanonType
        """
        return self._method

    @property
    def scgraph(self) -> SCGraph:
        """SCGraph : Colored graph in string form.

        Returns
        -------
        SCGraph
        """
        return self._graph_sc

    def as_str(self) -> str:
        """Convert to string form.

        Returns
        -------
        str
        """
        return ";".join((str(self._method), self._graph_sc.as_str()))

    def __len__(self) -> int:
        return len(self._graph_sc)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f".from_canongraph({repr(self.to_canongraph())})"
        )

    def to_split(self) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Convert to tuples, as suited for a `from_tuples` initializer.

        Returns
        -------
        tuple[tuple[tuple[int, int], ...], tuple[int, ...]]
        """
        return self._graph_sc.to_split()

    def to_canongraph(self) -> "CanonGraph":
        """Convert to graph form.

        Returns
        -------
        CanonGraph
        """
        return CanonGraph.from_scanongraph(self)

    def verify(self) -> bool:
        """Verify that graph is properly canonized.

        Should always be true. If this is False, it is likely because the graph
        was loaded from an incorrectly formatted string form. Try making sure
        you are using the same version of `sgrpy` as when your data was
        generated.

        Returns
        -------
        bool
        """
        scanon_dup = CanonGraph.from_cgraph(
            self.to_canongraph().cgraph, method=self._method
        )[0].to_scanongraph()

        return self == scanon_dup
