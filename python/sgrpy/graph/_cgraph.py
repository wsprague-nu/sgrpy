"""Colored graph type definitions."""

import dataclasses
import itertools
from collections.abc import Iterable, Sequence
from typing import final

import igraph

from ._bond import Bond
from ._canon_types import CanonType
from ._operations import AddNode, SubNode
from ._permutations import IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CGraph:
    """Colored graph (graph form).

    Complementary to `sgrpy.graph.SCGraph`. This form is not directly
    hashable  or comparable, though methods are implemented via
    `sgrpy.graph.SCGraph`. The strength of this method is that the graph is
    stored in native `igraph` form and therefore graph methods can be used to
    transform it.
    """

    _graph: igraph.Graph
    _colors: tuple[int, ...]

    @classmethod
    def empty(cls) -> "CGraph":
        """Initialize empty colored graph (no vertices).

        Returns both the colored graph and a blank index mapping.

        Returns
        -------
        CGraph
        """
        graph = igraph.Graph()
        colors_tuple: tuple[int, ...] = tuple()
        return CGraph(_graph=graph, _colors=colors_tuple)

    @classmethod
    def singlet(cls, color: int) -> "CGraph":
        """Initialize a colored graph with a single node.

        Returns the colored graph.

        Parameters
        ----------
        color : int
            Color of the single node in the graph.

        Returns
        -------
        CGraph
        """
        graph = igraph.Graph()
        graph.add_vertex()
        colors_tuple = (color,)
        return CGraph(_graph=graph, _colors=colors_tuple)

    @classmethod
    def from_tuples(
        cls,
        bonds: Iterable[tuple[int, int]],
        colors: Iterable[int],
    ) -> "CGraph":
        """Initialize a colored graph from a series of bonds and colors.

        The number of nodes is the minimum of the length of `colors` and the
        maximum index referred to by `bonds`. All indices should be expected to
        change.  Bonds are not given directionality, so source and target may be
        interchanged.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int]]
            Iterable of edges in the intended graph.
        colors : Iterable[int]
            Colors of the nodes in the graph.

        Returns
        -------
        CGraph
        """
        bond_gen = (Bond.new(src=src, trg=trg) for src, trg in bonds)
        return CGraph.from_bonds(bonds=bond_gen, colors=colors)

    @classmethod
    def from_bonds(
        cls,
        bonds: Iterable[Bond],
        colors: Iterable[int],
    ) -> "CGraph":
        """Initialize a colored graph from a series of bonds and colors.

        The number of nodes is the minimum of the length of `colors` and the
        maximum index referred to by `bonds`. All indices should be expected to
        change. The mapping representing the index changes is also returned.

        Parameters
        ----------
        bonds: Iterable[Bond]
            Iterable of edges in the intended graph.
        colors : Iterable[int]
            Colors of the nodes in the graph.

        Returns
        -------
        tuple[CGraph, IndexMapping]
        """
        colors_tuple = tuple(colors)
        num_vertices = len(colors_tuple)

        # create graph of correct size
        graph = igraph.Graph()
        graph.add_vertices(num_vertices)

        # enumerate bonds
        bonds_list = list(dict.fromkeys(bonds))
        if (
            len(bonds_list) > 0
            and max(max(b.as_tuple()) for b in bonds_list) >= num_vertices
        ):
            raise ValueError("Uncolored vertex present in bond list.")
        graph.add_edges(b.as_tuple() for b in bonds_list)
        return CGraph(_graph=graph, _colors=colors_tuple)

    @classmethod
    def from_scgraph(cls, scgraph: "SCGraph") -> "CGraph":
        """Convert a colored graph from string to graph form.

        Parameters
        ----------
        scgraph: SCGraph
            String-form colored graph to convert.

        Returns
        -------
        CGraph
        """
        # WARNING: this method assumes SCGraph has been correctly constructed
        bonds, colors = scgraph.to_split()

        # create graph of correct size
        graph = igraph.Graph()
        graph.add_vertices(len(colors))

        # enumerate bonds
        graph.add_edges(bonds)

        return CGraph(_graph=graph, _colors=colors)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CGraph):
            return False
        return self.to_scgraph() == other.to_scgraph()

    def __hash__(self) -> int:
        return hash(self.to_scgraph())

    def __len__(self) -> int:
        return len(self._colors)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_bonds("
            f"bonds=({','.join(repr(b) for b in self.get_bonds())}),"
            f"colors={self._colors})"
        )

    def nof_nodes(self) -> int:
        """Get number of nodes in graph.

        Returns
        -------
        int
        """
        return len(self._colors)

    def nof_edges(self) -> int:
        """Get number of edges/bonds in graph.

        Returns
        -------
        int
        """
        return len(self._graph.es)

    def to_scgraph(self) -> "SCGraph":
        """Convert from graph to hashable string container form.

        Returns
        -------
        SCGraph
        """
        return SCGraph.from_cgraph(self)

    def get_bonds(self) -> Iterable[Bond]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[Bond]
        """
        return (Bond.new(b.source, b.target) for b in self._graph.es)

    def get_colors(self) -> Sequence[int]:
        """Get graph node colors.

        Returns
        -------
        Sequence[int]
        """
        return self._colors

    def add_node(self, op: AddNode) -> tuple["CGraph", IndexMapping]:
        """Add node to CGraph, returning new CanonGraph and mapping.

        Parameters
        ----------
        op : AddNode
            Node addition operation.

        Returns
        -------
        tuple[CanonGraph, IndexMapping]
        """
        new_graph = self._graph.copy()
        new_colors = self._colors + (op.color,)
        i_new = len(self)
        new_graph.add_vertices(1)
        new_graph.add_edges((i, i_new) for i in op.connections)

        new_cgraph = CGraph(_graph=new_graph, _colors=new_colors)

        mapping = IndexMapping.identity(i_new).append(None)

        return new_cgraph, mapping

    def sub_node(self, op: SubNode) -> tuple["CGraph", IndexMapping]:
        """Subtract node from CGraph, returning new CGraph and mapping.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CGraph, IndexMapping]
        """
        new_graph = self._graph.copy()
        new_graph.delete_vertices(op.index)
        new_colors_l = list(self._colors)
        new_colors_l.pop(op.index)
        new_colors = tuple(new_colors_l)

        new_cgraph = CGraph(_graph=new_graph, _colors=new_colors)

        mapping = IndexMapping.identity_minus(len(self), op.index)

        return new_cgraph, mapping

    def permute(self, mapping: IndexMapping) -> "CGraph":
        """Permute node indices based on a mapping.

        Note: mapping must be a proper permutation or an error will be raised.

        Parameters
        ----------
        mapping : IndexMapping
            Mapping of relevant permutation to be applied.

        Returns
        -------
        CGraph
        """
        index_map = mapping
        index_map_intuple = index_map.inv().as_tuple()

        if any(x is None for x in index_map_intuple):
            raise ValueError("Mapping is not invertible")

        index_map_ituple = tuple(x for x in index_map_intuple if x is not None)

        new_colors = tuple(index_map.permute_not_none(self._colors))

        bonds = sorted(
            Bond.new(src=index_map_ituple[b.src], trg=index_map_ituple[b.trg])
            for b in self.get_bonds()
        )

        new_cgraph = CGraph.from_bonds(bonds, new_colors)

        return new_cgraph

    def canonize(self, method: CanonType) -> tuple["CGraph", IndexMapping]:
        """Canonize the colored graph.

        Indices are changed to canonize `self` according to the value of
        `method`. The mapping representing the index changes is also returned.
        If you are looking to obtain a `CanonGraph`, please use the
        method `CanonGraph.from_cgraph()` instead.

        Parameters
        ----------
        method : CanonType
            Type of canonization to use to form the canonical graph.

        Returns
        -------
        tuple[CGraph, IndexMapping]
        """
        self_colors = self._colors
        canonical_perm = self._graph.canonical_permutation(
            sh=str(method), color=self_colors
        )

        # index_map = IndexMapping.from_seq_inv(canonical_perm)
        index_map = IndexMapping.from_seq(canonical_perm)

        new_cgraph = self.permute(index_map)

        return new_cgraph, index_map

    def is_connected(self) -> bool:
        """Return `True` if graph has a single connected component.

        Returns
        -------
        bool
        """
        return bool(self._graph.is_connected())

    def summary(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Get summary tuple of colored graph.

        First entry in tuple is the edge list, second entry is the color list.

        Returns
        -------
        tuple[tuple[tuple[int, int], ...], tuple[int, ...]]
        """
        return self.to_scgraph().to_split()

    def nof_automorphisms(self, method: CanonType) -> int:
        """Calculate number of automorphisms of graph.

        Parameters
        ----------
        method : CanonType
            Canonization method to use for automorphism calculation.

        Returns
        -------
        int
        """
        return int(
            self._graph.count_automorphisms(sh=str(method), color=self._colors)
        )

    def nof_components(self) -> int:
        """Calculate number of (connected) components of graph.

        Returns
        -------
        int
        """
        return len(self._graph.components())

    def neighbors(self, vertex: int) -> tuple[int, ...]:
        """Get neighbor indices of a specific node.

        Returns
        -------
        int
        """
        return tuple(self._graph.neighbors(vertex))

    def get_parents(
        self,
    ) -> Iterable[tuple["CGraph", IndexMapping]]:
        """Get all direct "parents" of CGraph.

        Graph A is a "direct parent" of graph B if it has all the same nodes,
        colors, and connectivity as graph B except that it has one fewer nodes.

        Returns
        -------
        Iterable[tuple[CGraph, IndexMapping]]
        """
        for sub_i in range(len(self._colors)):
            sub_op = SubNode(sub_i)
            yield self.sub_node(sub_op)

    def subset(self, indices: Iterable[int]) -> tuple["CGraph", IndexMapping]:
        """Return a subset of `self` consisting of the indicated nodes.

        Also returns the mapping from `self` to the returned CGraph.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the nodes composing the subset of the graph.

        Returns
        -------
        tuple[CGraph, IndexMapping]
        """
        indices_to_get = sorted(frozenset(indices))
        sub_graph = self._graph.induced_subgraph(indices_to_get)
        self_colors = self._colors
        sub_colors = tuple(self_colors[i] for i in indices_to_get)

        new_cgraph = CGraph(_graph=sub_graph, _colors=sub_colors)

        mapping = IndexMapping.from_seq(indices_to_get)
        return new_cgraph, mapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class SCGraph:
    """Colored graph (string form).

    Complementary to `sgrpy.graph.SGraph`. This form is comparable, hashable,
    and (ideally) uses less memory than the graph form. It can also be easily
    encoded to a string.
    """

    _graph_s: str

    @classmethod
    def from_cgraph(cls, cgraph: CGraph) -> "SCGraph":
        """Convert colored graph from graph to string form.

        Parameters
        ----------
        cgraph: CGraph
            Graph-form CGraph to convert.

        Returns
        -------
        SCGraph
        """
        bond_str = ",".join(f"{b.src},{b.trg}" for b in cgraph.get_bonds())
        color_str = ",".join(str(color) for color in cgraph.get_colors())
        total_str = ";".join((bond_str, color_str))
        return SCGraph(_graph_s=total_str)

    @classmethod
    def from_str(cls, string: str) -> "SCGraph":
        """Initialize colored graph from a `str`.

        Parameters
        ----------
        string : str
            String from which to initialize the colored graph.
        checked : bool
            If True, perform check to ensure that graph is well-formed. This
            takes extra computing time but is worth it on unverified data since
            all methods of `SCraph` and `CGraph`
            assume a well-formed internal structure.

        Returns
        -------
        SCGraph
        """
        return SCGraph(_graph_s=string)

    def as_str(self) -> str:
        """Convert to string form.

        Returns
        -------
        str
        """
        return self._graph_s

    def __len__(self) -> int:
        if len(self._graph_s) == 0:
            return 0
        return len(self._graph_s.split(";", maxsplit=1)[1].split(","))

    def __lt__(self, other: "SCGraph") -> bool:
        self_split = self.to_split()
        self_split_rev = (
            tuple(reversed(self_split[1])),
            tuple(tuple(reversed(z)) for z in self_split[0]),
        )
        other_split = other.to_split()
        other_split_rev = (
            tuple(reversed(other_split[1])),
            tuple(tuple(reversed(z)) for z in other_split[0]),
        )

        return self_split_rev < other_split_rev

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_cgraph({repr(self.to_cgraph())})"
        )

    def to_split(
        self,
    ) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
        """Convert to tuples, as suited for a `from_tuples` initializer.

        Returns
        -------
        tuple[tuple[tuple[int, int], ...], tuple[int, ...]]
        """
        if len(self._graph_s) <= 1:
            return (tuple(), tuple())
        bond_str, labels_str = self.as_str().split(";", maxsplit=1)
        bonds = (
            (
                tuple(
                    (int(v1), int(v2))
                    for v1, v2 in itertools.batched(bond_str.split(","), 2)
                )
                if len(bond_str) > 0
                else tuple()
            )
            if len(bond_str) > 0
            else tuple()
        )
        colors: tuple[int, ...] = tuple(int(x) for x in labels_str.split(","))
        return bonds, colors

    def to_cgraph(self) -> "CGraph":
        """Convert to graph form.

        Returns
        -------
        CGraph
        """
        return CGraph.from_scgraph(self)
