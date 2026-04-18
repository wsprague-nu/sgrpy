"""Colored graph type definitions."""

import dataclasses
import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import final

import igraph
import numpy

from ._bond import TBond
from ._canon_types import CanonType
from ._cgraph import CGraph
from ._operations import AddNodeT, Extension, SubNode
from ._permutations import IndexMapping


def _select_other(edge: igraph.Edge, vertex_id: int) -> int:
    """Select other side of edge from current vertex."""
    result: int = edge.target if edge.source == vertex_id else edge.source
    return result


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CBGraph:
    """Colored bond graph (graph form).

    Complementary to `sgrpy.graph.CBGraphS`. This form is not directly
    hashable  or comparable, though methods are implemented via
    `sgrpy.graph.CBGraphS`. The strength of this method is that the graph is
    stored in native `igraph` form and therefore graph methods can be used to
    transform it.
    """

    _graph: igraph.Graph
    """Internal `igraph` object."""

    @classmethod
    def empty(cls) -> "CBGraph":
        """Initialize empty colored graph (no vertices).

        Returns the colored graph.

        Returns
        -------
        CBGraph
        """
        graph = igraph.Graph()
        return CBGraph(_graph=graph)

    @classmethod
    def singlet(cls, color: int) -> "CBGraph":
        """Initialize a colored graph with a single node.

        Returns the colored graph.

        Parameters
        ----------
        color : int
            Color of the single node in the graph.

        Returns
        -------
        CBGraph
        """
        graph = igraph.Graph()
        graph.add_vertex(typ=color)
        return CBGraph(_graph=graph)

    @classmethod
    def _from_bonds_fast(
        cls,
        bonds: Iterable[TBond[int]],
        colors: Sequence[int],
    ) -> "CBGraph":
        """Initialize from bonds (assume already ordered)."""
        bdict = {(b.src, b.trg): b.color for b in bonds}

        # create graph of correct size
        graph = igraph.Graph(n=len(colors), edges=list(bdict.keys()))
        graph.vs["typ"] = colors
        graph.es["typ"] = list(bdict.values())

        return CBGraph(_graph=graph)

    @classmethod
    def from_bonds(
        cls,
        bonds: Iterable[TBond[int]],
        colors: Sequence[int],
    ) -> "CBGraph":
        """Initialize a colored graph from a series of bonds and colors.

        The number of nodes is the length of `colors`.  Indices should be from 0
        to 1 - len(colors).

        Parameters
        ----------
        bonds: Iterable[TBond[int]]
            Iterable of edges in the intended graph.
        colors : Sequence[int]
            Colors of the nodes in the graph.

        Returns
        -------
        CBGraph
        """
        canon_bonds = (b.canon() for b in bonds)
        new_cbgraph = CBGraph._from_bonds_fast(canon_bonds, colors)

        return new_cbgraph

    @classmethod
    def from_tuples(
        cls,
        bonds: Iterable[tuple[int, int, int]],
        colors: Sequence[int],
    ) -> "CBGraph":
        """Initialize a colored graph from a series of bonds and colors.

        The number of nodes is the length of `colors`.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int, int]]
            Iterable of edges in the intended graph, with their color attached.
        colors : Sequence[int]
            Colors of the nodes in the graph.

        Returns
        -------
        CBGraph
        """
        return CBGraph.from_bonds(
            (TBond.from_tuple(bond) for bond in bonds), colors
        )

    @classmethod
    def from_cbgraphs(cls, CBGraphS: "CBGraphS") -> "CBGraph":
        """Convert a colored graph from string to graph form.

        Parameters
        ----------
        CBGraphS: CBGraphS
            String-form colored graph to convert.

        Returns
        -------
        CBGraph
        """
        # WARNING: this assumes that the CBGraphS is correctly constructed!
        bond_tups, colors = CBGraphS.to_split()

        # create graph of correct size
        return CBGraph.from_tuples(bond_tups, colors)

    @classmethod
    def from_dict(
        cls,
        bonds: Iterable[tuple[int, int, int]],
        nodes: Iterable[tuple[int, int]],
    ) -> tuple["CBGraph", IndexMapping]:
        """Initialize a colored graph from a series of bonds and colors.

        The number of nodes is the length of `colors`. All indices should be
        expected to change in order to compact the node indices. The mapping
        representing these index changes is returned. Bonds are not given
        directionality, so source and target may be interchanged.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int, int]]
            Iterable of edges in the intended graph.  Should be in order of
            (source, target, color).
        nodes : Iterable[tuple[int, int]]
            Nodes in the graph.  Should be in order of
            (index, color).

        Returns
        -------
        CBGraph
        """
        idx_remap: dict[int, int] = {}
        rev_idxmap: list[int] = []
        nc_list: list[int] = []
        for new_idx, (old_idx, nc) in enumerate(nodes):
            rev_idxmap.append(old_idx)
            nc_list.append(nc)
            idx_remap[old_idx] = new_idx

        bonds_remap = (
            (idx_remap[src], idx_remap[trg], ec) for src, trg, ec in bonds
        )

        new_cbgraph = CBGraph.from_tuples(bonds_remap, nc_list)

        idxmap = IndexMapping.from_seq(rev_idxmap)

        return new_cbgraph, idxmap

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CBGraph):
            return False
        return self.to_cbgraphs() == other.to_cbgraphs()

    def __repr__(self) -> str:
        all_colors = self.get_colors()
        all_bonds = self.get_bonds()
        return (
            f"{self.__class__.__name__}"
            f".from_tuples({[b.as_tuple() for b in all_bonds]}, {all_colors})"
        )

    def __hash__(self) -> int:
        return hash(self.to_cbgraphs())

    def nof_nodes(self) -> int:
        """Get number of nodes in graph.

        Returns
        -------
        int
        """
        return len(self._graph.vs)

    def nof_edges(self) -> int:
        """Get number of edges/bonds in graph.

        Returns
        -------
        int
        """
        return len(self._graph.es)

    def to_cbgraphs(self) -> "CBGraphS":
        """Convert from graph to hashable string container form.

        Returns
        -------
        CBGraphS
        """
        return CBGraphS.from_cgraph(self)

    def get_bonds(self) -> Iterable[TBond[int]]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[TBond[int]]
        """
        return (
            (
                TBond.new(b.source, b.target, bc)
                for (b, bc) in zip(
                    self._graph.es, self._graph.es["typ"], strict=True
                )
            )
            if len(self._graph.es) > 0
            else ()
        )

    def get_colors(self) -> Sequence[int]:
        """Get graph node colors.

        Returns
        -------
        Sequence[int]
        """
        if len(self._graph.vs) == 0:
            return ()
        vertex_colors: list[int] = self._graph.vs["typ"]
        return vertex_colors

    def get_edge_colors(self) -> Sequence[int]:
        """Get graph edge colors.

        Returns
        -------
        Sequence[int]
        """
        if len(self._graph.es) == 0:
            return ()
        edge_colors: list[int] = self._graph.es["typ"]
        return edge_colors

    def add_node(
        self, op: AddNodeT[int, int]
    ) -> tuple["CBGraph", IndexMapping]:
        """Add node to CBGraph, returning new CBGraph and mapping.

        Parameters
        ----------
        op : AddNodeT[int,int]
            Node addition operation.

        Returns
        -------
        tuple[CBGraph, IndexMapping]
        """
        new_graph = self._graph.copy()

        new_graph.add_vertex(typ=op.typ)
        i_new = len(self._graph.vs)
        new_graph.add_edges(
            ((ext.idx, i_new) for ext in op.ext),
            attributes={"typ": [ext.typ for ext in op.ext]},
        )

        new_cgraph = CBGraph(_graph=new_graph)

        mapping = IndexMapping.identity(i_new).append(None)

        return new_cgraph, mapping

    def sub_node(self, op: SubNode) -> tuple["CBGraph", IndexMapping]:
        """Subtract node from CBGraph, returning new CBGraph and mapping.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CBGraph, IndexMapping]
        """
        new_graph = self._graph.copy()
        new_graph.delete_vertices(op.index)

        new_cgraph = CBGraph(_graph=new_graph)

        mapping = IndexMapping.identity_minus(self.nof_nodes(), op.index)

        return new_cgraph, mapping

    def sub_node_info(
        self, op: SubNode
    ) -> tuple["CBGraph", IndexMapping, AddNodeT[int, int]]:
        """Subtract node from CBGraph, return new CBGraph plus additional info.

        Generally the same as `.sub_node` method, except that the inverse
        `AddNodeT` operation is also returned.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CBGraph, IndexMapping, AddNodeT[int, int]]
        """
        new_graph = self._graph.copy()

        # obtain partial inverse operation
        target_vert: igraph.Vertex = new_graph.vs[op.index]
        vert_color: int = target_vert["typ"]

        extensions = (
            Extension(_select_other(edge, op.index), edge["typ"])
            for edge in target_vert.all_edges()
        )

        add_op: AddNodeT[int, int] = AddNodeT.new(extensions, vert_color)

        new_graph.delete_vertices(op.index)

        new_cgraph = CBGraph(_graph=new_graph)

        mapping = IndexMapping.identity_minus(self.nof_nodes(), op.index)

        add_op_perm = add_op.permute(mapping)

        return new_cgraph, mapping, add_op_perm

    def permute(self, mapping: IndexMapping) -> "CBGraph":
        """Permute node indices based on a mapping.

        Note: mapping must be complete or an error will be raised.

        Parameters
        ----------
        mapping : IndexMapping
            Mapping of relevant permutation to be applied.

        Returns
        -------
        CBGraph
        """
        index_map = mapping

        if not index_map.is_permutation():
            raise ValueError("Mapping is not a perfect permutation!")

        index_map_tup = index_map.as_tuple_int()

        new_graph = self._graph.permute_vertices(index_map_tup)

        return CBGraph(_graph=new_graph)

    def shuffle(
        self, seed: int | None = None
    ) -> tuple["CBGraph", IndexMapping]:
        """Permute node indices randomly.

        Parameters
        ----------
        seed : int | None
            Random seed (optional).

        Returns
        -------
        tuple[CBGraph, IndexMapping]
        """
        rng = numpy.random.default_rng(seed)
        perm_base = numpy.arange(self.nof_nodes(), dtype=numpy.uintp)
        rng.shuffle(perm_base)

        new_graph = self._graph.permute_vertices(perm_base)
        mapping = IndexMapping.from_seq(perm_base.tolist())

        return CBGraph(_graph=new_graph), mapping

    def _convert_cgraph(self) -> CGraph:
        nc = len(self._graph.vs)
        if nc == 0:
            return CGraph.empty()

        vcolors: list[int] = self._graph.vs["typ"]
        ecolors: list[int] = (
            self._graph.es["typ"] if len(self._graph.es) > 0 else []
        )
        max_vcolor = max(vcolors)
        all_colors = vcolors + [ec + max_vcolor for ec in ecolors]

        all_subedges: Iterable[tuple[int, int]] = itertools.chain.from_iterable(
            ((e.source, nc + e_i), (nc + e_i, e.target))
            for e_i, e in enumerate(self._graph.es)
        )

        cg = CGraph.from_tuples(all_subedges, all_colors)
        return cg

    def reorder_edges(self) -> "CBGraph":
        """Reorder edges (not indices) to produce canonical ordering."""
        bonds_sorted = sorted(self.get_bonds())
        new_cbgraph = CBGraph._from_bonds_fast(bonds_sorted, self.get_colors())
        return new_cbgraph

    def canonize(self, method: CanonType) -> tuple["CBGraph", IndexMapping]:
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
        # obtain converted CGraph
        # (bonds become colored nodes with incremented colors)
        self_cg = self._convert_cgraph()

        # canonize converted CGraph and obtain index map
        canon_map = self_cg.canonize(method)[1]

        # extract relevant components (vertices) from mapping
        num_nodes = len(self._graph.vs)
        perm_reduc = [x for x in canon_map.as_tuple_int() if x < num_nodes]
        final_map = IndexMapping.from_seq(perm_reduc)

        # apply map to self to reorder vertices
        new_cbgraph = self.permute(final_map)

        # reorder edges to canonical ordering
        new_cbgraph_final = new_cbgraph.reorder_edges()

        return new_cbgraph_final, final_map

    def is_connected(self) -> bool:
        """Return `True` if graph has a single connected component.

        Returns
        -------
        bool
        """
        return bool(self._graph.is_connected())

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
        self_cg = self._convert_cgraph()
        return int(
            self_cg._graph.count_automorphisms(
                sh=str(method), color=self_cg.get_colors()
            )
        )

    def nof_components(self) -> int:
        """Calculate number of (connected) components of graph.

        Returns
        -------
        int
        """
        return len(self._graph.components())

    def all_components(self) -> Iterable[tuple["CBGraph", IndexMapping]]:
        """Return all connected components of graph.

        Returns
        -------
        Iterable[tuple[CBGraph, IndexMapping]]
        """
        components: list[list[int]] = list(self._graph.components())
        return (self.subset(component) for component in components)

    def neighbors(self, vertex: int) -> tuple[int, ...]:
        """Get neighbor indices of a specific node.

        Returns
        -------
        int
        """
        return tuple(self._graph.neighbors(vertex))

    def neighbors_ext(self, vertex: int) -> Iterable[Extension[int]]:
        """Get neighbor extensions toward `vertex`.

        Parameters
        ----------
        vertex : int
            Index of destination vertex.

        Returns
        -------
        Iterable[Extension[int]]
            Iterable of extensions approaching `vertex`.
        """
        ivertex: igraph.Vertex = self._graph.vs[vertex]
        ext_gen = (
            Extension(idx=_select_other(edge, vertex), typ=edge["typ"])
            for edge in ivertex.all_edges()
        )
        return ext_gen

    def get_parents(
        self,
    ) -> Iterable[tuple["CBGraph", IndexMapping, AddNodeT[int, int]]]:
        """Get all direct "parents" of CBGraph.

        Graph A is a "direct parent" of graph B if it has all the same nodes,
        colors, and connectivity as graph B except that it has one fewer nodes.

        Returns
        -------
        Iterable[tuple[CBGraph, IndexMapping, AddNodeT[int, int]]]
        """
        for sub_i in range(len(self._graph)):
            sub_op = SubNode(sub_i)
            yield self.sub_node_info(sub_op)

    def subset(self, indices: Iterable[int]) -> tuple["CBGraph", IndexMapping]:
        """Return a subset of `self` consisting of the indicated nodes.

        Also returns the mapping from `self` to the returned CBGraph.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the nodes composing the subset of the graph.

        Returns
        -------
        tuple[CBGraph, IndexMapping]
        """
        indices_to_get = sorted(frozenset(indices))
        sub_graph = self._graph.induced_subgraph(indices_to_get)

        new_cgraph = CBGraph(_graph=sub_graph)

        mapping = IndexMapping.from_seq(indices_to_get)
        return new_cgraph, mapping

    def recolor(
        self,
        vertmap: Mapping[int, int] | None,
        edgemap: Mapping[int, int] | None,
    ) -> "CBGraph":
        """Destructively recolor CBGraph (reassign node and vertex colors).

        Parameters
        ----------
        vertmap : Mapping[int, int] | None
            Mapping from old node colors to new node colors; None if no changes.
        edgemap : Mapping[int, int] | None
            Mapping from old edge colors to new edge colors; None if no changes.
        """
        gcopy = self._graph.copy()
        if vertmap is not None:
            gcopy.vs["typ"] = [vertmap[c] for c in gcopy.vs["typ"]]
        if edgemap is not None:
            gcopy.es["typ"] = [edgemap[c] for c in gcopy.es["typ"]]
        return CBGraph(_graph=gcopy)

    def __lt__(self, __value: "CBGraph") -> bool:
        return self.to_cbgraphs() < __value.to_cbgraphs()


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class CBGraphS:
    """Colored graph (string form).

    Complementary to `sgrpy.graph.CBGraph`. This form is comparable, hashable,
    and (ideally) uses less memory than the graph form. It can also be easily
    encoded to a string.
    """

    _graph_s: str

    @classmethod
    def from_cgraph(cls, cbgraph: CBGraph) -> "CBGraphS":
        """Convert colored graph from graph to string form.

        Parameters
        ----------
        cbgraph: CBGraph
            Graph-form CBGraph to convert.

        Returns
        -------
        CBGraphS
        """
        bond_str = ",".join(
            f"{b.src},{b.trg},{b.color}" for b in cbgraph.get_bonds()
        )
        color_str = ",".join(str(color) for color in cbgraph.get_colors())
        total_str = ";".join((bond_str, color_str))
        return CBGraphS(_graph_s=total_str)

    @classmethod
    def from_str(cls, string: str) -> "CBGraphS":
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
        return CBGraphS(_graph_s=string)

    def as_str(self) -> str:
        """Convert to string form.

        Returns
        -------
        str
        """
        return self._graph_s

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_cgraph({repr(self.to_cbgraph())})"
        )

    def to_split(
        self,
    ) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
        """Convert to tuples, as suited for a `from_tuples` initializer.

        Returns
        -------
        tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]
        """
        if len(self._graph_s) <= 1:
            return (tuple(), tuple())
        bond_str, labels_str = self.as_str().split(";", maxsplit=1)
        bonds = (
            (
                tuple(
                    (int(v1), int(v2), int(v3))
                    for v1, v2, v3 in itertools.batched(bond_str.split(","), 3)
                )
                if len(bond_str) > 0
                else tuple()
            )
            if len(bond_str) > 0
            else tuple()
        )
        colors: tuple[int, ...] = tuple(int(x) for x in labels_str.split(","))
        return bonds, colors

    def to_cbgraph(self) -> "CBGraph":
        """Convert to graph form.

        Returns
        -------
        CBGraph
        """
        return CBGraph.from_cbgraphs(self)
