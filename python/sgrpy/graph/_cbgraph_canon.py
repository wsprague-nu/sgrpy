"""Canonical graph type definitions."""

import dataclasses
from collections.abc import Iterable, Sequence
from typing import final

from ._bond import TBond
from ._canon_types import CanonType
from ._cbgraph import CBGraph, CBGraphS
from ._operations import AddNodeT, SubNode
from ._permutations import IndexMapping


@final
@dataclasses.dataclass(frozen=True, slots=True)
class CBGraphC:
    """Canonized colored graph (graph form).

    Complementary to `CBGraphCS`. This form is not directly
    hashable  or comparable, though methods are implemented via
    `CBGraphCS`. The strength of this method is that the graph is
    stored in native `igraph` form and therefore graph methods can be used to
    transform it.
    """

    _cbgraph: CBGraph
    _method: CanonType

    @classmethod
    def empty(cls, method: CanonType) -> tuple["CBGraphC", IndexMapping]:
        """Initialize empty canonical graph (no vertices).

        Returns both the canonical graph and a blank index mapping.

        Parameters
        ----------
        method : CanonType
            Type of canonization to associate with this canonical graph.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        return CBGraphC.from_cbgraph(CBGraph.empty(), method=method)

    @classmethod
    def singlet(
        cls, color: int, method: CanonType
    ) -> tuple["CBGraphC", IndexMapping]:
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
        tuple[CBGraphC, IndexMapping]
        """
        return CBGraphC.from_cbgraph(CBGraph.singlet(color), method=method)

    @classmethod
    def from_bonds(
        cls,
        bonds: Iterable[TBond[int]],
        colors: Sequence[int],
        method: CanonType,
    ) -> tuple["CBGraphC", IndexMapping]:
        """Initialize a canonical graph from a series of bonds and colors.

        The number of nodes is the minimum of the length of `colors` and the
        maximum index referred to by `bonds`. Since the resulting
        graph will be canonical, all indices should be expected to change. The
        mapping representing the index changes is also returned.

        Parameters
        ----------
        bonds: Iterable[Bond]
            Iterable of edges in the intended graph.
        colors : Sequence[int]
            Colors of the nodes in the graph.
        method : CanonType
            Type of canonization to associate with this canonical graph.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        cgraph = CBGraph.from_bonds(bonds=bonds, colors=colors)
        cgraph_canon, indexmap = cgraph.canonize(method=method)
        return CBGraphC(_cbgraph=cgraph_canon, _method=method), indexmap

    @classmethod
    def from_cbgraph(
        cls, cbgraph: CBGraph, method: CanonType
    ) -> tuple["CBGraphC", IndexMapping]:
        """Initialize a canonical graph from a colored graph.

        Indices are changed to canonize `cbgraph` according to the value of
        `method`. The mapping representing the index changes is also returned.

        Parameters
        ----------
        cbgraph: CBGraph
            Colored graph to canonize.
        method : CanonType
            Type of canonization to use to form this canonical graph.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        cgraph_canon, indexmap = cbgraph.canonize(method=method)
        return CBGraphC(_cbgraph=cgraph_canon, _method=method), indexmap

    @classmethod
    def from_sform(cls, cbgraphcs: "CBGraphCS") -> "CBGraphC":
        """Initialize a canonical graph from its string form.

        Parameters
        ----------
        cbgraphcs: CBGraphCS
            String-form CBGraphC to convert.

        Returns
        -------
        CBGraphC
        """
        cgraph = CBGraph.from_cbgraphs(cbgraphcs.cbgraphs)
        method = cbgraphcs.method
        canongraph = CBGraphC(_cbgraph=cgraph, _method=method)
        return canongraph

    @property
    def cbgraph(self) -> "CBGraph":
        """CBGraph : Colored graph in graph form.

        Returns
        -------
        CBGraph
        """
        return self._cbgraph

    @property
    def method(self) -> CanonType:
        """CanonType : Method used to canonize the graph.

        Returns
        -------
        CanonType
        """
        return self._method

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}.from_bonds("
            f"bonds=({','.join(repr(b) for b in self.cbgraph.get_bonds())}),"
            f"colors={tuple(self._cbgraph.get_colors())},"
            f"method={self.method})"
        )

    def nof_nodes(self) -> int:
        """Get number of nodes in graph."""
        return self.cbgraph.nof_nodes()

    def nof_edges(self) -> int:
        """Get number of edges/bonds in graph."""
        return self.cbgraph.nof_edges()

    def to_sform(self) -> "CBGraphCS":
        """Convert to string form.

        Returns
        -------
        CBGraphCS
        """
        return CBGraphCS.from_canongraph(self)

    def get_bonds(self) -> Iterable[TBond[int]]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[TBond[int]]
        """
        return self._cbgraph.get_bonds()

    def get_colors(self) -> Sequence[int]:
        """Get graph node colors.

        Returns
        -------
        Sequence[int]
        """
        return self._cbgraph.get_colors()

    def get_edge_colors(self) -> Sequence[int]:
        """Get graph edge colors.

        Returns
        -------
        Sequence[int]
        """
        return self._cbgraph.get_edge_colors()

    def nof_automorphisms(self, method: CanonType | None = None) -> int:
        """Calculate number of automorphisms of graph."""
        if method is None:
            method = self._method
        return self._cbgraph.nof_automorphisms(method=self._method)

    def add_node(
        self, op: AddNodeT[int, int]
    ) -> tuple["CBGraphC", IndexMapping]:
        """Add node to CBGraphC, returning new CBGraphC and mapping.

        Parameters
        ----------
        op : AddNodeT[int, int]
            Node addition operation.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        new_cgraph, new_cmap = self._cbgraph.add_node(op)
        new_cangraph, canon_map = CBGraphC.from_cbgraph(
            cbgraph=new_cgraph, method=self._method
        )
        total_map = canon_map.compose(new_cmap)
        return new_cangraph, total_map

    def sub_node(self, op: SubNode) -> tuple["CBGraphC", IndexMapping]:
        """Subtract node from CBGraphC, returning new CBGraphC and mapping.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        new_cgraph, new_cmap = self._cbgraph.sub_node(op)
        new_cangraph, canon_map = CBGraphC.from_cbgraph(
            cbgraph=new_cgraph, method=self._method
        )
        total_map = canon_map.compose(new_cmap)
        return new_cangraph, total_map

    def sub_node_info(
        self, op: SubNode
    ) -> tuple["CBGraphC", IndexMapping, AddNodeT[int, int]]:
        """Subtract node from CBGraphC, return new CBGraphC plus addtl. info.

        Generally the same as `.sub_node` method, except that the inverse
        `AddNodeT` operation is also returned.

        Parameters
        ----------
        op : SubNode
            Node subtraction operation.

        Returns
        -------
        tuple[CBGraphC, IndexMapping, AddNodeT[int, int]]
        """
        new_cbgraph, new_cmap, add_op = self._cbgraph.sub_node_info(op)

        # canonize
        new_cbgraphs, canon_map = CBGraphC.from_cbgraph(
            cbgraph=new_cbgraph, method=self._method
        )
        total_map = canon_map.compose(new_cmap)

        add_op_perm = add_op.permute(canon_map)

        return new_cbgraphs, total_map, add_op_perm

    def get_parents(
        self,
    ) -> Iterable[tuple["CBGraphC", IndexMapping, AddNodeT[int, int], SubNode]]:
        """Get all direct "parents" of CBGraphC.

        Graph A is a "direct parent" of graph B if it has all the same nodes,
        colors, and connectivity as graph B except that it has one fewer nodes.

        Returns
        -------
        Iterable[tuple[CBGraphC, IndexMapping, AddNodeT[int, int], SubNode]]
        """
        for sub_i in range(self._cbgraph.nof_nodes()):
            sub_op = SubNode(sub_i)
            yield self.sub_node_info(sub_op) + (sub_op,)

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
        return self == CBGraphC.from_cbgraph(self._cbgraph, self._method)[0]

    def subset(self, indices: Iterable[int]) -> tuple["CBGraphC", IndexMapping]:
        """Return a subset of `self` consisting of the indicated nodes.

        Also returns the mapping from `self` to the returned CBGraphC.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the nodes composing the subset of the graph.

        Returns
        -------
        tuple[CBGraphC, IndexMapping]
        """
        new_graph1, new_map1 = self._cbgraph.subset(indices=indices)
        new_graph2, new_map2 = new_graph1.canonize(method=self._method)
        total_map = new_map2.compose(new_map1)

        new_canongraph = CBGraphC(_cbgraph=new_graph2, _method=self._method)

        return new_canongraph, total_map

    def all_components(self) -> Iterable[tuple["CBGraphC", IndexMapping]]:
        """Return all connected components of graph.

        Returns
        -------
        Iterable[tuple[CBGraphC, IndexMapping]]
        """
        for comp, comp_map in self.cbgraph.all_components():
            canon_comp, canon_map = comp.canonize(self.method)
            total_map = canon_map.compose(comp_map)
            yield CBGraphC(_cbgraph=canon_comp, _method=self.method), total_map


@final
@dataclasses.dataclass(eq=True, frozen=True, order=True, slots=True)
class CBGraphCS:
    """Canonized colored graph (string form).

    Complementary to `CBGraphCS`. This form is comparable,
    hashable, and (ideally) uses less memory than the graph form. It can also be
    easily encoded to a string.
    """

    _graph_sc: CBGraphS
    _method: CanonType

    @classmethod
    def from_canongraph(cls, canongraph: CBGraphC) -> "CBGraphCS":
        """Convert canonical graph from graph to string form.

        Parameters
        ----------
        canongraph: CBGraphC
            Graph-form CBGraphC to convert.

        Returns
        -------
        CBGraphCS
        """
        scgraph = CBGraphS.from_cgraph(canongraph.cbgraph)
        return CBGraphCS(_graph_sc=scgraph, _method=canongraph.method)

    @classmethod
    def from_str(cls, string: str, checked: bool) -> "CBGraphCS":
        """Initialize canonical graph from a `str`.

        Parameters
        ----------
        string : str
            String from which to initialize the canonical graph.
        checked : bool
            If True, perform check to ensure that string is well-formed. This
            takes extra computing time but is worth it on unverified data since
            all methods of `CBGraphCS` and
            `CBGraphC` assume proper canonicity.

        Returns
        -------
        CBGraphCS
        """
        method_str, scgraph_str = string.split(";", maxsplit=1)
        method = CanonType(method_str)
        scgraph = CBGraphS.from_str(scgraph_str)
        scanongraph = CBGraphCS(_graph_sc=scgraph, _method=method)
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
    def cbgraphs(self) -> CBGraphS:
        """CBGraphS : Colored graph in string form.

        Returns
        -------
        CBGraphS
        """
        return self._graph_sc

    def as_str(self) -> str:
        """Convert to string form.

        Returns
        -------
        str
        """
        return ";".join((str(self._method), self._graph_sc.as_str()))

    # def __repr__(self) -> str:
    #     return (
    #         f"{self.__class__.__name__}"
    #         f".from_canongraph({repr(self.to_canongraph())})"
    #     )

    def to_split(
        self,
    ) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
        """Convert to tuples, as suited for a `from_tuples` initializer.

        Returns
        -------
        tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]
        """
        return self._graph_sc.to_split()

    def to_canongraph(self) -> "CBGraphC":
        """Convert to graph form.

        Returns
        -------
        CBGraphC
        """
        return CBGraphC.from_sform(self)

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
        scanon_dup = CBGraphC.from_cbgraph(
            self.to_canongraph().cbgraph, method=self._method
        )[0].to_sform()

        return self == scanon_dup
