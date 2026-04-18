"""Typed Graph generic class."""

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import final

from sgrpy.iotypes import CompareHash

from ._bond import TBond
from ._canon_types import CanonType
from ._cbgraph import CBGraph
from ._operations import Extension
from ._permutations import IndexMapping


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class TGraph[
    _NT: CompareHash,
    _ET: CompareHash,
]:
    """Generic-labeled vertex-bond graph."""

    _vlabels: tuple[_NT, ...]
    _elabels: tuple[_ET, ...]
    _cbgraph: CBGraph

    @classmethod
    def from_bonds(
        cls, bonds: Sequence[TBond[_ET]], labels: Sequence[_NT]
    ) -> "TGraph[_NT,_ET]":
        """Initialize a labeled TGraph from a series of bonds and labels.

        The number of nodes is the length of `labels`.  All indices should be
        expected to change.

        Parameters
        ----------
        bonds : Sequence[TBond[_ET]]
            Edges in the intended graph.
        labels : Sequence[_NT]
            Labels of the nodes in the graph.

        Returns
        -------
        TGraph[_NT,_ET]
        """
        elabel_map: dict[_ET, int] = {
            lab: l_i
            for l_i, lab in enumerate(dict.fromkeys(b.color for b in bonds))
        }
        elabels = tuple(elabel_map)

        bond_gen = (TBond.new(b.src, b.trg, elabel_map[b.color]) for b in bonds)

        vlabel_map: dict[_NT, int] = {
            lab: l_i for l_i, lab in enumerate(dict.fromkeys(labels))
        }
        vlabels = tuple(vlabel_map)

        color_gen = [vlabel_map[lab] for lab in labels]

        cbgraph = CBGraph.from_bonds(bond_gen, color_gen)

        return TGraph(_vlabels=vlabels, _elabels=elabels, _cbgraph=cbgraph)

    @classmethod
    def from_tuples(
        cls,
        bonds: Iterable[tuple[int, int, _ET]],
        labels: Sequence[_NT],
    ) -> "TGraph[_NT, _ET]":
        """Initialize a generic-labeled graph from a series of bonds and labels.

        The number of nodes is the the length of `labels` and the
        maximum index referred to by `bonds`. All indices should be expected to
        change. Bonds are not given directionality, so source and target may be
        interchanged.

        Parameters
        ----------
        bonds: Iterable[tuple[int, int, _ET]]
            Iterable of edges in the intended graph.
        labels : Sequence[_NT]
            Labels of the nodes in the graph.

        Returns
        -------
        TGraph[_NT, _ET]
        """
        return TGraph.from_bonds(
            [
                TBond.new(src=src, trg=trg, color=color)
                for src, trg, color in bonds
            ],
            labels=labels,
        )

    @classmethod
    def concat(
        cls, tgraphs: Sequence["TGraph[_NT, _ET]"]
    ) -> "TGraph[_NT, _ET]":
        """Compose multiple graphs into a single graph.

        Relative indices within each component will be maintained, with offset
        applied to each graph after the first to ensure independence.

        Parameters
        ----------
        tgraphs : Sequence[TGraph[_NT, _ET]]
            Graphs to be composed.

        Returns
        -------
        TGraph[_NT, _ET]
        """
        vert_list: list[_NT] = []
        bond_list: list[TBond[_ET]] = []
        v_offset = 0
        for tg in tgraphs:
            v_offset = len(vert_list)
            new_vertices = tg.get_labels()
            vert_list.extend(new_vertices)
            new_bonds = (
                TBond.new(nb.src + v_offset, nb.trg + v_offset, nb.color)
                for nb in tg.get_bonds()
            )
            bond_list.extend(new_bonds)

        tgraph_full = TGraph.from_bonds(bond_list, vert_list)

        return tgraph_full

    def nof_components(self) -> int:
        """Calculate number of components of graph."""
        return self._cbgraph.nof_components()

    def nof_nodes(self) -> int:
        """Get number of vertices in graph."""
        return self._cbgraph.nof_nodes()

    def nof_edges(self) -> int:
        """Get number of edges in graph."""
        return self._cbgraph.nof_edges()

    def get_bonds(self) -> Iterable[TBond[_ET]]:
        """Get bonds present in graph.

        Returns
        -------
        Iterable[TBond[_ET]]
        """
        results = (
            b.with_color(self._elabels[b.color])
            for b in self._cbgraph.get_bonds()
        )
        return results

    def get_labels(self) -> Iterable[_NT]:
        """Get graph node labels.

        Returns
        -------
        Iterable[_NT]
        """
        results = (self._vlabels[c] for c in self._cbgraph.get_colors())
        return results

    def get_edge_labels(self) -> Iterable[_ET]:
        """Get graph edge labels.

        Returns
        -------
        Iterable[_NT]
        """
        results = (self._elabels[c] for c in self._cbgraph.get_edge_colors())
        return results

    def canonize(
        self, method: CanonType
    ) -> tuple["TGraph[_NT, _ET]", IndexMapping]:
        """Canonize the labeled graph.

        Indices are changed to canonize `self` according to the value of
        `method`.  The mapping representing the index changes is also returned.

        Parameters
        ----------
        method : CanonType
            Type of canonization to use.

        Returns
        -------
        tuple[TGraph[_NT, _ET], IndexMapping]
        """
        actual_vcolors = frozenset(self._cbgraph.get_colors())
        actual_ecolors = frozenset(self._cbgraph.get_edge_colors())
        # first, sort and remap labels
        labelsdict_v = {
            label_v: iv_new
            for iv_new, label_v in enumerate(sorted(frozenset(self._vlabels)))
        }
        labels_v_new = tuple(labelsdict_v)
        labels_v_map = {
            iv_old: iv_new
            for iv_old, iv_new in enumerate(
                labelsdict_v[label_v] for label_v in self._vlabels
            )
            if iv_old in actual_vcolors
        }
        labelsdict_e = {
            label_e: ie_new
            for ie_new, label_e in enumerate(sorted(frozenset(self._elabels)))
        }
        labels_e_new = tuple(labelsdict_e)
        labels_e_map = {
            ie_old: ie_new
            for ie_old, ie_new in enumerate(
                labelsdict_e[label_e] for label_e in self._elabels
            )
            if ie_old in actual_ecolors
        }

        cbgraph_recolored = self._cbgraph.recolor(labels_v_map, labels_e_map)

        cbgraph_canon, imap = cbgraph_recolored.canonize(method)

        new_tgraph = TGraph(
            _vlabels=labels_v_new, _elabels=labels_e_new, _cbgraph=cbgraph_canon
        )

        return new_tgraph, imap

    def to_cbgraph_raw(self) -> CBGraph:
        """Return raw CBGraph (with internal indices on nodes and edges).

        Returns
        -------
        CBGraph
        """
        return self._cbgraph

    def to_cbgraph(
        self, vmap: Mapping[_NT, int], emap: Mapping[_ET, int]
    ) -> CBGraph:
        """Return CBGraph, with vertex and edge labels mapped.

        Parameters
        ----------
        vmap : Mapping[_NT, int]
            Mapping of vertex labels to integer colors.
        emap : Mapping[_ET, int]
            Mapping of edge labels to integer colors.

        Returns
        -------
        CBGraph
        """
        bonds = (b.with_color(emap[b.color]) for b in self.get_bonds())
        colors = [vmap[lab] for lab in self.get_labels()]

        new_cbgraph = CBGraph.from_bonds(bonds, colors)
        return new_cbgraph

    def remap_labels[_NTN: CompareHash, _ETN: CompareHash](
        self,
        vmap: Mapping[_NT, _NTN] | Callable[[_NT], _NTN],
        emap: Mapping[_ET, _ETN] | Callable[[_ET], _ETN],
    ) -> "TGraph[_NTN, _ETN]":
        """Remap this TGraph's labels.

        Parameters
        ----------
        vmap : Mapping[_NT, _NTN] | Callable[[_NT], _NTN]
            Mapping of vertex labels to new vertex labels.
        emap : Mapping[_ET, _ETN] | Callable[[_ET], _ETN]
            Mapping of edge labels to new edge labels.

        Returns
        -------
        TGraph[_NTN, _ETN]
        """
        if isinstance(vmap, Mapping):
            new_vlabels = tuple(vmap[lab] for lab in self._vlabels)
        else:
            new_vlabels = tuple(vmap(lab) for lab in self._vlabels)
        if isinstance(emap, Mapping):
            new_elabels = tuple(emap[lab] for lab in self._elabels)
        else:
            new_elabels = tuple(emap(lab) for lab in self._elabels)

        new_tgraph = TGraph(
            _vlabels=new_vlabels, _elabels=new_elabels, _cbgraph=self._cbgraph
        )

        return new_tgraph

    def canon_equiv(
        self, other: "TGraph[_NT, _ET]", method: CanonType = CanonType.F
    ) -> bool:
        """Test whether two TGraph objects are canonically equivalent.

        Checks available labels before actually calling canonization method; in
        many cases, this means that it should run substantially faster than
        canonizing each one before comparison.

        Parameters
        ----------
        other : TGraph[_NT, _ET]
            `TGraph` to compare with `self`.
        method : CanonType
            Canonization method (default: `CanonType.F`).
        """
        if sorted(frozenset(self._vlabels)) != sorted(
            frozenset(other._elabels)
        ):
            return False
        if sorted(frozenset(self._elabels)) != sorted(
            frozenset(other._elabels)
        ):
            return False
        return self.canonize(method)[0] == other.canonize(method)[0]

    def neighbors(self, vertex: int) -> tuple[int, ...]:
        """Get neighbors indices of a specific node.

        Returns
        -------
        int
        """
        return self._cbgraph.neighbors(vertex)

    def neighbors_ext(self, vertex: int) -> Iterable[Extension[_ET]]:
        """Get neighbor extensions toward `vertex`.

        Parameters
        ----------
        vertex : int
            Index of destination vertex.

        Returns
        -------
        Iterable[Extension[_ET]]
            Iterable of extensions approaching `vertex`.
        """
        ext_gen = self._cbgraph.neighbors_ext(vertex)
        new_ext_gen = (
            Extension(idx=ext.idx, typ=self._elabels[ext.typ])
            for ext in ext_gen
        )
        return new_ext_gen

    def subset(
        self, indices: Iterable[int]
    ) -> tuple["TGraph[_NT, _ET]", IndexMapping]:
        """Return a subset of `self` consisting of the indicated nodes.

        Also returns the mapping from `self` to the returned TGraph.

        Parameters
        ----------
        indices : Iterable[int]
            The indices of the nodes composing the subset of the graph.

        Returns
        -------
        tuple[TGraph[_NT, _ET], IndexMapping]
        """
        sub_cbgraph, new_map = self._cbgraph.subset(indices)
        new_tgraph = dataclasses.replace(self, _cbgraph=sub_cbgraph)
        return new_tgraph, new_map


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class STGraph:
    _graph_str: str

    @classmethod
    def from_tgraph(cls, tgraph: TGraph[str, str]) -> "STGraph":
        raise NotImplementedError

    @classmethod
    def from_str(cls, data: str) -> "STGraph":
        raise NotImplementedError

    def as_str(self) -> str:
        raise NotImplementedError

    def to_tgraph(self) -> TGraph[str, str]:
        raise NotImplementedError
