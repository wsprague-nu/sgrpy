"""Color mapper to convert between TGraph and CBGraph forms."""

import collections
import dataclasses
import operator
from collections.abc import Iterable
from typing import final

from sgrpy.iotypes import JSON, CompareHash

from ._cbgraph import CBGraph
from ._tgraph import TGraph


class ColorMappingError(Exception):
    """Raise when ColorMapper fails to map a graph."""


class EdgeKeyError(ColorMappingError):
    """Raise when ColorMapper is missing a key for an edge label."""


class VertexKeyError(ColorMappingError):
    """Raise when ColorMapper is missing a key for a vertex label."""


@final
@dataclasses.dataclass(frozen=True, slots=True)
class ColorMapper[_NT: CompareHash, _ET: CompareHash]:
    """Object which can convert between generic- and integer-label graphs."""

    _forward_dict_n: dict[_NT, int]
    _forward_dict_e: dict[_ET, int]
    _reverse_map_n: tuple[_NT]
    _reverse_map_e: tuple[_ET]

    @classmethod
    def from_graphs(
        cls, tgraphs: Iterable[TGraph[_NT, _ET]]
    ) -> "ColorMapper[_NT, _ET]":
        """Build ColorMapper from TGraph examples.

        Colors will be weighted by frequency and sorted order.

        Parameters
        ----------
        tgraphs: Iterable[TGraph[_NT, _ET]]
            Example graphs from which to build mapper.

        Returns
        -------
        ColorMapper[_NT, _ET]
        """
        vertex_c_t: collections.Counter[_NT] = collections.Counter()
        vertex_c_u: collections.Counter[_NT] = collections.Counter()
        edge_c_t: collections.Counter[_ET] = collections.Counter()
        edge_c_u: collections.Counter[_ET] = collections.Counter()

        # accumulate all counters
        for tgraph in tgraphs:
            vertex_c = collections.Counter(tgraph.get_labels())
            vertex_c_t += vertex_c
            vertex_c_u.update(vertex_c.keys())
            edge_c = collections.Counter(tgraph.get_edge_labels())
            edge_c_t += edge_c
            edge_c_u.update(edge_c.keys())

        # sort and assign vertex labels
        # (rev graph count, rev total count, sortable order)
        vertex_labels = sorted(
            (-count_u, -vertex_c_t[key], key)
            for key, count_u in vertex_c_u.items()
        )
        reverse_map_n: tuple[_NT] = tuple(
            map(operator.itemgetter(2), vertex_labels)
        )
        forward_dict_n = {label: i for i, label in enumerate(reverse_map_n)}

        # sort and assign edge labels
        edge_labels = sorted(
            (-count_u, -edge_c_t[key], key) for key, count_u in edge_c_u.items()
        )
        reverse_map_e: tuple[_ET] = tuple(
            map(operator.itemgetter(2), edge_labels)
        )
        forward_dict_e = {label: i for i, label in enumerate(reverse_map_e)}

        mapper = ColorMapper(
            _forward_dict_n=forward_dict_n,
            _forward_dict_e=forward_dict_e,
            _reverse_map_n=reverse_map_n,
            _reverse_map_e=reverse_map_e,
        )

        return mapper

    def color_graph(self, tgraph: TGraph[_NT, _ET]) -> CBGraph:
        """Color graph by assigning integers to each label.

        Parameters
        ----------
        tgraph : TGraph[_NT, _ET]
            Generic-labeled graph to be converted.

        Returns
        -------
        CBGraph
            Colored graph.

        Raises
        ------
        EdgeKeyError(ColorMappingError)
            If an edge label has no corresponding color.
        VertexKeyError(ColorMappingError)
            If a vertex label has no corresponding color.
        """
        try:
            vcolors = tuple(
                self._forward_dict_n[vl] for vl in tgraph.get_labels()
            )
        except KeyError as err:
            raise VertexKeyError from err
        try:
            bonds = tuple(
                bond.with_color(self._forward_dict_e[bond.color])
                for bond in tgraph.get_bonds()
            )
        except KeyError as err:
            raise EdgeKeyError from err

        cbgraph = CBGraph.from_bonds(bonds, vcolors)

        return cbgraph

    def label_graph(self, cbgraph: CBGraph) -> TGraph[_NT, _ET]:
        """Label graph by mapping integers to labels.

        Parameters
        ----------
        cbgraph : CBGraph
            Integer-labeled graph to be converted.

        Returns
        -------
        TGraph[_NT, _ET]
            Labeled graph.

        Raises
        ------
        EdgeKeyError(ColorMappingError)
            If an edge color has no corresponding label.
        VertexKeyError(ColorMappingError)
            If a vertex color has no corresponding label.
        """
        try:
            vlabels = tuple(
                self._reverse_map_n[vl] for vl in cbgraph.get_colors()
            )
        except KeyError as err:
            raise VertexKeyError from err
        try:
            bonds = tuple(
                bond.with_color(self._reverse_map_e[bond.color])
                for bond in cbgraph.get_bonds()
            )
        except KeyError as err:
            raise EdgeKeyError from err

        tgraph = TGraph.from_bonds(bonds, vlabels)

        return tgraph

    def to_json(self) -> JSON:
        raise NotImplementedError
