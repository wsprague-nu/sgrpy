"""UGraph equivalence datastructure."""

import dataclasses

from ._canon_types import CanonType
from ._cgraph_canon import CanonGraph, SCanonGraph
from ._ugraph import UGraph


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class UGraphEquiv:
    """UGraph equivalence datastructure.

    Can be used for UGraph canonical equality.
    """

    _labels: tuple[str, ...]
    _canongraph: SCanonGraph

    @classmethod
    def from_ugraph(cls, ugraph: UGraph) -> "UGraphEquiv":
        """Convert from string-labeled graph form.

        Parameters
        ----------
        ugraph : UGraph
            UGraph to be converted.

        Returns
        -------
        UGraphEquiv
        """
        labels = tuple(sorted(frozenset(ugraph.get_labels())))
        label_map = {label: i for i, label in enumerate(labels)}
        colors = (label_map[lab] for lab in ugraph.get_labels())
        cgraph, _ = CanonGraph.from_bonds(
            ugraph.get_bonds(), colors, CanonType.F
        )

        return UGraphEquiv(_labels=labels, _canongraph=cgraph.to_scanongraph())

    def to_ugraph(self) -> UGraph:
        """Convert back to UGraph form.

        The resulting UGraph will be canonical with respect to its indices.

        Returns
        -------
        UGraph
        """
        labelmap = self._labels
        canongraph = self._canongraph.to_canongraph()
        bonds = canongraph.get_bonds()
        labels = (labelmap[color] for color in canongraph.get_colors())
        return UGraph.from_bonds(bonds=bonds, labels=labels)


def canonize_ugraph(ug: UGraph) -> UGraph:
    """Return canonical `UGraph`.

    Parameters
    ----------
    ug : UGraph
        Graph to be canonized via `UGraphEquiv`.

    Returns
    -------
    UGraph
    """
    uge = UGraphEquiv.from_ugraph(ug)
    return uge.to_ugraph()
