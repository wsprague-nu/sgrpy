"""Test TGraph properties."""

from .._canon_types import CanonType
from .._tgraph import TGraph


def test_tgraph_order() -> None:
    """Test TGraph label ordering."""
    vertices = ["0", "1", "0", "3", "2"]
    tgraph: TGraph[str, int] = TGraph.from_bonds((), vertices)
    assert tuple(tgraph.get_labels()) == ("0", "1", "0", "3", "2")
    tgraph_can = tgraph.canonize(CanonType.F)[0]
    assert tuple(tgraph_can.get_labels()) == ("0", "0", "1", "2", "3")
