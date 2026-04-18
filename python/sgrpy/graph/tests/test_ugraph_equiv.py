"""Test UGraphEquiv structure."""

import sgrpy


def test_ug_1() -> None:
    """Ensure UGraphEquiv canonical ordering remains consistent."""
    bonds = (
        (1, 3),
        (2, 3),
        (0, 4),
        (1, 4),
        (0, 5),
        (2, 5),
    )
    labels = (
        "6,4,3,0,0,3,True,0",
        "6,4,3,0,0,3,True,0",
        "6,4,3,0,0,3,True,0",
        "6,4,3,0,0,3,True,0",
        "6,4,3,0,0,3,True,0",
        "!",
    )
    ugraph = sgrpy.graph.UGraph.from_tuples(bonds, labels)
    ugraph_equiv = sgrpy.graph.UGraphEquiv.from_ugraph(ugraph)
    ugraph_norm = ugraph_equiv.to_ugraph()
    ugraph_str = ugraph_norm.to_sugraph().as_str()
    assert (
        ugraph_str
        == '1,3,2,3,0,4,1,4,0,5,2,5;["!","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True'
        ',0","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0"]'
    )
