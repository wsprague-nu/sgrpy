"""Test query_fragment() and query_fragment_atomic() methods."""

import pathlib

import sgrpy

from .._frequencyentry import FragmentMatch
from .._most_common_n import (
    query_fragment,
    query_fragment_atomic,
)


def test_query_fragment_atomic_v1() -> None:
    """Test atomic queries on fragment."""
    with open(pathlib.Path(__file__).parent / "test_graph.txt", "r") as fin:
        test_graph = sgrpy.graph.SCGraph.from_str(
            fin.read().strip()
        ).to_cgraph()
    results = tuple(query_fragment_atomic(test_graph, sgrpy.graph.CanonType.F))

    result_fragments = tuple(r.fragment.as_str() for r in results)

    expected_fragments = (
        "f;;2",
        "f;;2",
        "f;;4",
        "f;;4",
        "f;;5",
        "f;;1",
        "f;;5",
        "f;;3",
        "f;;1",
        "f;;3",
    )

    assert result_fragments == expected_fragments

    result_maps = tuple(r.mapping.as_tuple_int() for r in results)

    expected_maps = (
        (0,),
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
        (9,),
    )

    assert result_maps == expected_maps


def test_query_fragment_v1() -> None:
    """Test atomic queries on fragment."""
    with open(pathlib.Path(__file__).parent / "test_graph.txt", "r") as fin:
        test_graph = sgrpy.graph.SCGraph.from_str(
            fin.read().strip()
        ).to_cgraph()

    query_graph, query_canonmap = sgrpy.graph.CanonGraph.from_cgraph(
        sgrpy.graph.SCGraph.from_str("0,1,1,2,2,3,0,3;5,2,5,3").to_cgraph(),
        method=sgrpy.graph.CanonType.F,
    )
    query_cg_mapping = sgrpy.graph.IndexMapping.from_seq((6, 0, 4, 7))
    query_mapping = query_canonmap.compose(query_cg_mapping)
    query_match = FragmentMatch.new(
        fragment=query_graph.to_scanongraph(), mapping=query_mapping
    )

    match_result = tuple(query_fragment(query_match, test_graph))

    result_graphs = tuple(r.fragment.as_str() for r in match_result)

    expected_graphs = (
        "f;0,2,1,3,2,3,1,4,2,4;2,2,3,5,5",
        "f;0,2,0,3,1,3,2,3,0,4,1,4,2,4;1,2,3,5,5",
        "f;0,3,1,3,0,4,1,4,2,4;2,3,4,5,5",
        "f;0,3,1,3,2,3,0,4,1,4,2,4;2,3,4,5,5",
    )

    assert result_graphs == expected_graphs

    result_maps = tuple(r.mapping.as_tuple_int() for r in match_result)

    expected_maps = (
        (1, 0, 7, 6, 4),
        (8, 0, 7, 6, 4),
        (0, 7, 2, 6, 4),
        (0, 7, 3, 6, 4),
    )

    assert result_maps == expected_maps
