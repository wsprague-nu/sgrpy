"""Test of most_common function as applies to CBGraphs."""

import pathlib
from collections.abc import Iterable

from sgrpy.graph import CBGraph, CBGraphS, SCGraph

from .._most_common_n_v2 import most_common


def __convert_cbgraphs(cgraph: SCGraph) -> CBGraphS:
    cg = cgraph.to_cgraph()
    all_colors = cg.get_colors()
    bonds = ((b.src, b.trg, 0) for b in cg.get_bonds())
    cbgs = CBGraph.from_tuples(bonds, all_colors).to_cbgraphs()
    return cbgs


def __import_testgraph_legacy(
    graph_str: str,
) -> Iterable[CBGraph]:
    with open(pathlib.Path(__file__).parent / f"{graph_str}.txt", "r") as fin:
        for line in fin:
            test_graph = __convert_cbgraphs(
                SCGraph.from_str(line.strip())
            ).to_cbgraph()
            yield test_graph


def __import_testgraph(graph_str: str) -> Iterable[CBGraph]:
    with open(pathlib.Path(__file__).parent / f"{graph_str}.txt", "r") as fin:
        for line in fin:
            test_graph = CBGraphS.from_str(line.strip()).to_cbgraph()
            yield test_graph


def __import_results_legacy(
    result_str: str,
) -> Iterable[tuple[str, int]]:
    with open(pathlib.Path(__file__).parent / f"{result_str}.txt", "r") as fin:
        for line in fin:
            frag_str, count_str = line.strip().split("\t")
            frag_fin = __convert_cbgraphs(SCGraph.from_str(frag_str)).as_str()
            result_tuple = (frag_fin, int(count_str))
            yield result_tuple


def __import_results(
    result_str: str,
) -> Iterable[tuple[str, int]]:
    with open(pathlib.Path(__file__).parent / f"{result_str}.txt", "r") as fin:
        for line in fin:
            frag_str, count_str = line.strip().split("\t")
            result_tuple = (frag_str, int(count_str))
            yield result_tuple


def test_most_common_empty_zero() -> None:
    """Test fragment detection for empty graph."""
    test_graph = CBGraph.empty()
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_empty_one() -> None:
    """Test fragment detection for empty graph."""
    test_graph = CBGraph.empty()
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_empty_many() -> None:
    """Test fragment detection for empty graph."""
    test_graph = CBGraph.empty()
    result_gen = most_common([test_graph], limit=None)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_singlet_zero() -> None:
    """Test fragment detection for most common singlet with limit 0."""
    test_graph = CBGraph.singlet(1)
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_singlet_one() -> None:
    """Test fragment detection for most common singlet with limit 1."""
    test_graph = CBGraph.singlet(1)
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_most_common_singlet_two() -> None:
    """Test fragment detection for most common singlet with limit 2."""
    test_graph = CBGraph.singlet(1)
    result_gen = most_common([test_graph], limit=2)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_most_common_singlet_many() -> None:
    """Test fragment detection for most common singlet with no limit."""
    test_graph = CBGraph.singlet(1)
    result_gen = most_common([test_graph], limit=None)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_random_graph_zero() -> None:
    """Test fragment detection for single random graph."""
    test_graphs = list(__import_testgraph_legacy("test_graph"))
    results = tuple(
        (sc.as_str(), i) for sc, i in most_common(test_graphs, limit=0)
    )
    expected = ()
    assert results == expected


def test_random_graph_one() -> None:
    """Test fragment detection for single random graph."""
    test_graphs = list(__import_testgraph_legacy("test_graph"))
    results = tuple(
        (sc.as_str(), i) for sc, i in most_common(test_graphs, limit=1)
    )
    expected = (
        ("f;;2", 2),
        ("f;;4", 2),
        ("f;;5", 2),
        ("f;;1", 2),
        ("f;;3", 2),
    )
    assert results == expected


def test_fragments_pathological_zero() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0;2,1,2,1,2,2,2"
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_fragments_pathological_one() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0;2,1,2,1,2,2,2"
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 5),
        ("f;;1", 2),
    )

    assert results == expected


def test_fragments_pathological_two() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = (
        "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0,4,7,0,5,8,0,6,9,0;"
        "2,1,2,1,2,2,2,2,2,2"
    )
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=2)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
    )

    assert results == expected


def test_fragments_pathological_three() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = (
        "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0,4,7,0,5,8,0,6,9,0;"
        "2,1,2,1,2,2,2,2,2,2"
    )
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=3)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1,0;2,2", 3),
    )

    assert results == expected


def test_fragments_pathological_four() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = (
        "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0,4,7,0,5,8,0,6,9,0;"
        "2,1,2,1,2,2,2,2,2,2"
    )
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=4)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1,0;2,2", 3),
        ("f;0,1,0;1,2", 2),
        ("f;0,1,0,0,2,0;1,2,2", 2),
    )

    assert results == expected


def test_fragments_pathological_five() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = (
        "0,1,0,1,2,0,2,3,0,3,4,0,3,5,0,3,6,0,4,7,0,5,8,0,6,9,0;"
        "2,1,2,1,2,2,2,2,2,2"
    )
    test_graph = CBGraphS.from_str(graph_str).to_cbgraph()
    result_gen = most_common([test_graph], limit=5)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1,0;2,2", 3),
        ("f;0,1,0;1,2", 2),
        ("f;0,1,0,0,2,0;1,2,2", 2),
    )

    assert results == expected


# def test_fragments_stefan_0() -> None:
#     """Test fragment detection for no reaction graphs."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=0, exc_dependent=True)
#     )
#     expected = ()
#     assert results == expected


# def test_fragments_stefan_1() -> None:
#     """Test fragment detection for 1 reaction graph."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=1, exc_dependent=True)
#     )
#     expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
#     assert results == expected


# def test_fragments_stefan_2() -> None:
#     """Test fragment detection for 2 reaction graphs."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=2, exc_dependent=True)
#     )
#     expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
#     assert results == expected


# def test_fragments_stefan_51() -> None:
#     """Test fragment detection for 51 reaction graphs."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=51, exc_dependent=True)
#     )
#     expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
#     assert results == expected


# def test_fragments_stefan_52() -> None:
#     """Test fragment detection for 52 reaction graphs."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=52, exc_dependent=True)
#     )
#     expected = tuple(__import_results("test_stefan_graphs_solutions"))[:52]
#     assert results == expected


# def test_fragments_stefan_86() -> None:
#     """Test fragment detection for 86 reaction graphs."""
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
#     results = tuple(
#         (sc.as_str(), i)
#         for sc, i in most_common(test_graphs, limit=86, exc_dependent=True)
#     )
#     expected = tuple(__import_results("test_stefan_graphs_solutions"))
#     assert results == expected


def test_fragments_er() -> None:
    """Test random graphs from file."""
    weights = [96, 97, 86, 98, 1, 32, 66, 93, 19, 46, 69, 51, 60, 20, 114, 60]
    test_graphs = list(__import_testgraph("cbgraphs_samples_128_4_4_16"))
    result_gen = most_common(test_graphs, weights=weights, limit=256)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)
    expected = tuple(__import_results("cbgraphs_samples_128_4_4_16_results"))
    assert results == expected
