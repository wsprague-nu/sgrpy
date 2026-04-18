"""Test of most_common() function."""

import collections.abc
import pathlib

import sgrpy

from .. import most_common


def __import_testgraph(
    graph_str: str,
) -> collections.abc.Iterable[sgrpy.graph.CGraph]:
    with open(pathlib.Path(__file__).parent / f"{graph_str}.txt", "r") as fin:
        for line in fin:
            test_graph = sgrpy.graph.SCGraph.from_str(line.strip()).to_cgraph()
            yield test_graph


def __import_results(
    result_str: str,
) -> collections.abc.Iterable[tuple[str, int]]:
    with open(pathlib.Path(__file__).parent / f"{result_str}.txt", "r") as fin:
        for line in fin:
            frag_str, count_str = line.strip().split("\t")
            result_tuple = (frag_str, int(count_str))
            yield result_tuple


def test_most_common_empty_zero() -> None:
    """Test fragment detection for empty graph."""
    test_graph = sgrpy.graph.CGraph.empty()
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_empty_one() -> None:
    """Test fragment detection for empty graph."""
    test_graph = sgrpy.graph.CGraph.empty()
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_empty_many() -> None:
    """Test fragment detection for empty graph."""
    test_graph = sgrpy.graph.CGraph.empty()
    result_gen = most_common([test_graph], limit=None)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_singlet_zero() -> None:
    """Test fragment detection for most common singlet with limit 0."""
    test_graph = sgrpy.graph.CGraph.singlet(1)
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_most_common_singlet_one() -> None:
    """Test fragment detection for most common singlet with limit 1."""
    test_graph = sgrpy.graph.CGraph.singlet(1)
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_most_common_singlet_two() -> None:
    """Test fragment detection for most common singlet with limit 2."""
    test_graph = sgrpy.graph.CGraph.singlet(1)
    result_gen = most_common([test_graph], limit=2)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_most_common_singlet_many() -> None:
    """Test fragment detection for most common singlet with no limit."""
    test_graph = sgrpy.graph.CGraph.singlet(1)
    result_gen = most_common([test_graph], limit=None)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (("f;;1", 1),)

    assert results == expected


def test_random_graph_zero() -> None:
    """Test fragment detection for single random graph."""
    test_graphs = list(__import_testgraph("test_graph"))
    results = tuple(
        (sc.as_str(), i) for sc, i in most_common(test_graphs, limit=0)
    )
    expected = ()
    assert results == expected


def test_random_graph_one() -> None:
    """Test fragment detection for single random graph."""
    test_graphs = list(__import_testgraph("test_graph"))
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
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6;2,1,2,1,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=0)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = ()

    assert results == expected


def test_fragments_pathological_one() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6;2,1,2,1,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=1)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 5),
        ("f;;1", 2),
    )

    assert results == expected


def test_fragments_pathological_two() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6,4,7,5,8,6,9;2,1,2,1,2,2,2,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=2)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
    )

    assert results == expected


def test_fragments_pathological_three() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6,4,7,5,8,6,9;2,1,2,1,2,2,2,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=3)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1;2,2", 3),
    )

    assert results == expected


def test_fragments_pathological_four() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6,4,7,5,8,6,9;2,1,2,1,2,2,2,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=4)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1;2,2", 3),
        ("f;0,1;1,2", 2),
        ("f;0,1,0,2;1,2,2", 2),
    )

    assert results == expected


def test_fragments_pathological_five() -> None:
    """Test fragment detection for pathological graph."""
    graph_str = "0,1,1,2,2,3,3,4,3,5,3,6,4,7,5,8,6,9;2,1,2,1,2,2,2,2,2,2"
    test_graph = sgrpy.graph.SCGraph.from_str(graph_str).to_cgraph()
    result_gen = most_common([test_graph], limit=5)
    results = tuple((sc.as_str(), i) for sc, i in result_gen)

    expected = (
        ("f;;2", 8),
        ("f;;1", 2),
        ("f;0,1;2,2", 3),
        ("f;0,1;1,2", 2),
        ("f;0,1,0,2;1,2,2", 2),
    )

    assert results == expected


def test_fragments_stefan_0() -> None:
    """Test fragment detection for no reaction graphs."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=0, exc_dependent=True)
    )
    expected = ()
    assert results == expected


def test_fragments_stefan_1() -> None:
    """Test fragment detection for 1 reaction graph."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=1, exc_dependent=True)
    )
    expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
    assert results == expected


def test_fragments_stefan_2() -> None:
    """Test fragment detection for 2 reaction graphs."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=2, exc_dependent=True)
    )
    expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
    assert results == expected


def test_fragments_stefan_51() -> None:
    """Test fragment detection for 51 reaction graphs."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=51, exc_dependent=True)
    )
    expected = tuple(__import_results("test_stefan_graphs_solutions"))[:51]
    assert results == expected


def test_fragments_stefan_52() -> None:
    """Test fragment detection for 52 reaction graphs."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=52, exc_dependent=True)
    )
    expected = tuple(__import_results("test_stefan_graphs_solutions"))[:52]
    assert results == expected


def test_fragments_stefan_86() -> None:
    """Test fragment detection for 86 reaction graphs."""
    test_graphs = tuple(__import_testgraph("test_stefan_graphs"))
    results = tuple(
        (sc.as_str(), i)
        for sc, i in most_common(test_graphs, limit=86, exc_dependent=True)
    )
    expected = tuple(__import_results("test_stefan_graphs_solutions"))
    assert results == expected
