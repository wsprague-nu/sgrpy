"""Tests for CBGraph object."""

import pathlib
from collections.abc import Iterable

import igraph
import numpy

from .._canon_types import CanonType
from .._cbgraph import CBGraph, CBGraphS

_DIRPATH = pathlib.Path(__file__).parent


def _make_randgraph() -> CBGraphS:
    vspace = 4
    espace = 4
    n = 256
    p = 1 / (n - 1)
    rng = numpy.random.default_rng()

    g = igraph.Graph.Erdos_Renyi(n=n, p=p)
    bt: list[tuple[int, int]] = g.to_tuple_list()
    btc = [(x[0], x[1], int(rng.integers(espace))) for x in bt]
    vt = [int(rng.integers(vspace)) for _ in range(n)]

    cg = CBGraph.from_tuples(btc, vt)
    cgs = cg.to_cbgraphs()

    return cgs


def _make_manygraphs(filename: str) -> None:
    n_graphs = 16
    all_graphs = [_make_randgraph() for _ in range(n_graphs)]
    with open(_DIRPATH / filename, "w") as fout:
        for g in all_graphs:
            fout.write(f"{g.as_str()}\n")


def _load_cbgraphs(filename: str) -> Iterable[CBGraph]:
    with open(_DIRPATH / filename) as fin:
        for line in fin:
            cgs = CBGraphS.from_str(line.strip())
            yield cgs.to_cbgraph()


def test_empty() -> None:
    """Test empty CBGraph."""
    emptyg = CBGraph.empty()
    assert tuple(emptyg.get_colors()) == ()
    assert tuple(emptyg.get_edge_colors()) == ()
    emptyg_can, _ = emptyg.canonize(CanonType.F)
    assert tuple(emptyg_can.get_colors()) == ()
    assert tuple(emptyg_can.get_edge_colors()) == ()


def test_singlet() -> None:
    """Test empty CBGraph."""
    singleg = CBGraph.singlet(7)
    assert tuple(singleg.get_colors()) == (7,)
    assert tuple(singleg.get_edge_colors()) == ()
    singleg_can, _ = singleg.canonize(CanonType.F)
    assert tuple(singleg_can.get_colors()) == (7,)
    assert tuple(singleg_can.get_edge_colors()) == ()


def test_cbgraph_canon() -> None:
    """Test canonization of CBGraph."""
    rng = numpy.random.default_rng(35146)

    for cbg_base in _load_cbgraphs("cbgraphs_samples_128_4_4_16.txt"):
        cbg_base_canon = cbg_base.canonize(CanonType.F)[0]
        cbgs_base = cbg_base_canon.to_cbgraphs()

        for _ in range(32):
            seed = int(rng.integers(65535))
            cbg_perm, shuf_perm = cbg_base.shuffle(seed)
            cbgs_perm = cbg_perm.to_cbgraphs()

            # check that returned permutation is correct/replicates shuffle
            cbg_perm2 = cbg_base.permute(shuf_perm)
            cbgs_perm2 = cbg_perm2.to_cbgraphs()
            assert cbgs_perm == cbgs_perm2

            # assert canonized graph is still the same
            cbg_canon = cbg_perm.canonize(CanonType.F)[0]
            cbgs_test = cbg_canon.to_cbgraphs()
            assert cbgs_base == cbgs_test

            # note: shuffle and canonization permutations are not necessarily
            # inverses due to potential automorphisms


# def test_cbgraph_search_single() -> None:
#     """Test FragmentTree search methods for reproducibility."""
#     rng = numpy.random.default_rng(12290)

#     for cbg_base in _load_cbgraphs("cbgraphs_samples_128_4_4_16.txt"):
#         cbg_base_canon = CBGraphC.from_cbgraph(
#             cbg_base, method=CanonType.F)[0]
#         frag_tree_base = FragmentTree.from_pins(
#             comp for comp, _ in cbg_base_canon.all_components()
#         )
#         cbgs_base = cbg_base_canon.to_cbgraphs()
#         cbgc = CBGraphC.from_cbgraph(cbg_base, CanonType.F)
#         cbg_base.all_components()
#         ...
