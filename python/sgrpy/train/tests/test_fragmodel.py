"""Tests for FragModel datastructure."""

import collections.abc
import pathlib

import sgrpy


def __import_testgraph(
    graph_str: str,
) -> collections.abc.Iterable[sgrpy.graph.CGraph]:
    with open(pathlib.Path(__file__).parent / f"{graph_str}.txt", "r") as fin:
        for line in fin:
            test_graph = sgrpy.graph.SCGraph.from_str(line.strip()).to_cgraph()
            yield test_graph


# def test_fragmodel_stefan() -> None:
#     # import graphs
#     test_graphs = tuple(__import_testgraph("test_stefan_graphs"))

#     # find most common subgraphs
#     common_sub = tuple(
#         (sc, i)
#         for sc, i in most_common(test_graphs, limit=192, exc_dependent=False)
#     )

#     fragments = tuple(f.to_canongraph() for f, _ in common_sub)

#     integer_weights = numpy.asarray(
#         tuple(w for _, w in common_sub), dtype=numpy.uintp
#     )

#     weights = normalize_int_log(integer_weights)

#     # build initial model
#     model = FragModel.new(fragments, weights)

#     weights_em, _ = model_em(
#         model, test_graphs, limit=64, useac=True, progress=True
#     )

#     assert len(weights_em) == model.nof_fragments()

#     result_tuple = tuple(
#         sorted(
#             (
#                 (frag, weight)
#                 for frag, weight in zip(
#                     model.fragtree.iter_pins(), weights_em, strict=True
#                 )
#             ),
#             key=lambda x: -x[1],
#         )
#     )

#     with open("./weightout.txt", "w") as fout:
#         for frag, weight in result_tuple:
#             fout.write(f"{frag.as_str()}\t{weight}\n")
