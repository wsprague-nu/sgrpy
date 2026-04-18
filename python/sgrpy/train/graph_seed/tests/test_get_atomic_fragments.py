"""Test the `get_atomic_fragments()` function."""

# import pathlib

# import sgrpy

# from .._most_common_n import get_atomic_fragments


# def test_atomic_fragments() -> None:
#     """Test whether `get_atomic_fragments` works as intended."""
#     with open(pathlib.Path(__file__).parent / "test_graph.txt", "r") as fin:
#         test_graph = sgrpy.graph.SCGraph.from_str(
#             fin.read().strip()
#         ).to_cgraph()
#     results = tuple(get_atomic_fragments([test_graph]))
#     print(results)
#     pass
