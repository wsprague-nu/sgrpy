"""Rust components implementing Dancing Links algorithm [1]_.

References
----------
.. [1] D. E. Knuth, "Dancing links", arXiv preprint cs/0011047, 2000.
"""

from collections.abc import Sequence

def get_all(
    partitions: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Get all partitions based on provided covers.

    Parameters
    ----------
    partitions : Sequence[Sequence[int]]
        Input covers.  Each entry represents a cover of a set of (positive)
        integers.

    Returns
    -------
    list[list[int]]
        List of partitions.  Each top-level entry is a different list of entries
        which correspond to the top-level indices of the covers provided in the
        `partitions` argument.
    """

def get_top(
    partitions: Sequence[Sequence[int]],
    weights: Sequence[float],
    max_iter: int | None = None,
    max_heap: int | None = None,
) -> list[list[int]]:
    """Get all partitions in order based on provided covers and weights.

    Parameters
    ----------
    partitions : Sequence[Sequence[int]]
        Input covers.  Each entry represents a cover of a set of (positive)
        integers.
    weights : Sequence[float]
        Additive (positive) weights corresponding to the cost of each cover.
    max_iter : int | None
        Maximum number of iterations to perform during search (default:
        unlimited).  Set to reduce computation time, at the cost of potentially
        missing high-cost partitions.
    max_heap : int | None
        Maximum size of search tree (default: unlimited).  Set to reduce memory
        usage, at the cost of potentially missing high-cost partitions.

    Returns
    -------
    list[list[int]]
        List of partitions.  Each top-level entry is a different list of entries
        which correspond to the top-level indices of the covers provided in the
        `partitions` argument.  Order of returned values is based on lowest
        total cost according to the `weights` argument.
    """

def get_topn(
    partitions: Sequence[Sequence[int]],
    weights: Sequence[float],
    n: int,
    max_iter: int | None = None,
    max_heap: int | None = None,
) -> list[list[int]]:
    """Get n partitions in order based on provided covers and weights.

    Parameters
    ----------
    partitions : Sequence[Sequence[int]]
        Input covers.  Each entry represents a cover of a set of (positive)
        integers.
    weights : Sequence[float]
        Additive (positive) weights corresponding to the cost of each cover.
    n : int
        Maximum number of partitions to return.  Lower to reduce computation
        time.
    max_iter : int | None
        Maximum number of iterations to perform during search (default:
        unlimited).  Set to reduce computation time, at the cost of potentially
        missing high-cost partitions.
    max_heap : int | None
        Maximum size of search tree (default: unlimited).  Set to reduce memory
        usage, at the cost of potentially missing high-cost partitions.

    Returns
    -------
    list[list[int]]
        List of partitions.  Each top-level entry is a different list of entries
        which correspond to the top-level indices of the covers provided in the
        `partitions` argument.  Order of returned values is based on lowest
        total cost according to the `weights` argument.
    """

def get_topn_sc(
    partitions: Sequence[Sequence[int]],
    weights: Sequence[float],
    edge_counts: Sequence[int],
    edge_total: int,
    labels: Sequence[int],
    vertex_total: int,
    n: int,
    max_iter: int | None = None,
    max_heap: int | None = None,
) -> list[list[int]]:
    """Get n partitions in order based on provided data and general algo.

    .. deprecated:: 0.1.16
        `get_topn_sc` uses an outdated summation metric which is not relevant to
        current usage of `sgrpy` and will be removed at a later date
    """

def get_topp(
    partitions: Sequence[Sequence[int]],
    weights: Sequence[float],
    p: float,
    max_iter: int | None = None,
    max_heap: int | None = None,
) -> list[list[int]]:
    """Get partitions covering at least `p` in order based on provided weights.

    `p` is a proportion, assuming that the weight provided are logweights and
    that the distribution over partitions is normalized.  Unlike `get_topn`,
    this method avoids returning very high-cost partitions, but may not have the
    same nice convergence properties (not yet studied in detail).

    Parameters
    ----------
    partitions : Sequence[Sequence[int]]
        Input covers.  Each entry represents a cover of a set of (positive)
        integers.
    weights : Sequence[float]
        Additive (positive) weights corresponding to the cost of each cover.
    p : float
        Minimum proportion of relevant partitions to return.
    max_iter : int | None
        Maximum number of iterations to perform during search (default:
        unlimited).  Set to reduce computation time, at the cost of potentially
        missing high-cost partitions.
    max_heap : int | None
        Maximum size of search tree (default: unlimited).  Set to reduce memory
        usage, at the cost of potentially missing high-cost partitions.

    Returns
    -------
    list[list[int]]
        List of partitions.  Each top-level entry is a different list of entries
        which correspond to the top-level indices of the covers provided in the
        `partitions` argument.  Order of returned values is based on lowest
        total cost according to the `weights` argument.
    """
