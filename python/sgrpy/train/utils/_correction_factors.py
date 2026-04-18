"""Various graph correction factors and calculations."""

_SEQUENCE_CACHE_GRAPHCOUNT = (
    1,
    1,
    1,
    2,
    6,
    21,
    112,
    853,
    11117,
    261080,
    11716571,
    1006700565,
    164059830476,
    50335907869219,
    29003487462848061,
    31397381142761241960,
    63969560113225176176277,
    245871831682084026519528568,
    1787331725248899088890200576580,
    24636021429399867655322650759681644,
)


def est_num_graphs(n: int) -> float:
    """Estimate log of number of connected undirected graphs on n nodes.

    Notes
    -----
    Estimate of log of sequence A001349.
    """
    if n < 0:
        raise ValueError(f"n must be positive (was {n})")
    if n < len(_SEQUENCE_CACHE_GRAPHCOUNT):
        return float(_SEQUENCE_CACHE_GRAPHCOUNT[n])
    raise NotImplementedError(
        f"No estimator implemented for n greater than 19 (was {n})"
    )
