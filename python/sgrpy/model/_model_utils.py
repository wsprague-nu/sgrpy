"""Utility functions."""

from collections.abc import Callable

import numpy

from sgrpy.graph import CGraph, SCGraph

from ._model_types import ObservationMultiStruct


def unpack_obs_to_vecs(
    cgstr: str,
    plexer: Callable[[CGraph], ObservationMultiStruct],
    size: int,
) -> list[tuple[float, list[int]]]:
    cg = SCGraph.from_str(cgstr).to_cgraph()
    obs = plexer(cg)
    all_obs: list[tuple[float, list[int]]] = []

    for lweight, obsdat in zip(obs.lweights, obs.data, strict=True):
        new_vector = numpy.zeros(size, dtype=numpy.uint64)
        for item in obsdat.matches:
            new_vector[item] += 1
        all_obs.append((lweight, new_vector.tolist()))

    return all_obs
