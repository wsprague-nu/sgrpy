"""Some common training datastructures."""

import collections
import dataclasses
import operator

import numpy

import sgrpy

from ._fragmodel import FragModel
from ._observation import Observation


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationResult:
    lweight: numpy.float64
    counts: dict[int, int]


def process_observation(obs: Observation) -> ObservationResult:
    lweight = obs.weight
    match_counts: dict[int, int] = dict(
        collections.Counter(
            map(operator.attrgetter("fragment_id"), obs.matches)
        )
    )
    return ObservationResult(lweight=lweight, counts=match_counts)


def perform_partition(
    model: FragModel,
    graph: tuple[int, sgrpy.graph.CGraph],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
) -> tuple[int, tuple[ObservationResult, ...]]:
    group_id, cgraph = graph
    result_gen = (
        process_observation(obs)
        for obs in model.partition(
            cgraph,
            limit=limit,
            max_iter=max_iter,
            max_heap=max_heap,
            useac=False,
        )
    )
    result_tup = tuple(result_gen)
    return group_id, result_tup
