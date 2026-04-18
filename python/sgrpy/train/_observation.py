"""Observation datastructures."""

import collections.abc
import dataclasses

import numpy

import sgrpy


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """Partition observation, consisting of fragment matches and a logweight."""

    matches: frozenset[sgrpy.graph.GraphMatch]
    weight: numpy.float64


def normalize_obs(
    observations: collections.abc.Iterable[Observation],
) -> collections.abc.Sequence[Observation]:
    raise NotImplementedError
