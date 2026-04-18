"""Expectation-maximization training functions."""

import collections
import collections.abc
import functools
import itertools
import operator

import joblib
import numpy
import numpy.typing
import scipy
import tqdm

import sgrpy

from ._common import ObservationResult, perform_partition
from ._fragmodel import FragModel
from .utils import normalize_log_log


def model_em_groups(
    model: FragModel,
    norm_groups: collections.abc.Iterable[
        collections.abc.Iterable[sgrpy.graph.CGraph]
    ],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
    useac: bool = False,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64]:
    """Perform one round of expectation-maximization.

    This function returns an unnormalized set of logweights for model fragments,
    after one round of expectation-maximization on the example graphs provided.
    Partitions are normalized to a probability of one for each norm group.

    Parameters
    ----------
    model : FragModel
        Fragment model to be tuned.
    examples : Iterable[Iterable[CGraph]]
        Normalization groups used for performing E-M.
    limit : None | int
        Maximum number of partitions per graph to generate.
    max_iter : None | int
        Maximum number of iterations to search.
    max_heap : None | int
        Maximum size of the search tree.
    useac : bool
        Use generic graph corrections to likelihood (default: True).
    progress : bool
        Print progress to command line via `tqdm` (default: False).

    Returns
    -------
    tuple[NDArray[float64],float64]
        First entry in tuple is the tuned logweights of fragments, second entry
        is the total loglikelihood of `examples`.
    """
    new_weight_agg: list[None | list[numpy.float64]] = [
        None for _ in range(model.nof_fragments())
    ]
    group_iter = tqdm.tqdm(norm_groups) if progress else norm_groups
    total_ll_agg: list[numpy.float64] = []
    for group in group_iter:
        graph_agg: dict[int, list[numpy.float64]] = {}
        prob_agg: list[numpy.float64] = []
        tallied = False
        for graph in group:
            tallied = True
            for observation in model.partition(
                graph, limit, max_iter, max_heap, useac
            ):
                label_counts = collections.Counter(
                    m.fragment_id for m in observation.matches
                )
                prob_agg.append(observation.weight)

                # tabulate counts and probabilities
                for label, count in label_counts.items():
                    new_value = (
                        numpy.log(numpy.float64(count)) + observation.weight
                    )
                    if label not in graph_agg:
                        graph_agg[label] = [new_value]
                    else:
                        graph_agg[label].append(new_value)

        if not tallied:
            continue

        # calculate total likelihood for group
        group_ll = scipy.special.logsumexp(prob_agg)
        total_ll_agg.append(group_ll)

        # normalize values by group total likelihood
        for component, weights in graph_agg.items():
            if group_ll == numpy.float64("-inf"):
                continue
            new_value = scipy.special.logsumexp(weights) - group_ll
            target_entry = new_weight_agg[component]
            if target_entry is None:
                new_weight_agg[component] = [new_value]
            else:
                target_entry.append(new_value)

    # tally all values from aggregator
    new_weights = numpy.asarray(
        [
            numpy.float64("-inf")
            if weightvec is None
            else scipy.special.logsumexp(weightvec)
            for weightvec in new_weight_agg
        ],
        dtype=numpy.float64,
    )

    new_weights = normalize_log_log(new_weights)
    total_ll = numpy.asarray(total_ll_agg, dtype=numpy.float64).sum()

    return new_weights, total_ll


def model_em(
    model: FragModel,
    examples: collections.abc.Iterable[sgrpy.graph.CGraph],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
    useac: bool = False,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64]:
    """Perform one round of expectation-maximization.

    This function returns an unnormalized set of logweights for model fragments,
    after one round of expectation-maximization on the example graphs provided.
    Partitions are normalized to a probability of one for each example.

    Parameters
    ----------
    model : FragModel
        Fragment model to be tuned.
    examples : Iterable[CGraph]
        Graphs to be used for performing E-M.
    limit : None | int
        Maximum number of partitions per graph to generate.
    max_iter : None | int
        Maximum number of iterations to search.
    max_heap : None | int
        Maximum size of the search tree.
    useac : bool
        Use generic graph corrections to likelihood (default: True).
    progress : bool
        Print progress to command line via `tqdm` (default: False).

    Returns
    -------
    tuple[NDArray[float64],float64]
        First entry in tuple is the tuned logweights of fragments, second entry
        is the total loglikelihood of `examples`.
    """
    norm_groups = (
        ((ex,) for ex in tqdm.tqdm(examples))
        if progress
        else ((ex,) for ex in examples)
    )
    return model_em_groups(
        model, norm_groups, limit, max_iter, max_heap, useac, progress=False
    )


def model_em_future(
    model: FragModel,
    examples: collections.abc.Iterable[
        collections.abc.Iterable[sgrpy.graph.CGraph]
    ],
    weights: collections.abc.Iterable[float | int],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
    n_cores: int | None = None,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64]:
    """Perform one round of expectation-maximization.

    This function returns an unnormalized set of logweights for model fragments,
    after one round of expectation-maximization on the example graphs provided.
    Partitions are normalized to a probability of one for each example.

    Parameters
    ----------
    model : FragModel
        Fragment model to be tuned.
    examples : Iterable[CGraph]
        Graphs to be used for performing E-M.
    groups : Iterable[int]
        Normalization groups.
    group_weights : Mapping[int, int | float]
        Additional weights of individual values after ingroup normalization.
    limit : None | int
        Maximum number of partitions per graph to generate.
    max_iter : None | int
        Maximum number of iterations to search.
    max_heap : None | int
        Maximum size of the search tree.
    progress : bool
        Print progress to command line via `tqdm` (default: False).

    Returns
    -------
    tuple[NDArray[float64],float64]
        First entry in tuple is the tuned logweights of fragments, second entry
        is the total loglikelihood of `examples`.
    """
    parallel_obj = joblib.Parallel(
        n_jobs=n_cores, prefer="threads", return_as="generator"
    )
    part_func = functools.partial(
        perform_partition,
        model,
        limit=limit,
        max_iter=max_iter,
        max_heap=max_heap,
    )
    if progress:
        examples = tqdm.tqdm(examples, desc="Estimating E-M coefficients...")
    output_gen: collections.abc.Iterable[
        tuple[int, tuple[ObservationResult, ...]] | None
    ] = parallel_obj(
        joblib.delayed(part_func)((gi, cgv))
        for gi, cgv in itertools.chain.from_iterable(
            ((group_i, cg) for cg in cg_it)
            for group_i, cg_it in enumerate(examples)
        )
    )
    gweight_agg: list[float | int] = []
    ll_agg: list[numpy.float64] = []
    frag_weight_agg: list[None | list[numpy.float64]] = [
        None for _ in range(model.nof_fragments())
    ]
    for (_, group_packet), gweight in zip(
        itertools.groupby(
            (og for og in output_gen if og is not None),
            key=operator.itemgetter(0),
        ),
        weights,
        strict=True,
    ):
        gweight_agg.append(gweight)
        lgweight = numpy.float64(numpy.log(gweight))

        lweight_agg: list[numpy.float64] = []
        obs_agg: list[dict[int, int]] = []
        for _, obs_tup in group_packet:
            for obs in obs_tup:
                lweight_agg.append(obs.lweight)
                obs_agg.append(obs.counts)
        if len(lweight_agg) == 0:
            continue
        lweight_array = numpy.asarray(lweight_agg, dtype=numpy.float64)
        lweight_norm = normalize_log_log(lweight_array)
        ll_cur = numpy.float64(
            gweight * (numpy.exp(lweight_norm) @ lweight_array)
        )
        ll_agg.append(ll_cur)
        for final_weight, frag_dict in zip(lweight_norm, obs_agg, strict=True):
            for frag_id, frag_count in frag_dict.items():
                if frag_count < 1:
                    raise ValueError("Fragment count should be positive.")
                frag_lweight = numpy.float64(
                    lgweight + final_weight + numpy.log(frag_count)
                )
                fwa_entry = frag_weight_agg[frag_id]
                if fwa_entry is None:
                    frag_weight_agg[frag_id] = [frag_lweight]
                else:
                    fwa_entry.append(frag_lweight)

    fragment_weight_list = [
        numpy.float64("-inf") if fwl is None else scipy.special.logsumexp(fwl)
        for fwl in frag_weight_agg
    ]
    fragment_weight_array_unnorm = numpy.asarray(
        fragment_weight_list, dtype=numpy.float64
    )
    fragment_weight_array = normalize_log_log(fragment_weight_array_unnorm)

    total_ll = numpy.float64(
        numpy.sum(numpy.asarray(ll_agg, dtype=numpy.float64))
    )

    return fragment_weight_array, total_ll
