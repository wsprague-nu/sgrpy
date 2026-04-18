"""Loss functions for fragment model."""

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


def _calc_q_i(p_s: numpy.float64, w_i: numpy.float64, n: int) -> numpy.float64:
    """Calculate ln(q_i) from paper."""
    return_val: numpy.float64 = p_s - n * numpy.log1p(-numpy.exp(w_i))
    return return_val


def model_loss_groups(
    model: FragModel,
    norm_groups: collections.abc.Iterable[
        collections.abc.Iterable[sgrpy.graph.CGraph]
    ],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
    useac: bool = False,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64, numpy.float64]:
    """Estimate greedy loss of removing each fragment.

    This function returns estimates of the loss obtained by removing each
    fragment in `model` from the model, given the example graphs provided.
    Partitions are normalized to a probability of one for each norm group.

    Parameters
    ----------
    model : FragModel
        Fragment model to be tuned.
    examples : Iterable[Iterable[CGraph]]
        Normalization groups to be used for estimating loss.
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
    tuple[NDArray[float64],float64,float64]
        First entry in tuple is the tuned logweights of fragments, second entry
        is the total loglikelihood of `examples`, third entry is the total
        library loglikelihood.
    """
    loss_vec = numpy.zeros(model.nof_fragments(), dtype=numpy.float64)
    group_iter = tqdm.tqdm(norm_groups) if progress else norm_groups
    total_ll_agg: list[numpy.float64] = []
    lossmodweights = numpy.log1p(-numpy.exp(model.weights))
    for group in group_iter:
        loss_agg: numpy.typing.NDArray[numpy.float64] = numpy.full(
            model.nof_fragments(), numpy.float64("-inf"), dtype=numpy.float64
        )
        prob_agg: list[numpy.float64] = []
        tallied = False
        for graph in group:
            tallied = True
            for observation in model.partition(
                graph, limit, max_iter, max_heap, useac
            ):
                labels_used = frozenset(
                    m.fragment_id for m in observation.matches
                )
                obs_size = len(observation.matches)

                prob_agg.append(observation.weight)

                # tabulate counts and probabilities
                p_s = observation.weight

                q_vec = p_s - obs_size * lossmodweights
                for i in labels_used:
                    q_vec[i] = numpy.float64("-inf")
                loss_agg = scipy.special.logsumexp([loss_agg, q_vec], axis=0)

        if not tallied:
            continue

        # normalize q_i values
        loss_vec_group = loss_agg

        # add total likelihood to aggregator
        total_ll_agg.append(scipy.special.logsumexp(prob_agg))

        # add normalized q_i values to lossvec
        loss_vec += loss_vec_group

    # add model losses
    color_weights = model.calc_color_weights()
    library_agg: list[numpy.float64] = []
    for i in range(len(loss_vec)):
        library_ll = model.calc_ll_fragment(
            i, color_weights, use_struc_fac=useac
        )
        loss_vec[i] -= library_ll
        library_agg.append(library_ll)

    # calculate total ll
    total_ll = numpy.asarray(total_ll_agg, dtype=numpy.float64).sum()

    # calculate library ll
    library_total_ll = numpy.asarray(library_agg, dtype=numpy.float64).sum()

    return loss_vec, total_ll, library_total_ll


def model_loss(
    model: FragModel,
    examples: collections.abc.Iterable[sgrpy.graph.CGraph],
    limit: None | int,
    max_iter: None | int,
    max_heap: None | int,
    useac: bool = False,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64, numpy.float64]:
    """Estimate greedy loss of removing each fragment.

    This function returns estimates of the loss obtained by removing each
    fragment in `model` from the model, given the example graphs provided.
    Partitions are normalized to a probability of one for each example.

    Parameters
    ----------
    model : FragModel
        Fragment model to be tuned.
    examples : Iterable[CGraph]
        Graphs to be used for estimating loss.
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
    tuple[NDArray[float64],float64,float64]
        First entry in tuple is the tuned logweights of fragments, second entry
        is the total loglikelihood of `examples`, third entry is the total
        library loglikelihood.
    """
    norm_groups = (
        ((ex,) for ex in tqdm.tqdm(examples))
        if progress
        else ((ex,) for ex in examples)
    )
    return model_loss_groups(
        model, norm_groups, limit, max_iter, max_heap, useac, progress=False
    )


def model_loss_future(
    model: FragModel,
    examples: collections.abc.Iterable[
        collections.abc.Iterable[sgrpy.graph.CGraph]
    ],
    weights: collections.abc.Iterable[float | int],
    limit: int | None,
    max_iter: int | None,
    max_heap: int | None,
    n_cores: int | None = None,
    progress: bool = False,
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.float64, numpy.float64]:
    # initialize loss vector and related quantities
    loss_vec = numpy.zeros(model.nof_fragments(), dtype=numpy.float64)
    total_ll_agg: list[numpy.float64] = []
    lossmodweights = numpy.log1p(-numpy.exp(model.weights))

    # initialize parallelization objects
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
        examples = tqdm.tqdm(examples, desc="Estimating fragment losses...")
    output_gen: collections.abc.Iterable[
        tuple[int, tuple[ObservationResult, ...]] | None
    ] = parallel_obj(
        joblib.delayed(part_func)((gi, cgv))
        for gi, cgv in itertools.chain.from_iterable(
            ((group_i, cg) for cg in cg_it)
            for group_i, cg_it in enumerate(examples)
        )
    )

    for (_, group_packet), gweight in zip(
        itertools.groupby(
            (og for og in output_gen if og is not None),
            key=operator.itemgetter(0),
        ),
        weights,
        strict=True,
    ):
        loss_agg: numpy.typing.NDArray[numpy.float64] = numpy.full(
            model.nof_fragments(), numpy.float64("-inf"), dtype=numpy.float64
        )
        prob_agg: list[numpy.float64] = []
        for _, obs_tup in group_packet:
            obs: ObservationResult
            for obs in obs_tup:
                labels_used = frozenset(obs.counts)
                obs_size = sum(obs.counts.values())

                prob_agg.append(obs.lweight)

                # tabulate counts and probabilities
                p_s = obs.lweight

                # adjust logprobability for case of removing each component
                q_vec = p_s - obs_size * lossmodweights
                # including if the component is present
                for i in labels_used:
                    q_vec[i] = numpy.float64("-inf")

                loss_agg = scipy.special.logsumexp([loss_agg, q_vec], axis=0)

        # include group weighting
        loss_vec_group = loss_agg * gweight

        # add total likelihood to aggregator
        total_ll_agg.append(scipy.special.logsumexp(prob_agg) * gweight)

        # add normalized q_i values to lossvec
        loss_vec += loss_vec_group

    # add model losses
    color_weights = model.calc_color_weights()
    library_agg: list[numpy.float64] = []
    for i in range(len(loss_vec)):
        library_ll = model.calc_ll_fragment(
            i, color_weights, use_struc_fac=False
        )
        loss_vec[i] -= library_ll
        library_agg.append(library_ll)

    # calculate total ll
    total_ll = numpy.asarray(total_ll_agg, dtype=numpy.float64).sum()

    # calculate library ll
    library_total_ll = numpy.asarray(library_agg, dtype=numpy.float64).sum()

    return loss_vec, total_ll, library_total_ll
