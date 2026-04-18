"""Generalized routines for training a FragModel."""

import logging
from collections.abc import Iterable, Sequence

import numpy

from sgrpy.graph import CGraph

from ._em_funcs import model_em_future
from ._fragmodel import FragModel
from ._loss_funcs import model_loss_future


def _train_em(
    cgraphs: Sequence[Sequence[CGraph]],
    weights: Iterable[float | int],
    model: FragModel,
    limit: int | None,
    max_iter: int | None,
    max_heap: int | None,
    wllimit: float,
    n_cores: int | None,
    progress: bool,
) -> tuple[FragModel, numpy.float64]:
    for cgraph_seq in cgraphs:
        for cgraph in cgraph_seq:
            n_comp = cgraph.nof_components()
            if n_comp > 1:
                print(cgraph)
                raise ValueError("FAIL")
    new_weights, new_likelihood = model_em_future(
        model=model,
        examples=cgraphs,
        weights=weights,
        limit=limit,
        max_iter=max_iter,
        max_heap=max_heap,
        n_cores=n_cores,
        progress=progress,
    )
    for i, weight in enumerate(new_weights):
        if weight < numpy.float64(wllimit):
            new_weights[i] = numpy.float64(wllimit)
    new_model = model.with_weights(new_weights)
    return new_model, new_likelihood


def _prune_model(
    cgraphs: Sequence[Sequence[CGraph]],
    weights: Iterable[float | int],
    model: FragModel,
    limit: int | None,
    max_iter: int | None,
    max_heap: int | None,
    prune_frac: float,
    n_cores: int | None,
    progress: bool,
) -> tuple[FragModel, numpy.float64]:
    prune_min = max(int(model.nof_fragments() * prune_frac), 1)
    loss_vals, new_likelihood, lib_likelihood = model_loss_future(
        model=model,
        examples=cgraphs,
        weights=weights,
        limit=limit,
        max_iter=max_iter,
        max_heap=max_heap,
        n_cores=n_cores,
        progress=progress,
    )
    total_ll = new_likelihood + lib_likelihood
    remove_priority: list[tuple[numpy.float64, numpy.float64, int]] = sorted(
        (
            (loss, -weight, i)
            for i, (loss, weight) in enumerate(
                zip(loss_vals, model.weights, strict=True)
            )
        ),
        reverse=True,
    )
    remove_set: set[int] = set()
    for _, _, i in remove_priority:
        if model.fragtree.get_fragment(i).to_canongraph().nof_nodes() == 1:
            continue
        if len(remove_set) == 0 or len(remove_set) < prune_min:
            remove_set.add(i)
        else:
            break

    for remove_frag in remove_set:
        frag_remove = model.fragtree.get_fragment(remove_frag)
        logging.debug(f"Pruning fragment {frag_remove}")

    pruned_model = model.with_subset(
        i for i in range(model.nof_fragments()) if i not in remove_set
    )

    return pruned_model, total_ll


def _prune_train(
    cgraphs: Sequence[Sequence[CGraph]],
    model: FragModel,
    weights: Iterable[float | int],
    limit: int | None,
    max_iter: int | None,
    max_heap: int | None,
    train_iter: int,
    wllimit: float,
    prune_frac: float,
    n_cores: int | None,
    progress: bool,
) -> tuple[FragModel, numpy.float64]:
    new_model, ret_ll = _prune_model(
        cgraphs=cgraphs,
        weights=weights,
        model=model,
        limit=limit,
        max_iter=max_iter,
        max_heap=max_heap,
        prune_frac=prune_frac,
        n_cores=n_cores,
        progress=progress,
    )
    if new_model.nof_fragments() == model.nof_fragments():
        return new_model, ret_ll
    for _ in range(train_iter):
        new_model, _ = _train_em(
            cgraphs=cgraphs,
            weights=weights,
            model=model,
            limit=limit,
            max_iter=max_iter,
            max_heap=max_heap,
            wllimit=wllimit,
            n_cores=n_cores,
            progress=progress,
        )
    return new_model, ret_ll


def tune_model_classical(
    model: FragModel,
    cgraphs: Sequence[Sequence[CGraph]],
    group_weights: Sequence[float | int],
    train_iter: int = 5,
    prune_frac: float = 0.2,
    wllimit: float = -100.0,
    plimit: int | None = 64,
    max_iter: int | None = 65536,
    max_heap: int | None = 65536,
    n_jobs: int | None = None,
    progress: bool = False,
) -> FragModel:
    # train initial model using E-M algorithm
    cur_model = model
    for _ in range(train_iter):
        cur_model, _ = _train_em(
            cgraphs=cgraphs,
            weights=group_weights,
            model=cur_model,
            limit=plimit,
            max_iter=max_iter,
            max_heap=max_heap,
            wllimit=wllimit,
            n_cores=n_jobs,
            progress=progress,
        )

    if prune_frac <= 0.0:
        return cur_model

    model_minus2: None | FragModel = None
    model_minus1: None | FragModel = None

    # begin training loop until optimized
    ll_vals: list[numpy.float64] = []
    while model_minus2 is None or ll_vals[-2] <= ll_vals[-1]:
        new_model, cur_ll_val = _prune_train(
            cgraphs=cgraphs,
            model=cur_model,
            weights=group_weights,
            limit=plimit,
            max_iter=max_iter,
            max_heap=max_heap,
            train_iter=train_iter,
            wllimit=wllimit,
            prune_frac=prune_frac,
            n_cores=n_jobs,
            progress=progress,
        )
        ll_vals.append(cur_ll_val)
        model_minus2 = model_minus1
        model_minus1 = cur_model
        cur_model = new_model
        if (
            model_minus2 is not None
            and model_minus1.nof_fragments() == model_minus2.nof_fragments()
        ):
            return model_minus1

    return model_minus2
