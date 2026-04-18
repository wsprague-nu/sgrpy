"""Functions for normalizing weight vectors."""

import numpy
import numpy.typing
import scipy


def normalize_log_log(
    weights: numpy.typing.NDArray[numpy.float64],
) -> numpy.typing.NDArray[numpy.float64]:
    """Normalize logweight vector to logprob vector."""
    # if vector is empty, return empty
    if len(weights) == 0:
        return weights.copy()

    # if any NaN values, return all NaN
    if numpy.isnan(weights).any():
        return numpy.full(weights.shape, numpy.nan, dtype=numpy.float64)

    # if any positive infinity, do a quick normalization first
    posinf_vec = numpy.isposinf(weights)
    if posinf_vec.any():
        weights = numpy.full(weights.shape, -numpy.inf, dtype=numpy.float64)
        weights[posinf_vec] = 0

    # if all negative infinity, perform full normalization
    neginf_vec = numpy.isneginf(weights)
    if neginf_vec.all():
        weights = numpy.zeros(weights.shape, dtype=numpy.float64)

    # use scipy logsumexp to find normalization constant
    norm_const = scipy.special.logsumexp(weights)

    # normalize via subtraction
    norm_weights: numpy.typing.NDArray[numpy.float64] = weights - norm_const

    return norm_weights


def normalize_int_log(
    weights: numpy.typing.NDArray[numpy.uintp],
) -> numpy.typing.NDArray[numpy.float64]:
    """Normalize integer weight vector to logprob vector."""
    logweight = numpy.log(weights)

    return normalize_log_log(logweight)
