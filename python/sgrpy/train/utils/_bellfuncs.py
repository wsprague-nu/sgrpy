"""Bell number estimation functions."""

import functools
import math

import numpy
import scipy


@functools.lru_cache(100)
def lnbell_est(n: int) -> float:
    """Estimate natural logarithm of the Nth Bell number."""
    return float(n * (numpy.log(0.792 * n) - numpy.log(numpy.log(n + 1))))


def lnbell_rat(n: int, m: int, count: int) -> float:
    return lnbell_est(n) - lnbell_est(m) + math.log(count)


def lnfac_unchecked(n: int) -> float:
    if n <= 0:
        return 0.0
    else:
        return float(scipy.special.gammaln(n + 1))


def lndfac_unchecked(n: int) -> float:
    if n <= 1:
        return 0.0
    else:
        return float(
            n * math.log(2)
            - math.log(math.pi) / 2
            + scipy.special.gammaln(n + 0.5)
        )
