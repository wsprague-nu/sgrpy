"""Tests for normalization functions."""

import numpy

from .._normalize_weights import normalize_log_log


def test_normalize_log_log_empty() -> None:
    """Test normalization of empty vector."""
    weights: list[float] = []

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected: list[float] = []

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert (
        array.shape == array_expected.shape and (result == array_expected).all()
    )


def test_normalize_log_log_single_zero() -> None:
    """Test normalization of singlet vector."""
    weights = [0]

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected = [0]

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert (
        array.shape == array_expected.shape and (result == array_expected).all()
    )


def test_normalize_log_log_single_one() -> None:
    """Test normalization of singlet vector."""
    weights = [1]

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected = [0]

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert (
        array.shape == array_expected.shape and (result == array_expected).all()
    )


def test_normalize_log_log_single_nan() -> None:
    """Test normalization of singlet vector."""
    weights = [float("nan")]

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected = [float("nan")]

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert array.shape == array_expected.shape and numpy.isnan(result).all()


def test_normalize_log_log_single_posinf() -> None:
    """Test normalization of singlet vector."""
    weights = [float("inf")]

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected = [0]

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert (
        array.shape == array_expected.shape and (result == array_expected).all()
    )


def test_normalize_log_log_single_neginf() -> None:
    """Test normalization of singlet vector."""
    weights = [-float("inf")]

    array = numpy.asarray(weights, dtype=numpy.float64)

    result = normalize_log_log(array)

    expected = [0]

    array_expected = numpy.asarray(expected, dtype=numpy.float64)

    assert (
        array.shape == array_expected.shape and (result == array_expected).all()
    )


def test_normalize_log_log_two() -> None:
    """Test normalization of various combinations of two components."""
    posinf = float("inf")
    neginf = -posinf
    nanfloat = float("nan")
    weights = (
        ((posinf, posinf), (-0.69314718, -0.69314718)),
        ((posinf, neginf), (0, neginf)),
        ((posinf, nanfloat), (nanfloat, nanfloat)),
        ((posinf, 0), (0, neginf)),
        ((posinf, 1), (0, neginf)),
        ((neginf, posinf), (neginf, 0)),
        ((neginf, neginf), (-0.69314718, -0.69314718)),
        ((neginf, nanfloat), (nanfloat, nanfloat)),
        ((neginf, 0), (neginf, 0)),
        ((neginf, 1), (neginf, 0)),
        ((nanfloat, posinf), (nanfloat, nanfloat)),
        ((nanfloat, neginf), (nanfloat, nanfloat)),
        ((nanfloat, nanfloat), (nanfloat, nanfloat)),
        ((nanfloat, 0), (nanfloat, nanfloat)),
        ((nanfloat, 1), (nanfloat, nanfloat)),
        ((0, posinf), (neginf, 0)),
        ((0, neginf), (0, neginf)),
        ((0, nanfloat), (nanfloat, nanfloat)),
        ((0, 0), (-0.69314718, -0.69314718)),
        ((0, 1), (-1.31326169, -0.31326169)),
        ((1, posinf), (neginf, 0)),
        ((1, neginf), (0, neginf)),
        ((1, nanfloat), (nanfloat, nanfloat)),
        ((1, 0), (-0.31326169, -1.31326169)),
        ((1, 1), (-0.69314718, -0.69314718)),
    )

    for array, expected in (
        (
            numpy.asarray(weight_input, dtype=numpy.float64),
            numpy.asarray(weight_output, dtype=numpy.float64),
        )
        for weight_input, weight_output in weights
    ):
        result = normalize_log_log(array)

        assert numpy.allclose(result, expected, equal_nan=True)


def test_normalize_random() -> None:
    """Test normalization of random components."""
    test_seed = 153334493705858486410082890909922861529
    vec_size = 64
    num_tests = 64

    random_gen = numpy.random.default_rng(test_seed)

    test_inputs = tuple(
        numpy.asarray(random_gen.standard_cauchy(vec_size), dtype=numpy.float64)
        for _ in range(num_tests)
    )

    for test_input in test_inputs:
        test_norm = normalize_log_log(test_input)

        # check that sum is approximately 1
        test_norm_sum = numpy.exp(test_norm).sum()
        assert numpy.allclose([test_norm_sum], [1.0])

        # check that ratio is approximately the same
        ratios = test_input - test_norm
        ratio_1 = numpy.full_like(test_norm, ratios[0], dtype=numpy.float64)
        assert numpy.allclose(ratios, ratio_1)
