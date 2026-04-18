"""Tests for permutation structures."""

import sgrpy


def test_identity() -> None:
    """Test identity permutation."""
    tup = (3, 1, 4, 5)
    mapping = sgrpy.graph.IndexMapping.identity(4)
    assert mapping.permute(tup) == (3, 1, 4, 5)


def test_identity_minus() -> None:
    """Test minus operation."""
    tup = (3, 1, 4, 5)
    mapping = sgrpy.graph.IndexMapping.identity_minus(4, 2)
    assert mapping.permute(tup) == (3, 1, 5)


def test_append() -> None:
    """Test append operation."""
    tup = (3, 1, 4, 5)
    mapping = sgrpy.graph.IndexMapping.identity(3).append(0)
    assert mapping.permute(tup) == (3, 1, 4, 3)


def test_subst() -> None:
    """Test substitution operation."""
    tup = (3, 1, 4, 5)
    mapping = sgrpy.graph.IndexMapping.identity(4).subst(0, 3)
    assert mapping.permute(tup) == (5, 1, 4, 5)


def test_inv() -> None:
    """Test inversion operation."""
    mapping = sgrpy.graph.IndexMapping.from_seq((4, 3, 1, 4, None)).inv()
    assert mapping.as_tuple() == (None, 2, None, 1, 0)


def test_compose() -> None:
    """Test composition operation."""
    mapAB = sgrpy.graph.IndexMapping.from_seq((4, 3, 1, 4, None))
    mapBC = sgrpy.graph.IndexMapping.from_seq((0, 5, 2, 8))
    assert mapBC.compose(mapAB).as_tuple() == (4, None, 1, None)


def test_permute() -> None:
    """Test permutation operation."""
    tup = (3, 1, 4, 5)
    mapAB = sgrpy.graph.IndexMapping.from_seq((0, 3, 1, 0))
    assert tuple(mapAB.permute(tup)) == (3, 5, 1, 3)


def test_perfect() -> None:
    """Test check for perfect permutation."""
    test_cases = [
        ((0, 3, 1, 0), False),
        ((0, 3, 1, 2), True),
        ((0, None, 1, 2), False),
        ((0, 0, 1, 2), False),
        ((None,), False),
        ((), True),
        ((0, 3, 1, 2), True),
        ((0, 3, 1, 2), True),
    ]
    for test_case, test_val in test_cases:
        mapping = sgrpy.graph.IndexMapping.from_seq(test_case)
        assert mapping.is_permutation() is test_val
