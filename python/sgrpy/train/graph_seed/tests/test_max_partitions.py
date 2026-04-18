"""Test calc_max_covers() function."""

from .._max_partitions import calc_max_covers


def test_calc_max_covers_empty() -> None:
    """Test max covers should return None for empty set."""
    test_set = ()
    assert calc_max_covers(test_set) == 0


def test_calc_max_covers_single() -> None:
    """Test max covers should return singlet for singlet set."""
    test_set = ((1,),)
    assert calc_max_covers(test_set) == 1


def test_calc_max_covers_multi_1() -> None:
    """Test max covers should return two for given set."""
    test_set = ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (3, 6))
    assert calc_max_covers(test_set) == 2  # noqa: PLR2004


def test_calc_max_covers_multi_2() -> None:
    """Test max covers should return two for given set."""
    test_set = (
        (0, 1, 2),
        (2, 3, 4),
        (2, 3, 5),
        (2, 3, 6),
        (4, 3, 5),
        (4, 3, 6),
        (5, 3, 6),
    )
    assert calc_max_covers(test_set) == 2  # noqa: PLR2004


def test_calc_max_covers_multi_3() -> None:
    """Test max covers should return one for given set."""
    test_set = (
        (2, 3, 4, 5),
        (2, 3, 4, 6),
        (2, 3, 5, 6),
        (3, 4, 5, 6),
    )
    assert calc_max_covers(test_set) == 1
