"""Basic Heap tests."""

import dataclasses

from .._basic_heap import BasicHeap


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class _HeapEntryTest:
    rank: int
    label: int = dataclasses.field(compare=False)


def test_BasicHeap_ordering() -> None:
    """Test basic ordering functionality."""
    example_vals = [
        (1, 5),
        (9, 1),
        (2, 6),
    ]

    entries = [_HeapEntryTest(rank=-x, label=y) for x, y in example_vals]

    heap: BasicHeap[_HeapEntryTest] = BasicHeap.new()
    for e in entries:
        heap.push(e)

    output = tuple((-entry.rank, entry.label) for entry in heap.drain())

    assert output == ((9, 1), (2, 6), (1, 5))


def test_BasicHeap_insertion_order() -> None:
    """Test that heap maintains insertion order."""
    example_vals = (
        (1, 5),
        (9, 1),
        (2, 6),
        (3, 2),
        (6, 3),
        (3, 9),
        (5, 9),
        (9, 1),
        (9, 8),
        (3, 3),
    )

    entries = [_HeapEntryTest(rank=-x, label=y) for x, y in example_vals]

    heap: BasicHeap[_HeapEntryTest] = BasicHeap.new()
    for e in entries:
        heap.push(e)

    output = tuple((-entry.rank, entry.label) for entry in heap.drain())

    assert output == (
        (9, 1),
        (9, 1),
        (9, 8),
        (6, 3),
        (5, 9),
        (3, 2),
        (3, 9),
        (3, 3),
        (2, 6),
        (1, 5),
    )


def test_BasicHeap_empty() -> None:
    """Test empty heap outputs."""
    heap: BasicHeap[_HeapEntryTest] = BasicHeap.new()
    output = tuple(heap.drain())
    assert output == ()


def test_BasicHeap_large() -> None:
    """Test heap with many entries."""
    heap: BasicHeap[_HeapEntryTest] = BasicHeap.new()
    example_vals = [
        (1, 5),
        (9, 1),
        (2, 6),
        (3, 2),
        (6, 3),
        (3, 9),
        (5, 9),
        (9, 1),
        (9, 8),
        (3, 3),
        (4, 3),
        (5, 8),
        (5, 7),
        (3, 8),
        (7, 5),
        (7, 6),
        (1, 5),
        (7, 3),
        (7, 2),
        (0, 6),
        (4, 5),
        (1, 3),
        (5, 2),
        (2, 9),
        (6, 6),
        (2, 2),
        (7, 6),
        (6, 9),
        (6, 1),
        (5, 6),
        (2, 3),
        (6, 3),
        (6, 8),
        (7, 7),
        (9, 1),
        (6, 9),
        (3, 3),
        (6, 8),
        (3, 8),
        (9, 7),
        (3, 9),
        (8, 0),
        (0, 7),
        (1, 3),
        (5, 2),
        (5, 7),
        (3, 6),
        (7, 1),
        (9, 0),
        (3, 1),
    ]

    entries = [_HeapEntryTest(rank=-x, label=y) for x, y in example_vals]

    for e in entries:
        heap.push(e)

    output = tuple((-entry.rank, entry.label) for entry in heap.drain())

    expected = (
        (9, 1),
        (9, 1),
        (9, 8),
        (9, 1),
        (9, 7),
        (9, 0),
        (8, 0),
        (7, 5),
        (7, 6),
        (7, 3),
        (7, 2),
        (7, 6),
        (7, 7),
        (7, 1),
        (6, 3),
        (6, 6),
        (6, 9),
        (6, 1),
        (6, 3),
        (6, 8),
        (6, 9),
        (6, 8),
        (5, 9),
        (5, 8),
        (5, 7),
        (5, 2),
        (5, 6),
        (5, 2),
        (5, 7),
        (4, 3),
        (4, 5),
        (3, 2),
        (3, 9),
        (3, 3),
        (3, 8),
        (3, 3),
        (3, 8),
        (3, 9),
        (3, 6),
        (3, 1),
        (2, 6),
        (2, 9),
        (2, 2),
        (2, 3),
        (1, 5),
        (1, 5),
        (1, 3),
        (1, 3),
        (0, 6),
        (0, 7),
    )

    assert output == expected
