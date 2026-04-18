"""Tests for colormap component."""

import sgrpy


def test_chunking_1() -> None:
    """Test chunking of color map components."""
    chunks = (
        (1, 4, "E"),
        (0, 1, "E"),
        (1, 2, "A"),
        (3, 3, "A"),
        (1, 2, "B"),
        (3, 0, "A"),
        (1, 2, "E"),
        (2, 4, "C"),
        (4, 0, "D"),
        (0, 2, "B"),
    )
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=0)
    assert tuple(cmap._colormap) == ("A", "B", "E", "C", "D")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=1)
    assert tuple(cmap._colormap) == ("A", "B", "E", "C", "D")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=2)
    assert tuple(cmap._colormap) == ("A", "B", "E")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=3)
    assert tuple(cmap._colormap) == ()


def test_chunking_2() -> None:
    """Test chunking of color map components."""
    chunks = (
        (1, 4, "D"),
        (0, 0, "E"),
        (0, 3, "D"),
        (3, 2, "A"),
        (4, 3, "E"),
        (0, 4, "D"),
        (0, 4, "D"),
        (4, 4, "C"),
        (2, 1, "E"),
        (2, 2, "B"),
    )
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=0)
    assert tuple(cmap._colormap) == ("E", "D", "A", "B", "C")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=1)
    assert tuple(cmap._colormap) == ("E", "D", "A", "B", "C")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=2)
    assert tuple(cmap._colormap) == ("E", "D")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=3)
    assert tuple(cmap._colormap) == ("E",)
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=4)
    assert tuple(cmap._colormap) == ()


def test_chunking_3() -> None:
    """Test chunking of color map components."""
    chunks = (
        (3, 3, "D"),
        (0, 0, "D"),
        (1, 1, "C"),
        (0, 0, "E"),
        (1, 1, "E"),
        (2, 2, "B"),
        (4, 4, "D"),
        (4, 4, "B"),
        (2, 2, "A"),
        (1, 1, "A"),
    )
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=0)
    assert tuple(cmap._colormap) == ("D", "A", "B", "E", "C")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=1)
    assert tuple(cmap._colormap) == ("D", "A", "B", "E", "C")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=2)
    assert tuple(cmap._colormap) == ()


def test_chunking_4() -> None:
    """Test chunking of color map components."""
    chunks = (
        (4, 4, "E"),
        (1, 1, "B"),
        (1, 1, "C"),
        (3, 3, "C"),
        (4, 4, "A"),
        (3, 3, "E"),
        (2, 2, "C"),
        (2, 2, "E"),
        (3, 3, "B"),
        (4, 4, "D"),
    )
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=0)
    assert tuple(cmap._colormap) == ("C", "E", "B", "A", "D")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=1)
    assert tuple(cmap._colormap) == ("C", "E", "B", "A", "D")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=2)
    assert tuple(cmap._colormap) == ("C", "B", "E")
    cmap = sgrpy.model.ModelColorMapSimple.from_chunks(chunks, min_count=3)
    assert tuple(cmap._colormap) == ()
