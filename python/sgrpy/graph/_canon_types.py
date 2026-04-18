"""Canonicalization types."""

import enum
import typing


@typing.final
class CanonType(enum.StrEnum):
    """Canonization type, defined by `igraph` package."""

    F = "f"
    """First non-singleton cell."""
    FL = "fl"
    """First largest non-singleton cell."""
    FS = "fs"
    """First smallest non-singleton cell."""
    FM = "fm"
    """First maximally non-trivially connected non-singleton cell."""
    FLM = "flm"
    """Largest maximally non-trivially connected non-singleton cell."""
    FSM = "fsm"
    """Smallest maximally non-trivially connected non-singleton cell."""
