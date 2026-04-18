"""Exception for rejecting a datapoint."""

from polars.exceptions import ComputeError


class ModelReject(Exception):
    """Raise when datapoint fails to pass model."""


def unpack_polars_error(
    err: ComputeError,
) -> ModelReject | None:
    if len(err.args) == 0:
        return None
    argstr: str = err.args[0]
    if not argstr.startswith("ModelReject"):
        return None

    prime_data = argstr[15:-2].split("', '")
    return ModelReject(*prime_data)
