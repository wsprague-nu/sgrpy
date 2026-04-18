"""JSON type annotation and functions."""

import json
import pathlib
import zlib
from collections.abc import Mapping, Sequence
from typing import TypeAlias

JSON: TypeAlias = (
    Mapping[str, "JSON"] | Sequence["JSON"] | str | int | float | bool | None
)
"""Generic JSON type."""


def json_to_dict(data: JSON) -> dict[str, JSON]:
    """No-copy type guard forcing conversion to dict, otherwise raising error.

    Would probably be faster to use intermediate conversion to Any instead of
    calling this function.

    Parameters
    ----------
    data : JSON
        Input data, should be a dict[str, JSON]

    Returns
    -------
    dict[str, JSON]
        Same value as `data`, if `data` is the right type.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)} when unpacking JSON")
    return data


def json_to_tuple(data: JSON) -> tuple[JSON, ...]:
    """Type guard forcing conversion to tuple, otherwise raising error.

    Would probably be faster to use intermediate conversion to Any instead of
    calling this function.

    Parameters
    ----------
    data : JSON
        Input data, should be a list[JSON]

    Returns
    -------
    tuple[JSON, ...]
        Same internal values as `data`, if `data` is the right type.
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)} when unpacking JSON")
    return tuple(data)


def json_to_list(data: JSON) -> list[JSON]:
    """No-copy type guard forcing conversion to tuple, otherwise raising error.

    Would probably be faster to use intermediate conversion to Any instead of
    calling this function.

    Parameters
    ----------
    data : JSON
        Input data, should be a list[JSON]

    Returns
    -------
    list[JSON]
        Same internal values as `data`, if `data` is the right type.
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)} when unpacking JSON")
    return data


def json_write_compressed(data: JSON, path: pathlib.Path | str) -> None:
    """Compress input JSON with zlib and write to a file.

    Parameters
    ----------
    data : JSON
        Data to be compressed.
    path : pathlib.Path | str
        Output path to be written to.
    """
    json_bytes = json.dumps(data).encode("utf-8")
    json_zip = zlib.compress(json_bytes)
    with open(path, "wb") as fout:
        fout.write(json_zip)


def json_read_compressed(path: pathlib.Path | str) -> JSON:
    """Read zlib-compressed, UTF-8 encoded JSON from file and return.

    Parameters
    ----------
    path : pathlib.Path | str
        Location of file.

    Returns
    -------
    JSON
        Data content of file.
    """
    with open(path, "rb") as fin:
        json_zip = fin.read()
    json_bytes = zlib.decompress(json_zip)
    json_data: JSON = json.loads(json_bytes.decode("utf-8"))
    return json_data
