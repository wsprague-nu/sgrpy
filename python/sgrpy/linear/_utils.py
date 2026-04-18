"""Sparse vector/array utility functions."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy
import scipy

from sgrpy.iotypes import JSON, json_to_dict


def csr_to_json(csr: scipy.sparse.csr_array) -> JSON:
    """Convert 2-D scipy Compressed Sparse Row array to JSON form.

    Parameters
    ----------
    csr : scipy.sparse.csr_array
        Array to be converted.

    Returns
    -------
    JSON
    """
    shape0, shape1 = csr.shape
    data: list[int | float] = csr.data.tolist()
    indices: list[int] = csr.indices.tolist()
    indptr: list[int] = csr.indptr.tolist()
    results = {
        "shape0": shape0,
        "shape1": shape1,
        "data": data,
        "indices": indices,
        "indptr": indptr,
    }
    return results


def dict_to_csrvecint(
    data_in: Mapping[str, Sequence[int]], size: int
) -> scipy.sparse.csr_array:
    """Initialize scipy Compressed Sparse Row vector from dict input.

    Ultimate shape is (1, size).

    Parameters
    ----------
    data_in : Mapping[str, Sequence[int]]
        Integer-valued data input for initialization.
    size : int
        Number of entries (columns) in vector.
    """
    data = data_in["data"]
    indices = data_in["indices"]
    indptr = data_in["indptr"]
    csr_mat = scipy.sparse.csr_array(
        (data, indices, indptr), shape=(1, size), dtype=numpy.int64
    )
    return csr_mat


def json_to_csr(json: JSON) -> scipy.sparse.csr_array:
    """Initialize 2-D scipy Compressed Sparse Row array from JSON form.

    See code for `csr_to_json` for expected JSON structure.

    Parameters
    ----------
    json : JSON
        Data containing array information.

    Returns
    -------
    scipy.sparse.csr_array
    """
    json_dict = json_to_dict(json)
    if not isinstance(json_dict["shape0"], int):
        raise TypeError("`shape0` was not an integer")
    if not isinstance(json_dict["shape1"], int):
        raise TypeError("`shape1` was not an integer")
    shape0 = int(json_dict["shape0"])
    shape1 = int(json_dict["shape1"])
    csr_mat = scipy.sparse.csr_array(
        (json_dict["data"], json_dict["indices"], json_dict["indptr"]),
        shape=(shape0, shape1),
    )
    return csr_mat


def np_to_json(array: numpy.typing.NDArray[Any]) -> JSON:
    """Convert numpy array to JSON form.

    Parameters
    ----------
    array : numpy.typing.NDArray[Any]
        Array to be converted.

    Returns
    -------
    JSON
        At the moment, result of `array.tolist()`.
    """
    result: list[JSON] = array.tolist()
    return result


def json_to_np(json: JSON) -> numpy.typing.NDArray[Any]:
    """Initialize numpy array from input JSON.

    Parameters
    ----------
    json : JSON
        Input data to be converted.

    Returns
    -------
    numpy.typing.NDArray[Any]
        At the moment, the entry type is inferred.
    """
    array = numpy.asarray(json)
    return array


def csr_equal(
    array1: scipy.sparse.csr_array, array2: scipy.sparse.csr_array
) -> bool:
    """Compare two scipy Compressed Sparse Row arrays for equality.

    May not return consistent answer if array is not sorted or contains
    duplicates.

    Parameters
    ----------
    array1 : scipy.sparse.csr_array
        First array for comparison.
    array2 : scipy.sparse.csr_array
        Second array for comparison.
    """
    return (
        numpy.array_equal(array1.data, array2.data)
        and numpy.array_equal(array1.indices, array2.indices)
        and numpy.array_equal(array1.indptr, array2.indptr)
    )
