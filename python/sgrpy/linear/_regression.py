"""Linear regression functions."""

import logging
from typing import Any

import igraph
import numpy
import scipy

_F64_EPS = numpy.finfo(numpy.float64).eps


def scale_gram_csr(
    csr_array: scipy.sparse.csr_array,
) -> scipy.sparse.csr_array:
    """Perform column scaling on training matrix, then obtain Gram matrix.

    Parameters
    ----------
    csr_array : scipy.sparse.csr_array
        Input array, can be of any dimension.

    Returns
    -------
    scipy.sparse.csr_array
        Gram matrix with scaled rows and columns
    """
    col_norms = scipy.sparse.linalg.norm(csr_array, axis=1)
    col_norms[col_norms == 0] = 0.0
    array_scaled = csr_array.T * numpy.reciprocal(col_norms)
    array_gram = array_scaled @ array_scaled.T
    return array_gram


def scale_csr(
    csr_array: scipy.sparse.csr_array,
) -> tuple[scipy.sparse.csr_array, numpy.typing.NDArray[Any]]:
    """Calculate scaled Gram matrix and column scale factors.

    Parameters
    ----------
    csr_array : scipy.sparse.csr_array
        Input array, can be of any dimension.

    Returns
    -------
    tuple[scipy.sparse.csr_array, numpy.typing.NDArray[Any]]
        First entry is Gram matrix with scaled rows and columns.  Second entry
        is column norms (to be used when checking later for whether an arbitrary
        vector is present in the kernel of `csr_array`).
    """
    col_norms = scipy.sparse.linalg.norm(csr_array, axis=0)
    col_norms[col_norms == 0] = numpy.float64(1.0)
    array_scaled = (csr_array * numpy.reciprocal(col_norms)).tocsr()
    return array_scaled, col_norms


def _scale_coo(
    coo_array: scipy.sparse.coo_array,
    col_norm_recip: numpy.typing.NDArray[numpy.float64],
) -> scipy.sparse.coo_array:
    new_data = [
        val * (col_norm_recip[i] * col_norm_recip[j])
        for val, i, j in zip(
            coo_array.data,
            coo_array.coords[0],
            coo_array.coords[1],
            strict=True,
        )
    ]
    new_array = scipy.sparse.coo_array(
        (new_data, coo_array.coords), shape=coo_array.shape, dtype=numpy.float64
    )
    return new_array


def scale_gram_sparse(
    sp_array: scipy.sparse.sparray,
) -> tuple[scipy.sparse.coo_array, numpy.typing.NDArray[numpy.float64]]:
    """Calculate scaled Gram matrix and column scale factors.

    Parameters
    ----------
    sp_array : scipy.sparse.sparray
        Sparse input array, can be of any dimension.

    Returns
    -------
    tuple[scipy.sparse.coo_array, numpy.typing.NDArray[numpy.float64]]
        First entry is Gram matrix with scaled rows and columns.  Second entry
        is column norms (to be used when checking later for whether an arbitrary
        vector is present in the kernel of `csr_array`).
    """
    col_norms = scipy.sparse.linalg.norm(sp_array, axis=0)
    col_norms[col_norms == 0] = numpy.float64(1.0)
    col_norm_recip = numpy.reciprocal(col_norms)
    gram_array = sp_array.T @ sp_array
    final_array = _scale_coo(gram_array.tocoo(), col_norm_recip)
    return final_array, col_norm_recip


def within_kern(
    scaled_covar: scipy.sparse.csr_array,
    vec: numpy.typing.NDArray[Any],
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> bool:
    """Check whether `vec` lies within the column space of `scaled_covar`.

    .. deprecated:: 0.1.16
        `within_kern` is deprecated due to its slow speed on relevant matrices

    Parameters
    ----------
    scaled_covar : scipy.sparse.csr_array
        Square 2-D array of size N.
    vec : numpy.typing.NDArray[Any]
        1-D vector of size N.
    atol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 1e-6).
    btol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 1e-6).
    conlim : int
        Parameter for scipy.sparse.linalg.lsqr (default: 0).
    iter_lim : int
        Parameter for scipy.sparse.linalg.lsqr (default: 10000).

    Returns
    -------
    bool
        True indicates that `vec` lies within the column space, but False may
        result from a failure to converge.
    """
    (
        _beta_scal,
        istop,
        _itn,
        _r1norm,
        _r2norm,
        _anorm,
        _acond,
        _arnorm,
        _xnorm,
        _var,
    ) = scipy.sparse.linalg.lsqr(
        scaled_covar,
        vec,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
    )
    return istop in (1, 4)


def within_kern_fast(
    covar: scipy.sparse.csr_array,
    vec: numpy.typing.NDArray[Any],
    atol: float = 1e-6,
    btol: float = 1e-6,
    rtol: float = 1e-5,
    conlim: int = 0,
    iter_lim: int | None = None,
) -> bool:
    """Check whether `vec` lies within the column space of `scaled_covar`.

    Parameters
    ----------
    scaled_covar : scipy.sparse.csr_array
        Square, symmetric 2-D sparse CSR array of size N.
    vec : numpy.typing.NDArray[Any]
        1-D vector of size N.
    atol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 1e-6).
    btol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 1e-6).
    rtol : int | float
        Parameter for scipy.sparse.linalg.minres (default: 1e-5).
    conlim : int
        Parameter for scipy.sparse.linalg.lsqr (default: 0).
    iter_lim : int
        Parameter for scipy.sparse.linalg.minres and scipy.sparse.linalg.lsqr
        (default: 10000).

    Returns
    -------
    bool
        True indicates that `vec` lies within the column space, but False may
        result from a failure to converge.
    """
    soln_approx, ret_code = scipy.sparse.linalg.minres(
        covar,
        vec,
        rtol=rtol,
        maxiter=iter_lim,
    )
    if ret_code != 0:
        logging.error(
            "MINRES failed to converge within "
            f"{5 * covar.shape[0] if iter_lim is None else iter_lim} "
            "iterations!"
        )
        return False
    (
        _beta_scal,
        istop,
        _itn,
        _r1norm,
        _r2norm,
        _anorm,
        _acond,
        _arnorm,
        _xnorm,
        _var,
    ) = scipy.sparse.linalg.lsqr(
        covar,
        vec,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
        x0=soln_approx,
    )

    return istop in (1, 4)


def kern_assoc(
    scaled_covar: scipy.sparse.csr_array,
    vec: numpy.typing.NDArray[Any],
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> tuple[int, ...] | None:
    (
        _beta_scal,
        istop,
        _itn,
        _r1norm,
        _r2norm,
        _anorm,
        _acond,
        _arnorm,
        _xnorm,
        _var,
    ) = scipy.sparse.linalg.lsqr(
        scaled_covar,
        vec,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
    )
    if istop in (1, 4):
        return None
    res_abs = numpy.abs(vec - scaled_covar @ _beta_scal)
    Anorm = scipy.sparse.linalg.norm(scaled_covar)
    xnorm = scipy.linalg.norm(_beta_scal)
    vec_abs = numpy.abs(vec)

    left_compare = res_abs
    right_compare_1 = atol * Anorm * xnorm
    right_compare_2 = btol * vec_abs
    compare_vec = left_compare > right_compare_1 + right_compare_2

    associated_integers = tuple(int(x) for x in numpy.nonzero(compare_vec)[0])

    return associated_integers


def linsolve_lsqr(
    scaled_covar: scipy.sparse.csr_array,
    col_norms: numpy.typing.NDArray[Any],
    target: numpy.typing.NDArray[Any],
    atol: int | float = 0,
    btol: int | float = 0,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> tuple[numpy.typing.NDArray[Any], float]:
    (
        beta_scal,
        _istop,
        _itn,
        r1norm,
        _r2norm,
        _anorm,
        _acond,
        _arnorm,
        _xnorm,
        _var,
    ) = scipy.sparse.linalg.lsqr(
        scaled_covar,
        target,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
    )
    beta = beta_scal / col_norms
    return beta, float(r1norm)


def linreg(
    A: scipy.sparse.csr_array,
    y: numpy.typing.NDArray[Any],
    sqweights: numpy.typing.NDArray[Any] | None,
    get_covar: bool = True,
    atol: int | float = 0,
    btol: int | float = 0,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> tuple[numpy.typing.NDArray[Any], scipy.sparse.csr_array | None]:
    """Solve sparse linear system Ax=y for x.

    Parameters
    ----------
    A : scipy.sparse.csr_array
        2-D sparse array with dimension NxM.
    y : numpy.typing.NDArray[Any]
        1-D vector of size N.
    sqweights : numpy.typing.NDArray[Any]
        Square roots of row weights (size N); optional.  For rows with a
        sqweight less than sqrt(F64_EPS) * M, where M is the maximum sqweight,
        those rows are dropped from A for both the linear regression and the
        Gram matrix returned by `get_covar=True`.
    get_covar : bool
        Whether to return the weighted Gram matrix of A (default: True).
    atol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 0).
    btol : int | float
        Parameter for scipy.sparse.linalg.lsqr (default: 0).
    conlim : int
        Parameter for scipy.sparse.linalg.lsqr (default: 0).
    iter_lim : int
        Parameter for scipy.sparse.linalg.lsqr (default: 10000).

    Returns
    -------
    tuple[numpy.typing.NDArray[Any], scipy.sparse.csr_array | None]
        The first entry is the regressed coefficient vector x of dimension M.
        The second entry is the (optional) weighted Gram matrix of A.
    """
    # check dimensions
    if y.ndim > 1:
        raise ValueError(f"`y` must have dimension one (was {y.ndim})")
    if y.shape[0] != A.shape[0]:
        raise ValueError(
            f"`y` must have same number of entries as `A` rows ({y.shape[0]} "
            f"vs {A.shape[0]})"
        )
    if sqweights is not None and sqweights.ndim > 1:
        raise ValueError(
            f"`sqweights` must have dimension one (was {sqweights.ndim})"
        )

    # apply weights
    A_weighted = (
        A
        if sqweights is None
        else scipy.sparse.csr_array(sqweights[:, numpy.newaxis]) * A
    )
    y_weighted = y if sqweights is None else sqweights * y

    # remove rows which are too small
    if sqweights is not None:
        max_weight = sqweights.max()
        weight_rat = numpy.sqrt(_F64_EPS)
        filter_rows = sqweights > (max_weight * weight_rat)
        A = A[filter_rows, :] if get_covar else A
        A_weighted = A_weighted[filter_rows, :]
        y_weighted = y_weighted[filter_rows]

    # obtain scaled matrices and Gram matrix (both weighted and unweighted)
    A_scaled_w, col_norms = scale_csr(A_weighted)
    A_gram_unw = (A.T @ A).tocsr() if get_covar else None

    # determine coefficients
    coefs, _ = linsolve_lsqr(
        scaled_covar=A_scaled_w,
        col_norms=col_norms,
        target=y_weighted,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
    )

    return coefs, A_gram_unw


def _accum_groups(
    assoc_vals: list[None | tuple[int, ...]], tot: int
) -> tuple[list[int], list[list[int]]]:
    norm_map: list[int] = [0 for _ in range(tot)]
    group_set: list[list[int]] = []
    edge_list: set[tuple[int, int]] = set()
    for i, assoc_list in enumerate(assoc_vals):
        if assoc_list is None:
            continue
        for j in assoc_list:
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            edge_list.add(edge)
    graph = igraph.Graph(n=tot, edges=sorted(edge_list))
    cur_part = 0
    for component in graph.components():
        if len(component) == 0 or len(component) == 1:
            continue
        cur_part += 1
        group_vals: list[int] = sorted(component)
        for i in group_vals:
            norm_map[i] = cur_part
        group_set.append(group_vals)

    return norm_map, group_set


def detect_cvg(
    scaled_covar: scipy.sparse.csr_array,
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> tuple[
    numpy.typing.NDArray[numpy.uintc],
    tuple[
        tuple[numpy.typing.NDArray[numpy.uintc], scipy.sparse.csr_array], ...
    ],
]:
    n_vals = scaled_covar.shape[0]
    assoc_vals: list[None | tuple[int, ...]] = []
    for i in range(n_vals):
        test_vec = numpy.zeros(n_vals, dtype=numpy.uintp)
        test_vec[i] = 1
        assoc_id_i = kern_assoc(
            scaled_covar,
            test_vec,
            atol=atol,
            btol=btol,
            conlim=conlim,
            iter_lim=iter_lim,
        )
        assoc_vals.append(assoc_id_i)
    map_vec, accum_groups = _accum_groups(assoc_vals, n_vals)
    map_vec_final = numpy.asarray(map_vec, dtype=numpy.uintc)
    covar_groups: list[
        tuple[numpy.typing.NDArray[numpy.uintc], scipy.sparse.csr_array]
    ] = []
    for int_group in accum_groups:
        slice_vec = numpy.asarray(int_group, dtype=numpy.uintc)
        slice_covar = scaled_covar[slice_vec].T[slice_vec]
        covar_groups.append((slice_vec, slice_covar))
    return map_vec_final, tuple(covar_groups)


def detect_noncovar(
    scaled_covar: scipy.sparse.csr_array,
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
) -> tuple[numpy.typing.NDArray[numpy.uintc], scipy.sparse.csr_array]:
    n_vals = scaled_covar.shape[0]
    id_mapping: list[int | None] = list(range(n_vals))
    for i in range(n_vals):
        test_vec = numpy.zeros(n_vals, dtype=numpy.uintp)
        test_vec[i] = 1
        map_selector = [i for i in id_mapping if i is not None]
        mapped_i = id_mapping[i]
        assert mapped_i is not None
        mapped_vec = test_vec[map_selector]
        independent_i = within_kern(
            scaled_covar,
            mapped_vec,
            atol=atol,
            btol=btol,
            conlim=conlim,
            iter_lim=iter_lim,
        )
        if independent_i:
            id_mapping[i] = None
            selector = mapped_vec == 0
            scaled_covar = scaled_covar[selector].T[selector]
    map_selector_arr = numpy.asarray(
        [i for i in id_mapping if i is not None], numpy.uintc
    )
    return map_selector_arr, scaled_covar
