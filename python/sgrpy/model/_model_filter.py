"""Filter structure."""

import abc
import dataclasses
from collections.abc import Iterable
from typing import Any

import numpy
import polars
import scipy

from sgrpy.iotypes import JSON, JSONable, json_to_dict
from sgrpy.linear import (
    csr_equal,
    csr_to_json,
    dict_to_csrvecint,
    json_to_csr,
    scale_gram_sparse,
    within_kern,
    within_kern_fast,
)

from ._model_types import PTYPE_FILTER


class ModelFilter(JSONable):
    """Component which filters a feature vector."""

    @abc.abstractmethod
    def filter(self, vec: numpy.typing.NDArray[Any]) -> bool:
        """Filter a feature vector."""

    @abc.abstractmethod
    def debug(self, vec: numpy.typing.NDArray[Any]) -> Iterable[str]:
        """Return debug message(s) for feature vector."""

    @abc.abstractmethod
    def filter_struct(self, struct: dict[str, list[int]]) -> bool:
        """Filter a feature vector based on CSR struct."""

    def filter_df(self, lf: polars.LazyFrame) -> polars.LazyFrame:
        """Filter a column of feature vectors."""
        lf_filt = lf.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                polars.col("csrvec").is_not_null(),
                polars.col("csrvec")
                .map_elements(
                    self.filter_struct,
                    return_dtype=polars.Boolean(),
                    strategy="threading",
                )
                .not_(),
            )
            .then(polars.lit("FILTER"))
            .otherwise(polars.col("filter"))
            .cast(PTYPE_FILTER)
            .alias("filter"),
        )
        return lf_filt


@dataclasses.dataclass(frozen=True, slots=True)
class CovarFilter(ModelFilter):
    scaled_covar: scipy.sparse.csr_array
    atol: int | float = 1e-6
    btol: int | float = 1e-6
    conlim: int = 0
    iter_lim: int = 10000

    @classmethod
    def _from_json(cls, json: JSON) -> "CovarFilter":
        json_data = json_to_dict(json)
        csr_matrix = json_to_csr(json_data["scaled_covar"])
        atol: int | float = json_data["atol"]  # type: ignore
        btol: int | float = json_data["btol"]  # type: ignore
        conlim: int = json_data["conlim"]  # type: ignore
        iter_lim: int = json_data["iter_lim"]  # type: ignore

        return CovarFilter(
            scaled_covar=csr_matrix,
            atol=atol,
            btol=btol,
            conlim=conlim,
            iter_lim=iter_lim,
        )

    def to_json(self) -> JSON:
        json_val = {
            "scaled_covar": csr_to_json(self.scaled_covar),
            "atol": self.atol,
            "btol": self.btol,
            "conlim": self.conlim,
            "iter_lim": self.iter_lim,
        }
        return json_val

    def filter(
        self,
        vec: numpy.typing.NDArray[Any],
    ) -> bool:
        raise DeprecationWarning("CovarFilter method is not reliable")
        return within_kern(
            scaled_covar=self.scaled_covar,
            vec=vec,
            atol=self.atol,
            btol=self.btol,
            conlim=self.conlim,
            iter_lim=self.iter_lim,
        )

    def debug(self, vec: numpy.typing.NDArray[Any]) -> Iterable[str]:
        can_filter = self.filter(vec)
        if can_filter:
            return (
                "Issues with covariance (more detailed reporting not yet "
                "available.)",
            )
        return ()

    def filter_struct(self, struct: dict[str, list[int]]) -> bool:
        """Filter a feature vector based on CSR struct."""
        size = self.scaled_covar.shape[1]
        vec_sp = dict_to_csrvecint(struct, size)
        vec_np = vec_sp.todense()[0, :]
        return self.filter(vec_np)


@dataclasses.dataclass(frozen=True, slots=True)
class CovarFilterFast(ModelFilter):
    _covar: scipy.sparse.csr_array
    _col_scale: numpy.typing.NDArray[numpy.float64]
    atol: float
    btol: float
    conlim: int
    rtol: float
    iter_lim: int

    @classmethod
    def from_train_mat(
        cls,
        train_mat: scipy.sparse.sparray,
        atol: float = 1e-6,
        btol: float = 1e-6,
        conlim: int = 0,
        rtol: float = 1e-12,
        iter_lim: int = 10000,
    ) -> "CovarFilterFast":
        train_csr = train_mat.tocsr()
        train_csr.eliminate_zeros()
        train_csr.sum_duplicates()
        train_scaled_arb, col_norm_recip = scale_gram_sparse(train_csr)
        train_scaled = train_scaled_arb.tocsr()

        if not csr_equal(train_scaled, train_scaled.T.tocsr()):
            raise ValueError("`covar` must be symmetric")

        return CovarFilterFast(
            _covar=train_scaled,
            _col_scale=col_norm_recip,
            atol=atol,
            btol=btol,
            conlim=conlim,
            rtol=rtol,
            iter_lim=iter_lim,
        )

    @classmethod
    def _from_json(cls, json: JSON) -> "CovarFilterFast":
        json_data = json_to_dict(json)
        csr_matrix = json_to_csr(json_data["covar"])
        col_norms = numpy.asarray(json_data["col_norms"], dtype=numpy.float64)
        atol: int | float = json_data["atol"]  # type: ignore
        btol: int | float = json_data["btol"]  # type: ignore
        conlim: int = json_data["conlim"]  # type: ignore
        rtol: float = json_data["rtol"]  # type: ignore
        iter_lim: int = (
            int(json_data["iter_lim"])  # type: ignore
        )

        return CovarFilterFast(
            _covar=csr_matrix,
            _col_scale=col_norms,
            atol=atol,
            btol=btol,
            conlim=conlim,
            rtol=rtol,
            iter_lim=iter_lim,
        )

    def to_json(self) -> JSON:
        json_val = {
            "covar": csr_to_json(self._covar),
            "col_norms": self._col_scale.tolist(),
            "atol": self.atol,
            "btol": self.btol,
            "conlim": self.conlim,
            "rtol": self.rtol,
            "iter_lim": self.iter_lim,
        }
        return json_val

    def filter(
        self,
        vec: numpy.typing.NDArray[Any],
    ) -> bool:
        vec_scaled = vec * self._col_scale
        return within_kern_fast(
            covar=self._covar,
            vec=vec_scaled,
            atol=self.atol,
            btol=self.btol,
            conlim=self.conlim,
            rtol=self.rtol,
            iter_lim=self.iter_lim,
        )

    def debug(self, vec: numpy.typing.NDArray[Any]) -> Iterable[str]:
        can_filter = self.filter(vec)
        if not can_filter:
            return (
                "Issues with covariance (more detailed reporting not yet "
                "available.)",
            )
        return ()

    def filter_struct(self, struct: dict[str, list[int]]) -> bool:
        """Filter a feature vector based on CSR struct."""
        size = self._covar.shape[1]
        vec_sp = dict_to_csrvecint(struct, size)
        vec_np = vec_sp.todense()[0, :]
        return self.filter(vec_np)


# @dataclasses.dataclass(frozen=True, slots=True)
# class SmartCovarFilter(ModelFilter):
#     mapping_vector: numpy.typing.NDArray[numpy.uintc]
#     covar_groups: tuple[
#         tuple[numpy.typing.NDArray[numpy.uintc], scipy.sparse.csr_array], ...
#     ]
#     atol: int | float
#     btol: int | float
#     conlim: int
#     iter_lim: int

#     @classmethod
#     def from_cvm(
#         cls,
#         scaled_covar: scipy.sparse.csr_array,
#         atol: int | float = 1e-6,
#         btol: int | float = 1e-6,
#         conlim: int = 0,
#         iter_lim: int = 10000,
#     ) -> "SmartCovarFilter":
#         mapvec, covar_groups = detect_cvg(
#             scaled_covar=scaled_covar,
#         )
#         return SmartCovarFilter(
#             mapping_vector=mapvec,
#             covar_groups=covar_groups,
#             atol=atol,
#             btol=btol,
#             conlim=conlim,
#             iter_lim=iter_lim,
#         )

#     @classmethod
#     def _from_json(cls, json: JSON) -> "SmartCovarFilter":
#         json_data = json_to_dict(json)
#         mapping_vector = numpy.asarray(
#             json_data["mapping_vector"], dtype=numpy.uintc
#         )
#         covar_groups = tuple(
#             (numpy.asarray(cg["mv"], dtype=numpy.uintc),
#              json_to_csr(cg["cvm"]))
#             for cg in json_data["covar_groups"]  # type: ignore
#         )
#         atol: int | float = json_data["atol"]  # type: ignore
#         btol: int | float = json_data["btol"]  # type: ignore
#         conlim: int = json_data["conlim"]  # type: ignore
#         iter_lim: int = json_data["iter_lim"]  # type: ignore

#         return SmartCovarFilter(
#             mapping_vector=mapping_vector,
#             covar_groups=covar_groups,
#             atol=atol,
#             btol=btol,
#             conlim=conlim,
#             iter_lim=iter_lim,
#         )

#     def to_json(self) -> JSON:
#         covar_groups = [
#             {"mv": mv.tolist(), "cvm": csr_to_json(cvm)}
#             for mv, cvm in self.covar_groups
#         ]
#         json_val = {
#             "mapping_vector": self.mapping_vector.tolist(),
#             "covar_groups": covar_groups,
#             "atol": self.atol,
#             "btol": self.btol,
#             "conlim": self.conlim,
#             "iter_lim": self.iter_lim,
#         }
#         return json_val

#     def filter(
#         self,
#         vec: numpy.typing.NDArray[Any],
#     ) -> bool:
#         nonzero_arr = vec.astype(bool).astype(numpy.uintp)
#         relevant_groups: list[int] = numpy.unique(
#             nonzero_arr * self.mapping_vector
#         ).tolist()

#         for cvg in relevant_groups:
#             if cvg == 0:
#                 continue
#             cvg_idxs, cvg_kern = self.covar_groups[cvg - 1]
#             cvg_slice = vec[cvg_idxs]
#             if not within_kern(
#                 scaled_covar=cvg_kern,
#                 vec=cvg_slice,
#                 atol=self.atol,
#                 btol=self.btol,
#                 conlim=self.conlim,
#                 iter_lim=self.iter_lim,
#             ):
#                 return False
#         return True

#     def debug(self, vec: numpy.typing.NDArray[Any]) -> Iterable[str]:
#         can_filter = self.filter(vec)
#         if can_filter:
#             return (
#                 "Issues with covariance (more detailed reporting not yet "
#                 "available.)",
#             )
#         return ()

#     def filter_struct(self, struct: dict[str, list[int]]) -> bool:
#         """Filter a feature vector based on CSR struct."""
#         size = self.mapping_vector.shape[0]
#         vec_sp = dict_to_csrvecint(struct, size)
#         vec_np = vec_sp.todense()[0, :]
#         return self.filter(vec_np)


def model_filter_from_json(json: JSON) -> ModelFilter:
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "CovarFilter":
            return CovarFilter.from_json(model_data)
        case "CovarFilterFast":
            return CovarFilterFast.from_json(model_data)
        # case "SmartCovarFilter":
        #     return SmartCovarFilter.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized prediction model {model_type}")


def model_filter_to_json(model_str: ModelFilter) -> JSON:
    return {"type": type(model_str).__name__, "data": model_str.to_json()}
