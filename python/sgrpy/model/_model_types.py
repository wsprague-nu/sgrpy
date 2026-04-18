"""Basic model type definitions."""

import dataclasses
from collections.abc import Iterable, Sequence

import polars
import scipy

from sgrpy.graph import GraphMatch, IndexMapping, UGraph

PTYPE_DATA = polars.String()
PTYPE_GROUP = polars.Int64()
PTYPE_WEIGHT = polars.Float64()
PTYPE_FILTER = polars.Enum(["NONE", "PARSE", "COLOR", "PARTITION", "FILTER"])
PTYPE_EXTERN_FEAT = polars.List(
    polars.Struct(
        (
            polars.Field("name", polars.String()),
            polars.Field("count", polars.Int64()),
        )
    )
)
PTYPE_INTERN_FEAT = polars.List(
    polars.Struct(
        (
            polars.Field("name", polars.String()),
            polars.Field("count", polars.Int64()),
        )
    )
)
PTYPE_EXTERN_FEAT_STRUCT = polars.Struct(
    (polars.Field("extern_feat", PTYPE_EXTERN_FEAT),)
)
PTYPE_INTERN_FEAT_STRUCT = polars.Struct(
    (polars.Field("intern_feat", PTYPE_INTERN_FEAT),)
)
PTYPE_EXTERN_FEAT_STR = polars.String()
PTYPE_INTERN_FEAT_STR = polars.String()
PTYPE_GROUP_ITEM = polars.UInt64()
PTYPE_WEIGHT_UG = polars.Float64()
PTYPE_STR_UG = polars.String()
PTYPE_GROUP_NORM = polars.UInt64()
PTYPE_STR_CG = polars.String()
PTYPE_EXTRAS_FEAT = polars.List(
    polars.Struct(
        (
            polars.Field("idx", polars.UInt64()),
            polars.Field("count", polars.Int64()),
        )
    )
)
PTYPE_EXTRAS_FEAT_STR = polars.String()
PTYPE_LWEIGHT_PART = polars.Float64()
# PTYPE_EXTRAS = polars.List(
#     polars.Struct(
#         (
#             polars.Field("idx_feat", polars.UInt64()),
#             polars.Field("num_feat", polars.Int64()),
#         )
#     )
# )
# PTYPE_EXTRAS_STR = polars.String()
PTYPE_CSRVEC = polars.Struct(
    (
        polars.Field("data", polars.List(polars.Int64())),
        polars.Field("indices", polars.List(polars.UInt64())),
        polars.Field("indptr", polars.List(polars.UInt64())),
    )
)
PTYPE_CSRVEC_STR = polars.String()
PTYPE_LWEIGHT_NORM = polars.Float64()
PTYPE_FINAL_WEIGHT = polars.Float64()
PTYPE_PRED_COL = polars.Float64()


def convert_extern_feat_to_str(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy().with_columns(
        polars.struct(extern_feat=polars.col("extern_feat"))
        .struct.json_encode()
        .cast(PTYPE_EXTERN_FEAT_STR)
        .alias("extern_feat")
    )
    return lf


def convert_extern_feat_from_str(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy().with_columns(
        polars.col("extern_feat")
        .str.json_decode(PTYPE_EXTERN_FEAT_STRUCT)
        .struct.field("extern_feat")
        .cast(PTYPE_EXTERN_FEAT)
        .alias("extern_feat")
    )
    return lf


def convert_intern_feat_to_str(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy().with_columns(
        polars.struct(intern_feat=polars.col("intern_feat"))
        .struct.json_encode()
        .cast(PTYPE_INTERN_FEAT_STR)
        .alias("intern_feat")
    )
    return lf


# def convert_extras_to_str(
#     df: polars.DataFrame | polars.LazyFrame,
# ) -> polars.LazyFrame:
#     lf = df.lazy().with_columns(
#         polars.struct(extras=polars.col("extras"))
#         .struct.json_encode()
#         .cast(PTYPE_EXTRAS_STR)
#         .alias("extras")
#     )
#     return lf


def convert_extras_feat_to_str(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy().with_columns(
        polars.struct(extras_feat=polars.col("extras_feat"))
        .struct.json_encode()
        .cast(PTYPE_EXTRAS_FEAT_STR)
        .alias("extras_feat")
    )
    return lf


def convert_csrvec_to_str(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy().with_columns(
        polars.col("csrvec")
        .struct.json_encode()
        .cast(PTYPE_CSRVEC_STR)
        .alias("csrvec")
    )
    return lf


def initialize_lf(
    df: polars.DataFrame | polars.LazyFrame,
    data_column: str,
    group_column: str | None,
    weight_column: str | None,
    extern_column: str | None,
    target_columns: Sequence[str] | None,
    extern_from_str: bool,
) -> polars.LazyFrame:
    data_select = (
        polars.when(
            polars.col(data_column)
            .cast(PTYPE_DATA)
            .ne(polars.lit("", polars.String()))
        )
        .then(polars.col(data_column).cast(PTYPE_DATA))
        .alias("data")
    )

    # fill in group column
    if group_column is None:
        group_select = polars.int_range(polars.len(), dtype=PTYPE_GROUP).alias(
            "group"
        )
    else:
        group_select = polars.col(group_column).cast(PTYPE_GROUP).alias("group")

    # fill in weight column
    if weight_column is None:
        weight_select = polars.lit(1.0, PTYPE_WEIGHT).alias("weight")
    else:
        weight_select = (
            polars.col(weight_column)
            .cast(PTYPE_WEIGHT)
            .fill_null(polars.lit(1.0, PTYPE_WEIGHT))
            .alias("weight")
        )

    # fill in extras column
    if extern_column is None:
        extern_select = polars.lit(None, dtype=PTYPE_EXTERN_FEAT).alias(
            "extern_feat"
        )
    elif extern_from_str:
        extern_select = (
            polars.col(extern_column)
            .cast(PTYPE_EXTERN_FEAT_STR)
            .str.json_decode(PTYPE_EXTERN_FEAT_STRUCT)
            .struct.field("extern_feat")
            .cast(PTYPE_EXTERN_FEAT)
            .alias("extern_feat")
        )
    else:
        extern_select = (
            polars.col(extern_column)
            .cast(PTYPE_EXTERN_FEAT)
            .alias("extern_feat")
        )
    filter_select = polars.lit("NONE", dtype=PTYPE_FILTER).alias("filter")

    if target_columns is None:
        lf = df.lazy().select(
            data_select,
            group_select,
            weight_select,
            extern_select,
            filter_select,
        )
    else:
        target_select = (
            polars.col(target_columns)
            .cast(PTYPE_PRED_COL)
            .name.prefix("value_")
        )
        lf = df.lazy().select(
            data_select,
            group_select,
            weight_select,
            extern_select,
            target_select,
            filter_select,
        )

    return lf


def downconvert_lf(
    df: polars.DataFrame | polars.LazyFrame,
) -> polars.LazyFrame:
    lf = df.lazy()
    lf = convert_extern_feat_to_str(lf)
    lf = convert_intern_feat_to_str(lf)
    lf = convert_extras_feat_to_str(lf)
    # lf = convert_extras_to_str(lf)
    lf = convert_csrvec_to_str(lf)
    return lf


@dataclasses.dataclass(frozen=True, slots=True)
class UGraphStruct:
    weights: tuple[float, ...]
    graphs: tuple[UGraph | None, ...]
    group_norm: tuple[int, ...]
    extras: tuple[tuple[str, ...], ...]

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[UGraph, float, int, Sequence[str] | None]]
    ) -> "UGraphStruct":
        weights: list[float] = []
        graphs: list[UGraph] = []
        group_norms: list[int] = []
        features: list[tuple[str, ...]] = []
        for graph, weight, grpn, extras in pairs:
            weights.append(weight)
            graphs.append(graph)
            group_norms.append(grpn)
            if extras is None:
                features.append(())
            else:
                features.append(tuple(extras))
        weight_sum = sum(weights)
        weight_final = (w / weight_sum for w in weights)
        return UGraphStruct(
            weights=tuple(weight_final),
            graphs=tuple(graphs),
            group_norm=tuple(group_norms),
            extras=tuple(features),
        )

    def as_dicts(
        self,
    ) -> Iterable[
        dict[str, float | str | tuple[dict[str, int | str], ...] | None]
    ]:
        return (
            {
                "weight_ug": weight,
                "str_ug": ug.to_sugraph().as_str() if ug is not None else None,
                "group_norm": grpn,
                "intern_feat": tuple({"name": ex, "count": 1} for ex in extras),
            }
            for weight, ug, grpn, extras in zip(
                self.weights,
                self.graphs,
                self.group_norm,
                self.extras,
                strict=True,
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationStruct:
    matches: tuple[int, ...]
    maps: tuple[IndexMapping, ...]

    @classmethod
    def from_graphmatch(
        cls, matches: Iterable[GraphMatch]
    ) -> "ObservationStruct":
        match_gen = ((m.fragment_id, m.mapping) for m in matches)
        return ObservationStruct.from_pairs(match_gen)

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[int, IndexMapping]]
    ) -> "ObservationStruct":
        mappings: list[IndexMapping] = []
        graphs: list[int] = []
        for graph, mapping in pairs:
            mappings.append(mapping)
            graphs.append(graph)
        return ObservationStruct(matches=tuple(graphs), maps=tuple(mappings))


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationMultiStruct:
    lweights: tuple[float, ...]
    data: tuple[ObservationStruct, ...]

    @classmethod
    def from_pairs(
        cls, pairs: Iterable[tuple[ObservationStruct, float]]
    ) -> "ObservationMultiStruct":
        lweights: list[float] = []
        obs_list: list[ObservationStruct] = []
        for obs, lweight in pairs:
            obs_list.append(obs)
            lweights.append(lweight)
        return ObservationMultiStruct(
            lweights=tuple(lweights), data=tuple(obs_list)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationVector:
    lweights: tuple[float, ...]
    vectors: tuple[scipy.sparse.csr_array]
