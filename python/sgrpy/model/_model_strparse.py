"""String-parsing component of model."""

import abc
import collections
import dataclasses
import itertools
from collections.abc import Iterable
from typing import Self

import polars

import sgrpy.shims.rdkit
from sgrpy.graph import SUGraph, UGraph, UGraphEquiv
from sgrpy.iotypes import JSON, json_to_dict

from ._model_types import (
    PTYPE_FILTER,
    PTYPE_GROUP_NORM,
    PTYPE_INTERN_FEAT,
    PTYPE_STR_UG,
    PTYPE_WEIGHT_UG,
    UGraphStruct,
)

PTYPE_STRPARSE_STRUCT = polars.Struct(
    (
        polars.Field("weight_ug", PTYPE_WEIGHT_UG),
        polars.Field("str_ug", PTYPE_STR_UG),
        polars.Field("group_norm", PTYPE_GROUP_NORM),
        polars.Field("intern_feat", PTYPE_INTERN_FEAT),
    )
)


class ModelStrParse(abc.ABC):
    """Component which parses a string into a UGraph."""

    @classmethod
    def from_json(cls, json: JSON) -> Self:
        """Initialize object from JSON data."""
        try:
            return cls._from_json(json)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {cls} from JSON {json}"
            ) from e

    @classmethod
    @abc.abstractmethod
    def _from_json(cls, json: JSON) -> Self: ...

    @abc.abstractmethod
    def to_json(self) -> JSON:
        """Convert object to JSON data."""

    @abc.abstractmethod
    def parse_str(self, data: str) -> UGraphStruct:
        """Parse string into UGraph."""

    @abc.abstractmethod
    def debug(self, data: str) -> Iterable[str]:
        """Debug issues with parsing."""

    def parse_str_dicts(
        self, data: str
    ) -> tuple[
        dict[str, str | float | int | tuple[dict[str, int | str], ...] | None],
        ...,
    ]:
        """Parse string into structs."""
        ugs = self.parse_str(data)
        dict_tup = tuple(ugs.as_dicts())
        return dict_tup

    def parse_polars(self, lf: polars.LazyFrame) -> polars.LazyFrame:
        """Parse polars LazyFrame (must have str column "data")."""
        new_lf = lf.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                polars.all_horizontal(
                    polars.col("group", "weight", "data").is_not_null()
                ),
            )
            .then(
                polars.col("data").map_elements(
                    self.parse_str_dicts,
                    return_dtype=polars.List(PTYPE_STRPARSE_STRUCT),
                )
            )
            .alias("packed_ugdat")
        )

        # explode and unpack lazyframe
        exploded_lf = (
            new_lf.explode("packed_ugdat")
            .unnest("packed_ugdat")
            .with_columns(
                polars.when(polars.col("intern_feat").list.len() > 0)
                .then("intern_feat")
                .alias("intern_feat"),
                polars.when(
                    polars.col("str_ug").ne(polars.lit("", polars.String()))
                )
                .then(polars.col("str_ug"))
                .alias("str_ug"),
            )
        )
        filtered_lf = exploded_lf.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                (
                    polars.col("data").ne(polars.lit("", polars.String()))
                    & polars.col("data").is_not_null()
                    & polars.col("str_ug").is_null()
                    & polars.col("intern_feat").is_null()
                ),
                (polars.col("extern_feat").list.len() > 0).not_(),
            )
            .then(polars.lit("PARSE"))
            .otherwise(polars.col("filter"))
            .cast(PTYPE_FILTER)
            .alias("filter")
        )
        return filtered_lf


@dataclasses.dataclass(frozen=True, slots=True)
class ModelStrParseRaw(ModelStrParse):
    @classmethod
    def _from_json(cls, json: JSON) -> "ModelStrParseRaw":
        assert json is None
        return ModelStrParseRaw()

    def to_json(self) -> JSON:
        return None

    def debug(self, data: str) -> Iterable[str]:
        return ("Unknown parsing error",)

    def parse_str(self, data: str) -> UGraphStruct:
        ug = SUGraph.from_str(data).to_ugraph()
        pair = (ug, 1.0, 0, ())
        return UGraphStruct.from_pairs((pair,))


@dataclasses.dataclass(frozen=True, slots=True)
class ModelStrParseRDKit(ModelStrParse):
    stereo_pi: bool
    ext_ring: bool
    ext_conj: bool
    feat_ring: int  # 0 for none, 1 for core, 2 for extended
    feat_conj: int  # 0 for none, 1 for core, 2 for extended
    trim: bool

    @classmethod
    def _from_json(cls, json: JSON) -> "ModelStrParseRDKit":
        json_dict = json_to_dict(json)
        stereo_pi = bool(json_dict["stereo_pi"])
        extra_ring = bool(json_dict["ext_ring"])
        extra_conj = bool(json_dict["ext_conj"])
        feat_ring = int(json_dict["feat_ring"])  # type: ignore[arg-type]
        feat_conj = int(json_dict["feat_conj"])  # type: ignore[arg-type]
        trim = bool(json_dict["trim"])
        return ModelStrParseRDKit(
            stereo_pi=stereo_pi,
            ext_ring=extra_ring,
            ext_conj=extra_conj,
            feat_ring=feat_ring,
            feat_conj=feat_conj,
            trim=trim,
        )

    def to_json(self) -> JSON:
        return {
            "stereo_pi": self.stereo_pi,
            "ext_ring": self.ext_ring,
            "ext_conj": self.ext_conj,
            "feat_ring": self.feat_ring,
            "feat_conj": self.feat_conj,
            "trim": self.trim,
        }

    def debug(self, data: str | None) -> Iterable[str]:
        if data is None:
            return ("`None` passed for data string",)
        try:
            result = self.parse_str(data, debug=True)
        except (
            NotImplementedError,
            sgrpy.shims.rdkit.RDKitTranslationError,
        ) as err:
            return (str(err),)
        if len(result.graphs) == 0:
            return ("Parsing error (extra detail not available at this time)",)
        return ()

    def parse_str(self, data: str, debug: bool = False) -> UGraphStruct:
        if data == "":
            ugraph = UGraph.from_tuples((), ())
            return UGraphStruct.from_pairs(((ugraph, 1.0, 0, None),))

        mol = sgrpy.shims.rdkit.load_smiles(data)
        if mol is None:
            return UGraphStruct.from_pairs(())

        result_graphs = sgrpy.shims.rdkit.map_benson_v4_compat_v3(
            mol,
            node_cis=self.stereo_pi,
            trim_ends=self.trim,
            ext_ring=self.ext_ring,
            ext_conj=self.ext_conj,
            feat_ring=self.feat_ring,
            feat_conj=self.feat_conj,
        )

        try:
            # not actually unique any more...but sorting is good for replication
            ugraphs = [
                (
                    UGraphEquiv.from_ugraph(ug.to_ugraph())
                    .to_ugraph()
                    .to_sugraph(),
                    tuple(
                        sorted(
                            ()
                            if ug.extras is None
                            else itertools.chain.from_iterable(
                                (f"{key}({value})" for value in values)
                                for key, values in ug.extras.items()
                            )
                        )
                    ),
                )
                for ug in result_graphs
            ]
        except (
            NotImplementedError,
            sgrpy.shims.rdkit.RDKitTranslationError,
            sgrpy.shims.rdkit.benson_v4.ParsingError,
        ) as err:
            if debug:
                raise err
            return UGraphStruct.from_pairs(())

        ugraph_weights = collections.Counter(ugraphs)

        weight = 1.0 / sum(ugraph_weights.values())

        ugs = UGraphStruct.from_pairs(
            (
                (ug[0].to_ugraph(), count * weight, i, ug[1])
                for i, (ug, count) in enumerate(ugraph_weights.items())
            )
        )

        return ugs


def model_str_from_json(json: JSON) -> ModelStrParse:
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "ModelStrParseRaw":
            return ModelStrParseRaw.from_json(model_data)
        case "ModelStrParseRDKit":
            return ModelStrParseRDKit.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized string parsing model {model_type}")


def model_str_to_json(model_str: ModelStrParse) -> JSON:
    return {"type": type(model_str).__name__, "data": model_str.to_json()}
