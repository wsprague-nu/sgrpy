"""CGraph fragmenter model."""

import abc
import dataclasses
from collections.abc import Iterable

import numpy
import polars
import scipy

from sgrpy.graph import CGraph, SCanonGraph, SCGraph
from sgrpy.iotypes import JSON, JSONable, json_to_dict
from sgrpy.train import FragModel

from ._model_types import (
    PTYPE_CSRVEC,
    PTYPE_FILTER,
    PTYPE_LWEIGHT_PART,
    ObservationMultiStruct,
    ObservationStruct,
)

PTYPE_PLEX_RESULT = polars.List(
    polars.Struct(
        (
            polars.Field("lweight_part", PTYPE_LWEIGHT_PART),
            polars.Field("csrvec", PTYPE_CSRVEC),
        )
    )
)


class ModelPlex(JSONable):
    """Component which multiplexes a CGraph into many Observations."""

    @abc.abstractmethod
    def plex(
        self,
        cgraph: CGraph,
    ) -> ObservationMultiStruct:
        """Partition CGraph into Observations."""

    @abc.abstractmethod
    def get_size(self) -> int:
        """Get number of color indices."""

    @abc.abstractmethod
    def num_features(self) -> int:
        """Get number of feature indices."""

    @abc.abstractmethod
    def get_labels(self) -> tuple[str, ...]:
        """Get feature labels."""

    def plex_str_dicts(
        self,
        cgraph_str: str | None,
        extra_features: Iterable[dict[str, int]] | None,
    ) -> list[tuple[float, dict[str, list[int]]]] | None:
        extra_feat_tup = (
            None
            if extra_features is None
            else ((x["idx"], x["count"]) for x in extra_features)
        )
        return self.plex_str(cgraph_str, extra_feat_tup)

    def plex_str(
        self,
        cgraph_str: str | None,
        extra_features: Iterable[tuple[int, int]] | None,
    ) -> list[tuple[float, dict[str, list[int]]]] | None:
        """Apply plexer to string CGraph."""
        # if no valid input, return nothing
        cg = (
            cgraph_str
            if cgraph_str is None
            else SCGraph.from_str(cgraph_str).to_cgraph()
        )

        result = self.plex_cg(cg, extra_features)
        if result is None:
            return None

        all_results: list[tuple[float, dict[str, list[int]]]] = []
        for lweight, obs_vec in result:
            obs_csr = scipy.sparse.csr_array(obs_vec)
            data = [int(x) for x in obs_csr.data.tolist()]
            indices = [int(x) for x in obs_csr.indices.tolist()]
            indptr = [int(x) for x in obs_csr.indptr.tolist()]
            all_results.append(
                (lweight, {"data": data, "indices": indices, "indptr": indptr})
            )

        return all_results

    def plex_cg(
        self,
        cg: CGraph | None,
        extra_features: Iterable[tuple[int, int]] | None,
    ) -> list[tuple[float, numpy.typing.NDArray[numpy.int64]]] | None:
        """Apply plexer to CGraph."""
        # if no valid input, return nothing
        if cg is None and extra_features is None:
            return []

        frag_size = self.get_size()
        total_size = frag_size + self.num_features()

        # create common base vector from extra_features, if exist
        obs_vec_root = numpy.zeros((1, total_size), dtype=numpy.int64)
        if extra_features is not None:
            for feat_i, feat_count in extra_features:
                obs_vec_root[0, feat_i + frag_size] += feat_count

        # if no partitioning necessary, return single entry with default weight
        if cg is None:
            return [(0.0, obs_vec_root)]

        obs_result = self.plex(cg)
        all_results: list[tuple[float, numpy.typing.NDArray[numpy.int64]]] = []

        for lweight, obs_struc in zip(
            obs_result.lweights, obs_result.data, strict=True
        ):
            obs_vec = obs_vec_root.copy()
            for f_i in obs_struc.matches:
                obs_vec[0, f_i] += 1
            all_results.append((lweight, obs_vec))

        return all_results

    def plex_dict(
        self, cgd: dict[str, str | Iterable[dict[str, int] | None]]
    ) -> list[tuple[float, dict[str, list[int]]]] | None:
        cg: str | None = cgd["str_cg"]  # type: ignore[assignment]
        feat: Iterable[dict[str, int]] | None = cgd["extras_feat"]  # type: ignore[assignment]
        return self.plex_str_dicts(cg, feat)

    def plex_polars(self, lf: polars.LazyFrame) -> polars.LazyFrame:
        """Apply plexer to polars LazyFrame."""
        lf_result = (
            lf.with_columns(
                polars.when(
                    polars.col("filter").eq(polars.lit("NONE")),
                    polars.any_horizontal(
                        polars.col("str_cg", "extras_feat").is_not_null()
                    ),
                )
                .then(
                    polars.struct("str_cg", "extras_feat").map_elements(
                        self.plex_dict,
                        return_dtype=PTYPE_PLEX_RESULT,
                        strategy="threading",
                    )
                )
                .alias("part_result")
            )
            .explode("part_result")
            .unnest("part_result")
        )
        lf_filtered = lf_result.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                polars.col("str_cg").is_not_null(),
                polars.any_horizontal(
                    polars.col(("lweight_part", "csrvec")).is_null()
                ),
            )
            .then(polars.lit("PARTITION"))
            .otherwise(polars.col("filter"))
            .cast(PTYPE_FILTER)
            .alias("filter")
        )
        return lf_filtered


@dataclasses.dataclass(frozen=True, slots=True)
class ModelPartitionTree(ModelPlex):
    fragmodel: FragModel
    limit: int | None
    max_iter: int | None
    max_heap: int | None
    total_feat: int

    @classmethod
    def _from_json(cls, json: JSON) -> "ModelPartitionTree":
        json_dict = json_to_dict(json)
        model = FragModel.from_json(json_dict["fragmodel"])
        limit: int | None = json_dict["limit"]  # type: ignore
        max_iter: int | None = json_dict["max_iter"]  # type: ignore
        max_heap: int | None = json_dict["max_heap"]  # type: ignore
        total_feat: int = json_dict["total_feat"]  # type: ignore
        return ModelPartitionTree(
            fragmodel=model,
            limit=limit,
            max_iter=max_iter,
            max_heap=max_heap,
            total_feat=total_feat,
        )

    def to_json(self) -> JSON:
        model_json = self.fragmodel.to_json()
        return {
            "fragmodel": model_json,
            "limit": self.limit,
            "max_iter": self.max_iter,
            "max_heap": self.max_heap,
            "total_feat": self.total_feat,
        }

    def num_fragments(self) -> int:
        return self.fragmodel.fragtree.nof_pinned()

    def num_features(self) -> int:
        return self.total_feat

    def plex(
        self,
        cgraph: CGraph,
    ) -> ObservationMultiStruct:
        part_iter = self.fragmodel.partition(
            graph=cgraph,
            limit=self.limit,
            max_iter=self.max_iter,
            max_heap=self.max_heap,
            useac=False,
        )

        match_iter = ((obs.weight, obs.matches) for obs in part_iter)

        os_iter = (
            (ObservationStruct.from_graphmatch(matches), lweight)
            for lweight, matches in match_iter
        )
        osm = ObservationMultiStruct.from_pairs(os_iter)
        return osm

    def iter_fragments(self) -> Iterable[SCanonGraph]:
        return self.fragmodel.fragtree.iter_pins()

    def get_size(self) -> int:
        return self.fragmodel.nof_fragments()

    def get_labels(self) -> tuple[str, ...]:
        return tuple(scg.as_str() for scg in self.iter_fragments())


def model_partition_from_json(json: JSON) -> ModelPlex:
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "ModelPartitionTree":
            return ModelPartitionTree.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized partitioner model {model_type}")


def model_partition_to_json(model_str: ModelPlex) -> JSON:
    return {"type": type(model_str).__name__, "data": model_str.to_json()}
