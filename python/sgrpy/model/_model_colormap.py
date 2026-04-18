"""Color mapping model."""

import abc
import collections
import dataclasses
import functools
import itertools
import logging
import operator
from collections.abc import Iterable, Sequence
from typing import Self

import polars

from sgrpy.graph import CGraph, SUGraph, UGraph
from sgrpy.iotypes import JSON, json_to_dict, json_to_list

from ._model_reject import ModelReject
from ._model_types import (
    PTYPE_EXTERN_FEAT,
    PTYPE_EXTRAS_FEAT,
    PTYPE_FILTER,
    PTYPE_STR_CG,
)


class ModelColorMap(abc.ABC):
    """Component which parses a UGraph into a CGraph."""

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
    def num_colors(self) -> int:
        """Get number of colors."""

    @abc.abstractmethod
    def get_colors(self) -> Iterable[str]:
        """Get color labels."""

    @abc.abstractmethod
    def num_features(self) -> int:
        """Get number of non-color features."""

    @abc.abstractmethod
    def get_features(self) -> Iterable[str]:
        """Get non-color features."""

    @abc.abstractmethod
    def parse_ug(self, ugraph: UGraph) -> CGraph | None:
        """Parse UGraph into CGraph."""

    @abc.abstractmethod
    def debug(
        self, ugraph: UGraph | None, features: Iterable[str]
    ) -> Iterable[str]:
        """Debug missing labels."""

    @abc.abstractmethod
    def parse_ug_fail(self, ugraph: UGraph) -> CGraph:
        """Parse UGraph into CGraph."""

    @abc.abstractmethod
    def parse_feat(
        self, features: Sequence[tuple[str, int]]
    ) -> tuple[tuple[int, int], ...] | None:
        """Parse features into feature index tuple."""

    @abc.abstractmethod
    def parse_feat_fail(
        self, features: Sequence[tuple[str, int]]
    ) -> tuple[tuple[int, int], ...]:
        """Parse features into feature index tuple."""

    def parse_ug_str(self, ugraph_str: str) -> str | None:
        """Parse string forms of UGraph and CGraph."""
        ug = SUGraph.from_str(ugraph_str).to_ugraph()
        cg = self.parse_ug(ug)
        return cg if cg is None else cg.to_scgraph().as_str()

    def parse_feat_dict(
        self, features: Iterable[dict[str, str | int]]
    ) -> tuple[dict[str, int], ...] | None:
        feature_map = [(feat["name"], feat["count"]) for feat in features]
        new_features = self.parse_feat(feature_map)  # type: ignore[arg-type]
        if new_features is None:
            return None
        return tuple({"idx": f[0], "count": f[1]} for f in new_features)

    def parse_ug_polars(self, lf: polars.LazyFrame) -> polars.LazyFrame:
        """Parse polars LazyFrame form."""
        lf_mapped_1 = lf.with_columns(
            polars.when(polars.col("filter").eq(polars.lit("NONE")))
            .then(
                polars.col("str_ug").map_elements(
                    functools.partial(convert_cgraph, self),
                    return_dtype=PTYPE_STR_CG,
                )
            )
            .alias("str_cg")
        )
        lf_mapped_2 = lf_mapped_1.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                polars.any_horizontal(
                    polars.col("extern_feat", "intern_feat").is_not_null()
                ),
            )
            .then(
                polars.col("extern_feat")
                .fill_null(polars.lit([], dtype=PTYPE_EXTERN_FEAT))
                .list.concat(
                    polars.col("intern_feat").fill_null(
                        polars.lit([], dtype=PTYPE_EXTERN_FEAT)
                    )
                )
                .map_elements(
                    self.parse_feat_dict,
                    return_dtype=PTYPE_EXTRAS_FEAT,
                )
            )
            .alias("extras_feat")
        )
        lf_mapped_3 = lf_mapped_2.with_columns(
            polars.when(polars.col("extras_feat").list.len() > 0)
            .then(polars.col("extras_feat"))
            .alias("extras_feat")
        )
        lf_filtered = lf_mapped_3.with_columns(
            polars.when(
                polars.col("filter").eq(polars.lit("NONE")),
                (
                    (
                        (polars.col("extern_feat").list.len() > 0)
                        | (polars.col("intern_feat").list.len() > 0)
                    )
                    & polars.col("extras_feat").is_null()
                )
                | (
                    polars.col("str_ug").is_not_null()
                    & polars.col("str_cg").is_null()
                ),
            )
            .then(polars.lit("COLOR"))
            .otherwise(polars.col("filter"))
            .cast(PTYPE_FILTER)
            .alias("filter"),
            polars.col("group_norm")
            .fill_null(polars.lit(0, polars.UInt64()))
            .alias("group_norm"),
            polars.col("weight_ug")
            .fill_null(polars.lit(1.0, polars.Float64()))
            .alias("weight_ug"),
        )
        return lf_filtered


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureString:
    string: str
    is_color: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ModelColorMapSimple(ModelColorMap):
    _colormap: dict[str, int]
    _featuremap: dict[str, int]

    @classmethod
    def from_chunks(
        cls,
        color_chunks: Iterable[tuple[int, int, str]],
        min_count: int,
        feature_chunks: Iterable[tuple[int, int, str]] | None = None,
    ) -> "ModelColorMapSimple":
        # construct initial maps
        group_map: dict[int, dict[int, set[FeatureString]]] = {}
        for group_id, item_id, key in color_chunks:
            featurestring = FeatureString(string=key, is_color=True)
            if group_id not in group_map:
                group_map[group_id] = {}
            if item_id not in group_map[group_id]:
                group_map[group_id][item_id] = {featurestring}
            else:
                group_map[group_id][item_id].add(featurestring)
        if feature_chunks is None:
            feature_chunks = ()
        for group_id, item_id, key in feature_chunks:
            featurestring = FeatureString(string=key, is_color=False)
            if group_id not in group_map:
                group_map[group_id] = {}
            if item_id not in group_map[group_id]:
                group_map[group_id][item_id] = {featurestring}
            else:
                group_map[group_id][item_id].add(featurestring)

        # create reverse label map
        label_map: dict[FeatureString, dict[int, set[int]]] = {}
        for group_id, item_set in group_map.items():
            for item_id, label_set in item_set.items():
                for label in label_set:
                    # if there is no label entry in the label map, add it
                    if label not in label_map:
                        label_map[label] = {group_id: {item_id}}
                        continue
                    # if there is no group entry in the label entry, add it
                    label_entry = label_map[label]
                    if group_id not in label_entry:
                        label_entry[group_id] = {item_id}
                        continue
                    # else append item to group entry
                    label_entry[group_id].add(item_id)

        # find initial bad labels and add their entries to the remove list
        remove_labels: list[tuple[FeatureString, dict[int, set[int]]]] = [
            (label, label_grps)
            for label, label_grps in label_map.items()
            if len(label_grps) < min_count
        ]
        # create initial bad group list and remove from label map
        remove_stack: list[tuple[int, int]] = []
        for label, label_grps in remove_labels:
            remove_stack.extend(
                itertools.chain.from_iterable(
                    ((group_id, item_id) for item_id in items)
                    for group_id, items in label_grps.items()
                )
            )
            del label_map[label]
            logging.debug(f'Removed label "{label.string}"')

        # begin iteration using FILO stack method
        while len(remove_stack) > 0:
            # pop items to remove from stack
            remove_group, remove_item = remove_stack.pop()

            # check each item to see if it is still in the map
            group_entry = group_map.get(remove_group)
            if group_entry is None:
                continue
            if remove_item not in group_entry:
                continue

            # remove item entry from group map
            item_entry = group_entry.pop(remove_item)

            # remove group entry if it no longer exists
            if len(group_entry) == 0:
                del group_map[remove_group]

            # begin removing label maps
            for remove_label in item_entry:
                label_entry_n = label_map.get(remove_label)
                if label_entry_n is None:
                    continue
                label_group_entry = label_entry_n.get(remove_group)
                if (
                    label_group_entry is None
                    or remove_item not in label_group_entry
                ):
                    continue
                label_group_entry.remove(remove_item)

                # if group still has entries for that label, do nothing
                if len(label_group_entry) > 0:
                    continue

                # since group has no more entries, remove it and see if the
                # label itself ought to be removed
                del label_entry_n[remove_group]
                if len(label_entry_n) < min_count:
                    remove_stack.extend(
                        itertools.chain.from_iterable(
                            ((group_id, item_id) for item_id in items)
                            for group_id, items in label_entry_n.items()
                        )
                    )
                    del label_map[remove_label]
                    logging.debug(f'Removed label "{remove_label.string}"')

        # pack all group labels in common
        all_group_labels = [
            frozenset(itertools.chain.from_iterable(d.values()))
            for d in group_map.values()
        ]

        # concatenate all iterators
        color_gen = (
            lab.string
            for lab in itertools.chain.from_iterable(all_group_labels)
            if lab.is_color
        )

        feature_gen = (
            lab.string
            for lab in itertools.chain.from_iterable(all_group_labels)
            if not lab.is_color
        )

        return ModelColorMapSimple.from_labels(color_gen, feature_gen)

    @classmethod
    def from_labels(
        cls, colors: Iterable[str], features: Iterable[str]
    ) -> "ModelColorMapSimple":
        color_counter = collections.Counter(colors)
        labels_sorted = sorted(
            ((-count, key) for key, count in color_counter.items())
        )
        colormap = {key: i for i, (_, key) in enumerate(labels_sorted)}

        feature_counter = collections.Counter(features)
        features_sorted = sorted(
            (-count, key) for key, count in feature_counter.items()
        )
        featuremap = {key: i for i, (_, key) in enumerate(features_sorted)}

        return ModelColorMapSimple(_colormap=colormap, _featuremap=featuremap)

    @classmethod
    def _from_json(cls, json: JSON) -> "ModelColorMapSimple":
        json_dict = json_to_dict(json)
        color_list = json_to_list(json_dict["colormap"])
        colormap = {str(key): index for index, key in enumerate(color_list)}
        feature_list = json_to_list(json_dict["featuremap"])
        featuremap = {str(key): index for index, key in enumerate(feature_list)}
        return ModelColorMapSimple(_colormap=colormap, _featuremap=featuremap)

    def to_json(self) -> JSON:
        colormap = [k for k in self._colormap]
        featuremap = [k for k in self._featuremap]
        return {"colormap": colormap, "featuremap": featuremap}

    def parse_ug_fail(self, ugraph: UGraph) -> CGraph:
        all_bonds = ugraph.get_bonds()
        all_colors = [self._colormap.get(k) for k in ugraph.get_labels()]
        if None in all_colors:
            bad_groups = sorted(
                frozenset(
                    k
                    for k in ugraph.get_labels()
                    if self._colormap.get(k) is None
                )
            )
            if len(bad_groups) == 1:
                raise ModelReject("Missing group: ", bad_groups[0])
            raise ModelReject("Missing groups: ", *bad_groups)
        return CGraph.from_bonds(
            all_bonds, (c for c in all_colors if c is not None)
        )

    def parse_ug(self, ugraph: UGraph) -> CGraph | None:
        try:
            return self.parse_ug_fail(ugraph)
        except ModelReject:
            return None

    def parse_feat_fail(
        self, features: Sequence[tuple[str, int]]
    ) -> tuple[tuple[int, int], ...]:
        all_features = [(self._featuremap.get(k), c) for k, c in features]
        if None in map(operator.itemgetter(0), all_features):
            bad_features = sorted(
                frozenset(k for k, _ in features if k not in self._featuremap)
            )
            if len(bad_features) == 1:
                raise ModelReject("Missing feature: ", bad_features[0])
            raise ModelReject("Missing features: ", *bad_features)
        return tuple(all_features)  # type: ignore[arg-type]

    def parse_feat(
        self, features: Sequence[tuple[str, int]]
    ) -> tuple[tuple[int, int], ...] | None:
        try:
            return self.parse_feat_fail(features)
        except ModelReject:
            return None

    def debug(
        self, ugraph: UGraph | None, features: Iterable[str] | None
    ) -> Iterable[str]:
        if ugraph is not None:
            for lab in ugraph.get_labels():
                if lab not in self._colormap:
                    yield f"Unmapped label: label({lab})"
        if features is not None:
            for feat in features:
                if feat not in self._featuremap:
                    yield f"Unmapped feature: {feat}"

    def num_colors(self) -> int:
        return len(self._colormap)

    def get_colors(self) -> Iterable[str]:
        colors_sorted = sorted(
            self._colormap.items(), key=operator.itemgetter(1)
        )
        return map(operator.itemgetter(0), colors_sorted)

    def num_features(self) -> int:
        return len(self._featuremap)

    def get_features(self) -> Iterable[str]:
        feats_sorted = sorted(
            self._featuremap.items(), key=operator.itemgetter(1)
        )
        return map(operator.itemgetter(0), feats_sorted)


def model_colormap_from_json(json: JSON) -> ModelColorMap:
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "ModelColorMapSimple":
            return ModelColorMapSimple.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized color map model {model_type}")


def model_colormap_to_json(model_str: ModelColorMap) -> JSON:
    return {"type": type(model_str).__name__, "data": model_str.to_json()}


def convert_cgraph(cgmap: ModelColorMap, ugs: str) -> str | None:
    try:
        return convert_cgraph_fail(cgmap, ugs)
    except ModelReject:
        return None


def convert_cgraph_fail(cgmap: ModelColorMap, ugs: str) -> str:
    ug = SUGraph.from_str(ugs).to_ugraph()
    cg = cgmap.parse_ug_fail(ug)
    return cg.to_scgraph().as_str()
