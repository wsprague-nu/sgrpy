"""Model structures."""

import abc
import collections
import dataclasses
import itertools
import json
import pathlib
from collections.abc import Iterable, Sequence

import numpy
import polars
import scipy

from sgrpy.graph import CGraph, SUGraph, UGraph
from sgrpy.iotypes import (
    JSON,
    JSONable,
    json_read_compressed,
    json_to_dict,
    json_write_compressed,
)
from sgrpy.linear import dict_to_csrvecint

from ._model_colormap import (
    ModelColorMap,
    model_colormap_from_json,
    model_colormap_to_json,
)
from ._model_filter import (
    ModelFilter,
    model_filter_from_json,
    model_filter_to_json,
)
from ._model_fragtree import (
    ModelPlex,
    model_partition_from_json,
    model_partition_to_json,
)
from ._model_predict import (
    ModelPredict,
    model_predict_from_json,
    model_predict_to_json,
)
from ._model_strparse import (
    ModelStrParse,
    model_str_from_json,
    model_str_to_json,
)
from ._model_types import (
    PTYPE_DATA,
    PTYPE_EXTERN_FEAT_STR,
    PTYPE_GROUP_ITEM,
    UGraphStruct,
    initialize_lf,
)

_POLARS_NO_OPTIMIZE = polars.QueryOptFlags.none()


def compact_lf(
    lf: polars.DataFrame | polars.LazyFrame,
    pred_cols: Iterable[str] | None = None,
    strict: bool = True,
) -> polars.LazyFrame:
    """Compact LazyFrame containing process information.

    This process first performs a weighted average of `lweight_part` and all
    prediction columns between valid rows which are part of
    the same original row and initial UGraph.  It then performs a weighted
    average between different UGraph translations for each original row.  If
    done strictly, the second weighted average will have a null result if any of
    the UGraph translations had no valid final rows/predictions.

    As an example, this means that if a molecule containing ambiguous
    cis/trans stereochemistry were fed in initially, the model would attempt to
    determine values for both stereoisomers with equal weight.  When both
    resulting UGraphs produce a prediction, this value will be averaged.  When
    only one of them produces a prediction, `strict=True` will return a null
    result, but `strict=False` will return the value which did succeed.

    Parameters
    ----------
    lf : polars.LazyFrame | polars.DataFrame
        Input data (must be correctly formatted under the right column spec).
        Should be correct if outputted by `Model.process_df`.
        Required columns: "data", "group", "weight", "filter", "extern_feat",
        "intern_feat", "group_item", "weight_ug", "str_ug", "group_norm",
        "str_cg", "extras_feat", "lweight_part", "csrvec", "lweight_norm",
        and all values in `pred_cols`, if any.
    pred_cols : Iterable[str] | None
        Names of prediction columns in `lf`, should start with `"pred_"`
        (default: None).
    strict : bool
        Whether to aggregate strictly (default: True).  See documentation for
        more details.

    Returns
    -------
    polars.LazyFrame
    """
    lf = lf.lazy()
    pred_cols = () if pred_cols is None else tuple(pred_cols)

    # lf_grouped = (
    #     lf.group_by(("group", "group_item", "group_norm"))
    #     .agg(
    #         polars.col(
    #             "data",
    #             "weight",
    #             "extern_feat",
    #             "intern_feat",
    #             "weight_ug",
    #             "str_ug",
    #             "str_cg",
    #             "extras_feat",
    #         ),
    #         polars.struct(
    #             ("filter", "lweight_part", "csrvec") + pred_cols
    #         ).alias("megastruct"),
    #     )
    #     .with_columns(
    #         polars.col(
    #             "data",
    #             "weight",
    #             "extern_feat",
    #             "intern_feat",
    #             "weight_ug",
    #             "str_ug",
    #             "str_cg",
    #             "extras_feat",
    #         ).list.first()
    #     )
    # )

    if strict:
        # strict filtering permits no invalid entries per top level group
        lf_filtered = (
            lf.with_columns(
                polars.col("filter")
                .ne("NONE")
                .all()
                .over("group", "group_item", "group_norm")
                .alias("bad_part")
            )
            .with_columns(
                polars.col("bad_part")
                .any()
                .not_()
                .over("group", "group_item")
                .alias("good")
            )
            .filter(polars.col("good"))
            .drop("bad_part", "good")
        )
    else:
        # non-strict filtering simply removes bad rows
        lf_filtered = lf.filter(polars.col("filter").eq("NONE"))

    lf_filtered_max = lf_filtered.with_columns(
        polars.col("lweight_part")
        .max()
        .over("group", "group_item", "group_norm")
        .alias("lweight_max")
    ).with_columns(polars.col("lweight_part").sub("lweight_max"))

    lf_grouped = (
        lf_filtered_max.group_by(("group", "group_item", "group_norm"))
        .agg(
            polars.col(
                "data",
                "weight",
                "extern_feat",
                "intern_feat",
                "weight_ug",
                "str_ug",
                "str_cg",
                "extras_feat",
                "lweight_max",
            ),
            polars.struct(("lweight_part", "csrvec") + pred_cols).alias(
                "megastruct"
            ),
        )
        .with_columns(
            polars.col(
                "data",
                "weight",
                "extern_feat",
                "intern_feat",
                "weight_ug",
                "str_ug",
                "str_cg",
                "extras_feat",
                "lweight_max",
            ).list.first()
        )
    )

    lf_grouped_agg = (
        lf_grouped.with_columns(
            polars.col("megastruct")
            .list.eval(polars.element().struct.field("lweight_part").exp())
            .list.sum()
            .alias("lweight_sum")
        )
        .with_columns(
            polars.col("lweight_sum")
            .log()
            .add(polars.col("lweight_max"))
            .alias("lweight_part")
        )
        .drop("lweight_max")
    )
    for pred_col in pred_cols:
        lf_grouped_agg = lf_grouped_agg.with_columns(
            polars.col("megastruct")
            .list.eval(
                polars.element()
                .struct.field("lweight_part")
                .exp()
                .mul(polars.element().struct.field(pred_col))
            )
            .list.sum()
            .truediv("lweight_sum")
            .alias(pred_col),
        )

    lf_partweighted = lf_grouped_agg.drop("lweight_sum", "megastruct")

    # finally, average each thing over the weight_ug components
    lf_final = (
        lf_partweighted.with_columns(
            polars.col("weight_ug").truediv(
                polars.col("weight_ug").sum().over("group", "group_item")
            )
        )
        .with_columns(
            polars.col(("lweight_part",) + pred_cols).mul("weight_ug")
        )
        .group_by("group", "group_item")
        .agg(
            polars.col(
                "data",
                "weight",
                "extern_feat",
            ).first(),
            polars.col(("lweight_part",) + pred_cols).sum(),
        )
        .drop("group_item")
    )

    return lf_final


@dataclasses.dataclass(frozen=True, slots=True)
class _DataPartRow:
    lweight_part: float
    csrvec: scipy.sparse.csr_array
    props: dict[str, float] | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _DataParseRow:
    weight_ug: float
    ug: UGraph | None
    intern_feat: Sequence[tuple[str, int]] | None
    cg: CGraph | None = None
    extras_feat: Sequence[tuple[int, int]] | None = None
    part_rows: list[_DataPartRow] | None = None

    @classmethod
    def new(
        cls,
        weight_ug: float,
        ug: UGraph | None,
        extras: Sequence[str],
    ) -> "_DataParseRow | None":
        if ug is None and len(extras) == 0:
            return None

        intern_feat = tuple(collections.Counter(extras).items())
        return _DataParseRow(
            weight_ug=weight_ug, ug=ug, intern_feat=intern_feat
        )

    def calc_info(self) -> float | None:
        dpprows = self.part_rows
        if dpprows is None:
            return None
        infos = numpy.asarray(
            [x.lweight_part for x in dpprows], dtype=numpy.float64
        )
        info_shift = numpy.max(infos)
        infos = numpy.log(numpy.sum(numpy.exp(infos - info_shift))) + info_shift

        return float(infos)

    def calc_prop(self, prop: str) -> float | None:
        dpprows = self.part_rows
        if dpprows is None:
            return None
        lweights = numpy.asarray(
            [x.lweight_part for x in dpprows], dtype=numpy.float64
        )
        props = numpy.asarray(
            [x.props[prop] for x in dpprows],  # type: ignore
            dtype=numpy.float64,
        )
        return float(numpy.average(props, weights=numpy.exp(lweights)))


@dataclasses.dataclass(frozen=True, slots=True)
class _DataRow:
    data: str | None
    extern_feat: list[tuple[str, int]] | None
    parse_rows: list[_DataParseRow] | None
    filt_status: str = "NONE"

    @classmethod
    def new(
        cls,
        data: str | None = None,
        extern_feat: list[tuple[str, int]] | None = None,
    ) -> "_DataRow":
        if data == "":
            data = None
        if extern_feat is not None and len(extern_feat) == 0:
            extern_feat = None
        return _DataRow(data=data, extern_feat=extern_feat, parse_rows=None)

    def add_ugs(self, ugs: UGraphStruct) -> "_DataRow":
        if len(ugs.weights) == 0:
            return self
        all_weights = numpy.asarray(ugs.weights, dtype=numpy.float64)
        all_weights = all_weights / numpy.sum(all_weights)
        parse_rows = [
            (_DataParseRow.new(float(weight), graph, extras))
            for weight, graph, extras in zip(
                all_weights,
                ugs.graphs,
                ugs.extras,
                strict=True,
            )
        ]
        if len(parse_rows) == 0 or any(r is None for r in parse_rows):
            return self
        return dataclasses.replace(self, parse_rows=parse_rows)  # type: ignore[arg-type]

    def add_blank(self) -> "_DataRow":
        dpr = _DataParseRow(1.0, None, None)
        return dataclasses.replace(self, parse_rows=[dpr])

    def calc_info(self) -> float | None:
        dprows = self.parse_rows
        if dprows is None:
            return None
        infos: list[float] = []
        weights: list[float] = []
        for dp in dprows:
            weights.append(dp.weight_ug)
            info = dp.calc_info()
            if info is None:
                return None
            infos.append(info)
        if len(infos) == 0:
            return None

        return float(numpy.average(infos, weights=weights))

    def calc_prop(self, prop: str) -> float | None:
        dprows = self.parse_rows
        if dprows is None:
            return None
        values: list[float] = []
        weights: list[float] = []
        for dp in dprows:
            weights.append(dp.weight_ug)
            value = dp.calc_prop(prop)
            if value is None:
                return None
            values.append(value)
        if len(values) == 0:
            return None

        return float(numpy.average(values, weights=weights))

    def calc_filter(self) -> bool:
        return self.filt_status == "NONE"


class SGRModel(JSONable):
    """Model implementing SGR methods."""

    @abc.abstractmethod
    def process_df(
        self,
        df: polars.LazyFrame | polars.DataFrame,
        predictions: Sequence[str] | None = None,
        disable_filter: bool = False,
    ) -> polars.LazyFrame:
        """Process DataFrame input."""

    @abc.abstractmethod
    def debug(
        self, data: str | None, extern_feat: str | None = None
    ) -> Iterable[str]:
        """Debug any issues with processing `data` input."""

    @abc.abstractmethod
    def get_ugraphs(
        self, data: str
    ) -> Iterable[tuple[float, UGraph | None, Sequence[str]]]:
        """Get mapped UGraph from `data` string."""

    @abc.abstractmethod
    def get_data_labels(self, data: str) -> Iterable[Iterable[str]]:
        """Return all vertex labels present in `data` input."""

    @abc.abstractmethod
    def get_labels(self) -> Iterable[str]:
        """Return registered/mappable vertex labels."""

    @abc.abstractmethod
    def get_data_features(self, data: str) -> Iterable[Iterable[str]]:
        """Return all internal features assigned to `data` input."""

    @abc.abstractmethod
    def get_features(self) -> Iterable[str]:
        """Return all registered/mappable internal features."""

    @abc.abstractmethod
    def get_final_features(
        self, data: str, extern_feat: str | None = None
    ) -> Iterable[Iterable[Iterable[str]]]:
        """Return all features found in `data` input."""

    @abc.abstractmethod
    def calc_info(
        self,
        data: str,
        extern_feat: str | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> float | None:
        """Calculate information of `data` input."""

    @abc.abstractmethod
    def calc_est(
        self,
        data: str,
        prop: str,
        extern_feat: str | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> float | None:
        """Calculate prediction of `prop` from `data` input."""

    @abc.abstractmethod
    def est_filter(
        self,
        data: str,
        extern_feat: str | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> bool:
        """Determine whether `data` passes the model filter."""

    @abc.abstractmethod
    def filter_status(
        self,
        data: str,
        extern_feat: str | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> str:
        """Return status of model filter."""


@dataclasses.dataclass(frozen=True, slots=True)
class SGRModel_v1(SGRModel):
    parser: ModelStrParse
    colormap: ModelColorMap
    partition: ModelPlex
    vfilter: ModelFilter | None
    predictors: dict[str, ModelPredict]

    @classmethod
    def _from_json(cls, json: JSON) -> "SGRModel_v1":
        json_dict = json_to_dict(json)
        parser = model_str_from_json(json_dict["parser"])
        colormap = model_colormap_from_json(json_dict["colormap"])
        partition = model_partition_from_json(json_dict["partition"])
        vfilter = (
            model_filter_from_json(json_dict["vfilter"])
            if json_dict["vfilter"] not in ("", None)
            else None
        )
        predictors = {
            name: model_predict_from_json(modeldat)
            for name, modeldat in json_to_dict(json_dict["predictors"]).items()
        }
        return SGRModel_v1(
            parser=parser,
            colormap=colormap,
            partition=partition,
            vfilter=vfilter,
            predictors=predictors,
        )

    def _eval_single_fast(
        self,
        data: str | None,
        extern_feat: list[tuple[str, int]] | None,
        properties: list[str] | None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> _DataRow:
        initial_row = _DataRow.new(data, extern_feat)

        # perform string parse
        if initial_row.data is not None:
            strparse = self.parser.parse_str(initial_row.data)
            row_stage_2 = initial_row.add_ugs(strparse)
        elif initial_row.extern_feat is not None:
            row_stage_2 = initial_row.add_blank()
        else:
            return dataclasses.replace(initial_row, filt_status="PARSE")
        if row_stage_2.parse_rows is None:
            return dataclasses.replace(row_stage_2, filt_status="PARSE")

        # perform colorization
        new_dprows: list[_DataParseRow] = []
        for dprow in row_stage_2.parse_rows:
            feat_strs = [] if extern_feat is None else extern_feat
            feat_strs.extend(
                [] if dprow.intern_feat is None else dprow.intern_feat
            )
            if len(feat_strs) > 0:
                extras = self.colormap.parse_feat(feat_strs)
                if extras is None:
                    if strict:
                        return dataclasses.replace(
                            row_stage_2, filt_status="COLOR"
                        )
                    else:
                        continue
            else:
                extras = None
            if dprow.ug is not None:
                cg = self.colormap.parse_ug(dprow.ug)
                if cg is None:
                    if strict:
                        return dataclasses.replace(
                            row_stage_2, filt_status="COLOR"
                        )
                    else:
                        continue
            else:
                cg = None
            new_dprow = dataclasses.replace(dprow, cg=cg, extras_feat=extras)
            new_dprows.append(new_dprow)

        # return failure if no colorizations available
        if len(new_dprows) == 0:
            return dataclasses.replace(row_stage_2, filt_status="COLOR")
        row_stage_3 = dataclasses.replace(row_stage_2, parse_rows=new_dprows)

        # perform partition
        filt_rej = True
        new_dprows = []
        for dprow in row_stage_3.parse_rows:  # type: ignore
            all_lweights: list[float] = []
            all_vecs: list[scipy.sparse.coo_array] = []

            plex_result = self.partition.plex_cg(dprow.cg, dprow.extras_feat)
            if plex_result is None:
                if strict:
                    return dataclasses.replace(
                        row_stage_3, filt_status="PARTITION"
                    )
                else:
                    filt_rej = False
                    continue

            for lweight, feat_coo in plex_result:
                all_lweights.append(lweight)
                all_vecs.append(scipy.sparse.csr_array(feat_coo))

            if len(all_lweights) == 0:
                if strict:
                    return dataclasses.replace(
                        row_stage_3, filt_status="PARTITION"
                    )
                else:
                    filt_rej = False
                    continue

            new_dpprows: list[_DataPartRow] = []
            for lweight_part, csr_array in zip(
                all_lweights, all_vecs, strict=True
            ):
                if (
                    not disable_filter
                    and self.vfilter is not None
                    and not self.vfilter.filter(csr_array.todense()[0, :])
                ):
                    continue
                if properties is not None:
                    proplist: dict[str, float] = {}
                    for prop in properties:
                        proplist[prop] = self.predictors[prop].predict(
                            csr_array.todense()[0, :]
                        )
                    new_dpprows.append(
                        _DataPartRow(lweight_part, csr_array, props=proplist)
                    )
                else:
                    new_dpprows.append(_DataPartRow(lweight_part, csr_array))
            if len(new_dpprows) == 0:
                match strict, len(all_lweights) > 0:
                    case True, True:
                        return dataclasses.replace(
                            row_stage_3, filt_status="FILTER"
                        )
                    case True, False:
                        return dataclasses.replace(
                            row_stage_3, filt_status="PARTITION"
                        )
                    case False, True:
                        continue
                    case False, False:
                        filt_rej = False
                        continue

            new_dprows.append(dataclasses.replace(dprow, part_rows=new_dpprows))

        if len(new_dprows) == 0:
            stat = "FILTER" if filt_rej else "PARTITION"
            return dataclasses.replace(row_stage_3, filt_status=stat)
        row_stage_4 = dataclasses.replace(row_stage_3, parse_rows=new_dprows)

        return row_stage_4

    def get_ugraphs(
        self, data: str
    ) -> Iterable[tuple[float, UGraph | None, Sequence[str]]]:
        map_ugs = self.parser.parse_str(data)

        return (
            (weight, graph, feats)
            for weight, graph, feats in zip(
                map_ugs.weights, map_ugs.graphs, map_ugs.extras, strict=True
            )
        )

    def calc_info(
        self,
        data: str,
        extern_feat: str | Sequence[tuple[str, int]] | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> float | None:
        if isinstance(extern_feat, str):
            json_datadict = json.loads(extern_feat)
            if "extern_feat" not in json_datadict:
                extern_feat = None
            else:
                datalist = json_datadict["extern_feat"]
                if datalist is None or len(datalist) == 0:
                    extern_feat = None
                else:
                    extern_feat = [
                        (entry["name"], entry["count"]) for entry in datalist
                    ]
        datapoint = self._eval_single_fast(
            data=data,
            extern_feat=list(extern_feat) if extern_feat is not None else None,
            properties=None,
            disable_filter=disable_filter,
            strict=strict,
        )
        return datapoint.calc_info()

    def calc_est(
        self,
        data: str,
        prop: str,
        extern_feat: str | Sequence[tuple[str, int]] | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> float | None:
        if isinstance(extern_feat, str):
            json_datadict = json.loads(extern_feat)
            if "extern_feat" not in json_datadict:
                extern_feat = None
            else:
                datalist = json_datadict["extern_feat"]
                if datalist is None or len(datalist) == 0:
                    extern_feat = None
                else:
                    extern_feat = [
                        (entry["name"], entry["count"]) for entry in datalist
                    ]
        datapoint = self._eval_single_fast(
            data=data,
            extern_feat=list(extern_feat) if extern_feat is not None else None,
            properties=[prop],
            disable_filter=disable_filter,
            strict=strict,
        )
        return datapoint.calc_prop(prop)

    def filter_status(
        self,
        data: str,
        extern_feat: str | Sequence[tuple[str, int]] | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> str:
        if isinstance(extern_feat, str):
            json_datadict = json.loads(extern_feat)
            if "extern_feat" not in json_datadict:
                extern_feat = None
            else:
                datalist = json_datadict["extern_feat"]
                if datalist is None or len(datalist) == 0:
                    extern_feat = None
                else:
                    extern_feat = [
                        (entry["name"], entry["count"]) for entry in datalist
                    ]
        datapoint = self._eval_single_fast(
            data=data,
            extern_feat=list(extern_feat) if extern_feat is not None else None,
            properties=None,
            disable_filter=disable_filter,
            strict=strict,
        )
        return datapoint.filt_status

    def est_filter(
        self,
        data: str,
        extern_feat: str | Sequence[tuple[str, int]] | None = None,
        disable_filter: bool = False,
        strict: bool = True,
    ) -> bool:
        return (
            self.filter_status(
                data=data,
                extern_feat=extern_feat,
                disable_filter=disable_filter,
                strict=strict,
            )
            == "NONE"
        )

    def process_df(
        self,
        df: polars.LazyFrame | polars.DataFrame,
        predictions: Sequence[str] | None = None,
        disable_filter: bool = False,
        compact_lf: bool = False,
    ) -> polars.LazyFrame:
        df_input = df.lazy().with_columns(
            polars.int_range(polars.len(), dtype=PTYPE_GROUP_ITEM).alias(
                "group_item"
            )
        )

        # enumerate ugraphs
        df_stage_1 = self.parser.parse_polars(df_input)

        # get cgraphs
        df_stage_2 = self.colormap.parse_ug_polars(df_stage_1)

        # partition
        df_stage_3 = self.partition.plex_polars(df_stage_2)

        # apply filter
        self_vfilter = self.vfilter
        df_stage_4 = (
            df_stage_3
            if disable_filter or self_vfilter is None
            else self_vfilter.filter_df(df_stage_3)
        )

        # apply predictions
        if predictions is None:
            return df_stage_4

        for prediction_str in predictions:
            df_stage_4 = self.predictors[prediction_str].predict_df(
                df_stage_4, prediction_str
            )

        return df_stage_4

    def to_json(self) -> JSON:
        parser_json = model_str_to_json(self.parser)
        colormap_json = model_colormap_to_json(self.colormap)
        partition_json = model_partition_to_json(self.partition)
        vfilter_json = (
            None if self.vfilter is None else model_filter_to_json(self.vfilter)
        )
        predictors_json = {
            name: model_predict_to_json(pred)
            for name, pred in self.predictors.items()
        }
        return {
            "parser": parser_json,
            "colormap": colormap_json,
            "partition": partition_json,
            "vfilter": vfilter_json,
            "predictors": predictors_json,
        }

    def debug(
        self, data: str | None, extern_feat: str | None = None
    ) -> Iterable[str]:
        df = polars.DataFrame(
            (
                polars.Series("data", [data], dtype=PTYPE_DATA),
                polars.Series(
                    "extern_feat", [extern_feat], dtype=PTYPE_EXTERN_FEAT_STR
                ),
            )
        )
        lf_orig = self.process_df(
            initialize_lf(df, "data", None, None, "extern_feat", None, False)
        )
        feat_lookup = list(self.get_features())
        frag_lookup = self.partition.get_labels()
        num_frags = len(frag_lookup)
        num_feat_total = num_frags + len(feat_lookup)
        df_groups = (
            lf_orig.filter(polars.col("filter").ne("NONE")).sort(
                "group",
                "group_item",
                "group_norm",
                "filter",
                polars.col("lweight_part").neg(),
            )
        ).collect(engine="streaming")
        for row in df_groups.iter_rows(named=True):
            filter_msg = str(row["filter"])
            match filter_msg:
                case "FILTER":
                    if self.vfilter is None:
                        yield (
                            "FILTER: model has no filter, this is a bug in "
                            "filter tagging"
                        )
                        continue
                    vec_entry = row["csrvec"]
                    if vec_entry is None:
                        yield (
                            "FILTER: no entry in csrvec, this is a bug in "
                            "filter tagging"
                        )
                        continue
                    vec = dict_to_csrvecint(
                        row["csrvec"], num_feat_total
                    ).toarray()[0, :]
                    for errmsg in self.vfilter.debug(vec):
                        yield f"FILTER: {errmsg}"
                case "PARSE":
                    data = row["data"]
                    extern_feat = row["extern_feat"]
                    if (data is None or data == "") and (
                        extern_feat is None or len(extern_feat) == 0
                    ):
                        yield (
                            "PARSE: both data and extern_feat fields are empty"
                        )
                        continue
                    if data is None:
                        continue
                    for errmsg in self.parser.debug(row["data"]):
                        yield f"PARSE: {errmsg}"
                case "COLOR":
                    row_str_ug = row["str_ug"]
                    row_extern_feat = row["extern_feat"]
                    row_intern_feat = row["intern_feat"]
                    if (
                        row_str_ug is None
                        and (
                            row_extern_feat is None or len(row_extern_feat) == 0
                        )
                        and (
                            row_intern_feat is None or len(row_intern_feat) == 0
                        )
                    ):
                        yield (
                            "COLOR: str_ug, extern_feat, and intern_feat "
                            "fields do not exist; this is a bug"
                        )
                        continue

                    ug = (
                        None
                        if row_str_ug is None
                        else SUGraph.from_str(row_str_ug).to_ugraph()
                    )
                    row_extern_feat = (
                        [] if row_extern_feat is None else row_extern_feat
                    )
                    row_intern_feat = (
                        [] if row_intern_feat is None else row_intern_feat
                    )
                    features = (
                        feat["name"]
                        for feat in itertools.chain(
                            row_extern_feat, row_intern_feat
                        )
                        if feat["count"] != 0
                    )
                    for errmsg in self.colormap.debug(ug, features):
                        yield f"COLOR: {errmsg}"
                case "PARTITION":
                    yield (
                        "PARTITION: likely ran out of time (try increasing "
                        "maximum heap or iterations)"
                    )
                case _:
                    raise ValueError(f"Unknown filter error type {filter_msg}")

    def get_labels(self) -> Iterable[str]:
        return (f"label({lab})" for lab in self.colormap.get_colors())

    def get_data_labels(self, data: str) -> Iterable[Iterable[str]]:
        parsed = self.parser.parse_str(data)
        for graph in parsed.graphs:
            yield (
                ()
                if graph is None
                else (f"label({lab})" for lab in graph.get_labels())
            )

    def get_features(self) -> Iterable[str]:
        return self.colormap.get_features()

    def get_data_features(self, data: str) -> Iterable[Iterable[str]]:
        parsed = self.parser.parse_str(data)
        return parsed.extras

    def get_final_features(
        self, data: str, extern_feat: str | None = None
    ) -> Iterable[Iterable[Iterable[str]]]:
        df = polars.DataFrame(
            (
                polars.Series("data", [data], dtype=PTYPE_DATA),
                polars.Series(
                    "extern_feat", [extern_feat], dtype=PTYPE_EXTERN_FEAT_STR
                ),
            )
        )
        lf_orig = self.process_df(
            initialize_lf(df, "data", None, None, "extern_feat", None, False),
            disable_filter=True,
        )
        feat_lookup = list(self.get_features())
        frag_lookup = self.partition.get_labels()
        num_frags = len(frag_lookup)
        num_feat_total = num_frags + len(feat_lookup)
        lf_groups = (
            lf_orig.group_by("group", "group_item", "group_norm")
            .agg("csrvec")
            .group_by("group", "group_item")
            .agg("csrvec")
        ).collect(engine="streaming")
        assert len(lf_groups) < 2  # noqa: PLR2004
        if len(lf_groups) == 0:
            return ()
        group_collect: list[list[dict[str, list[int]]]] = lf_groups["csrvec"][
            0
        ].to_list()
        return (
            (
                (
                    (
                        f"frag({frag_lookup[i]})"
                        if i < num_frags
                        else feat_lookup[i + num_frags]
                    )
                    for _, i in dict_to_csrvecint(
                        csr_dict, num_feat_total
                    ).nonzero()
                )
                for csr_dict in mg
            )
            for mg in group_collect
        )


def save_model_to_json(model: SGRModel) -> JSON:
    """Save model to JSON (not string but actual nested objects).

    Parameters
    ----------
    model : SGRModel
        Model to be saved.

    Returns
    -------
    JSON
    """
    return {"type": type(model).__name__, "data": model.to_json()}


def load_model_from_json(json: JSON) -> SGRModel:
    """Load model from JSON (not string but actual nested objects).

    Parameters
    ----------
    json : JSON
        JSON data containing model information.

    Returns
    -------
    SGRModel
    """
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "SGRModel_v1":
            return SGRModel_v1.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized model type {model_type}")


def save_model(model: SGRModel, path: pathlib.Path | str) -> None:
    """Save model to file.

    File is saved as a compressed JSON.

    Parameters
    ----------
    model : SGRModel
        Model to be saved.
    path : pathlib.Path | str
        Path of the destination file.
    """
    json_data = save_model_to_json(model)
    json_write_compressed(json_data, path)


def load_model(path: pathlib.Path | str) -> SGRModel:
    """Load model from file.

    File is loaded from compressed JSON.

    Parameters
    ----------
    path : pathlib.Path | str
        Path of the destination file.

    Returns
    -------
    SGRModel
    """
    json_data = json_read_compressed(path)
    return load_model_from_json(json_data)
