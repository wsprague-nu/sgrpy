"""Functions for training models."""

import itertools
import operator
import pathlib
from collections.abc import Iterable, Iterator, Sequence

import numpy
import polars
import scipy
import tqdm

from sgrpy.graph import CanonGraph, SCGraph, SUGraph
from sgrpy.iotypes import save_dataframe
from sgrpy.linear import linreg
from sgrpy.train import FragModel, tune_model_classical
from sgrpy.train.graph_seed import most_common_grouped

from ._model_colormap import ModelColorMapSimple
from ._model_filter import CovarFilterFast
from ._model_fragtree import ModelPartitionTree
from ._model_predict import ModelPredict, ModelPredictLinear
from ._model_strparse import ModelStrParseRDKit
from ._model_types import (
    PTYPE_DATA,
    PTYPE_EXTERN_FEAT,
    PTYPE_GROUP,
    PTYPE_GROUP_ITEM,
    PTYPE_INTERN_FEAT,
    PTYPE_PRED_COL,
    PTYPE_WEIGHT,
    downconvert_lf,
    initialize_lf,
)
from ._models import SGRModel, SGRModel_v1

_NP_SQRT_F64EPS: numpy.float64 = numpy.sqrt(numpy.finfo(numpy.float64).eps)


def get_training_matrix(
    vecs: Iterable[scipy.sparse.csr_array],
) -> scipy.sparse.csr_array:
    all_vecs = tuple(vecs)
    big_matrix = scipy.sparse.vstack(all_vecs, dtype=numpy.int64)
    return big_matrix


def train_model_smiles_mol_df(
    df: polars.DataFrame | polars.LazyFrame,
    smiles: str,
    weights: str | None = None,
    groups: str | None = None,
    train_targets: Sequence[str] | None = None,
    extra_feat_col: str | None = None,
    stereo_pi: bool = True,
    extra_ring: bool = True,
    extra_conj: bool = True,
    feat_ring: int = 0,
    feat_conj: int = 0,
    trim: bool = True,
    min_count: int = 1,
    slimit: int | None = None,
    max_ssize: int | None = 1,
    train_iter: int = 5,
    prune_frac: float = 0.2,
    plimit: int | None = 64,
    max_iter: int | None = 65536,
    max_heap: int | None = 65536,
    wllimit: float = -100.0,
    n_jobs: int | None = None,
    progress: bool = False,
    covar_filter: bool = True,
    min_part_weight: float = _NP_SQRT_F64EPS,
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
    dataframe_out: pathlib.Path | str | None = None,
) -> SGRModel:
    """Train SMILES-based SGR model from DataFrame."""
    df_prime_input = initialize_lf(
        df,
        data_column=smiles,
        group_column=groups,
        weight_column=weights,
        extern_column=extra_feat_col,
        target_columns=train_targets,
        extern_from_str=False,
    )

    train_targets_col: tuple[str, ...]
    if train_targets is None:
        train_targets = ()
        train_targets_col = ()
    else:
        train_targets_col = tuple(f"value_{tname}" for tname in train_targets)

    # build initial dataframe
    df_input = df_prime_input.sort(
        ("group", "data", "weight", "extern_feat") + train_targets_col
    ).with_columns(
        polars.int_range(polars.len(), dtype=PTYPE_GROUP_ITEM).alias(
            "group_item"
        )
    )

    # create string parser object
    parse_str = ModelStrParseRDKit(
        stereo_pi=stereo_pi,
        ext_ring=extra_ring,
        ext_conj=extra_conj,
        feat_ring=feat_ring,
        feat_conj=feat_conj,
        trim=trim,
    )

    # enumerate stage 1 columns
    df_stage_1 = parse_str.parse_polars(df_input).collect(engine="streaming")

    # filter incomplete or ambiguous parses
    df_bad_rows: frozenset[int] = frozenset(
        df_stage_1.filter(polars.col("filter").ne("NONE")).select(
            polars.col("group_item").unique(),
        )["group_item"]
    )
    df_stage_1 = df_stage_1.filter(
        polars.col("group_item").is_in(df_bad_rows).not_()
    )

    # train colormap
    row_gen_stage_2_color: Iterator[tuple[int, int, str | None]] = (
        df_stage_1.select(
            polars.col("group"), polars.col("group_item"), polars.col("str_ug")
        ).iter_rows()
    )
    chunk_gen_stage_2_color = itertools.chain.from_iterable(
        (
            (group, group_item, label)
            for label in SUGraph.from_str(str_ug).to_ugraph().get_labels()
        )
        for group, group_item, str_ug in row_gen_stage_2_color
        if str_ug is not None
    )
    row_gen_stage_2_feature: Iterable[
        tuple[int, int, Sequence[dict[str, str | int]]]
    ] = df_stage_1.select(
        polars.col("group"),
        polars.col("group_item"),
        polars.col("extern_feat")
        .fill_null(polars.lit([], PTYPE_EXTERN_FEAT))
        .list.concat(
            polars.col("intern_feat").fill_null(
                polars.lit([], PTYPE_INTERN_FEAT)
            )
        ),
    ).iter_rows()
    chunk_gen_stage_2_feature: Iterator[tuple[int, int, str]] = (
        itertools.chain.from_iterable(
            ((group, group_item, label["name"]) for label in extras_list)  # type: ignore[misc]
            for group, group_item, extras_list in row_gen_stage_2_feature
        )
    )
    colormap = ModelColorMapSimple.from_chunks(
        chunk_gen_stage_2_color,
        min_count=min_count,
        feature_chunks=chunk_gen_stage_2_feature,
    )

    # create stage 2 dataframe
    df_stage_2_prev = colormap.parse_ug_polars(df_stage_1.lazy()).collect()
    df_bad_row_items: frozenset[int] = frozenset(
        df_stage_2_prev.filter(polars.col("filter").ne("NONE")).select(
            polars.col("group_item").unique(),
        )["group_item"]
    )
    df_stage_2 = df_stage_2_prev.filter(
        polars.col("group_item").is_in(df_bad_row_items).not_()
    ).lazy()
    df_stage_2_conc = (
        df_stage_2.select("str_cg", "group", "weight", "weight_ug")
        .filter(polars.col("str_cg").is_not_null())
        .collect(engine="streaming")
    )

    # train partitioner
    frag_gen = most_common_grouped(
        tuple(
            SCGraph.from_str(cgstr).to_cgraph()
            for cgstr in df_stage_2_conc["str_cg"]
        ),
        groups=df_stage_2_conc["group"].to_list(),
        weights=df_stage_2_conc.select(
            polars.col("weight").mul(polars.col("weight_ug")).alias("mulw")
        )["mulw"].to_list(),
        limit=slimit,
        min_count=min_count,
        max_size=max_ssize,
    )

    seed_fragments = (
        tuple(frag_gen)
        if not progress
        else tuple(tqdm.tqdm(frag_gen, desc="Generating seed fragments..."))
    )
    seed_graphs = map(
        lambda scg: CanonGraph.from_scanongraph(scg[0]), seed_fragments
    )
    seed_lweights = numpy.log(
        numpy.asarray(
            tuple(map(operator.itemgetter(1), seed_fragments)),
            dtype=numpy.float64,
        )
    )
    test_model = FragModel.new(seed_graphs, seed_lweights)
    cg_group_compact = (
        df_stage_2.filter(polars.col("str_cg").is_not_null())
        .group_by("group", "group_item", "group_norm")
        .agg(
            polars.col("str_cg"),
            polars.col("weight_ug")
            .mul(polars.col("weight"))
            .mean()
            .alias("gweight"),
        )
        .select("str_cg", "gweight")
        .collect(engine="streaming")
    )
    cgraphs_ex = tuple(
        tuple(SCGraph.from_str(cgs).to_cgraph() for cgs in cgg)
        for cgg in cg_group_compact["str_cg"]
    )
    cgraph_wts = cg_group_compact["gweight"]
    tuned_model = tune_model_classical(
        model=test_model,
        cgraphs=cgraphs_ex,
        group_weights=cgraph_wts,  # type: ignore
        train_iter=train_iter,
        prune_frac=prune_frac,
        wllimit=wllimit,
        plimit=plimit,
        max_iter=max_iter,
        max_heap=max_heap,
        n_jobs=n_jobs,
        progress=progress,
    )
    partitioner = ModelPartitionTree(
        fragmodel=tuned_model,
        limit=plimit,
        max_iter=max_iter,
        max_heap=max_heap,
        total_feat=colormap.num_features(),
    )

    if len(train_targets_col) == 0 and not covar_filter:
        if dataframe_out:
            save_dataframe(
                downconvert_lf(df_stage_2),
                dataframe_out,
            )

        final_model = SGRModel_v1(
            parser=parse_str,
            colormap=colormap,
            partition=partitioner,
            vfilter=None,
            predictors={},
        )

    # get feature matrix
    df_stage_3 = partitioner.plex_polars(df_stage_2).filter(
        polars.col("filter").eq(polars.lit("NONE"))
    )

    # find partition weight normalization factors
    df_max_lweights = (
        df_stage_3.group_by(("group_item", "group_norm"))
        .agg(polars.col("lweight_part"))
        .with_columns(
            polars.col("lweight_part").list.max().alias("lweight_shift")
        )
        .with_columns(
            polars.col("lweight_part")
            .sub(polars.col("lweight_shift"))
            .list.eval(polars.element().exp())
            .alias("lweight_part")
        )
        .with_columns(
            polars.col("lweight_part")
            .list.sum()
            .log()
            .add(polars.col("lweight_shift"))
            .alias("lweight_part")
        )
        .rename({"lweight_part": "lweight_norm"})
        .drop("lweight_shift")
    )

    # determine final weight, filter by epsilon weight
    df_stage_4 = (
        df_stage_3.join(df_max_lweights, on=("group_item", "group_norm"))
        .with_columns(
            polars.col("lweight_part")
            .sub("lweight_norm")
            .add(polars.col("weight").log())
            .add(polars.col("weight_ug").log())
            .exp()
            .alias("final_weight")
        )
        .filter(polars.col("final_weight").gt(min_part_weight))
    )

    # build training matrix
    num_factors = partitioner.get_size() + partitioner.num_features()
    vector_df = df_stage_4.select(
        ("final_weight", "csrvec") + tuple(train_targets_col)
    ).collect(engine="streaming")
    vector_gen = (
        scipy.sparse.csr_array(
            (csrval["data"], csrval["indices"], csrval["indptr"]),
            shape=(1, num_factors),
        )
        for csrval in vector_df["csrvec"]
    )
    training_matrix = get_training_matrix(vector_gen)
    if covar_filter:
        vfilter = CovarFilterFast.from_train_mat(
            training_matrix,
            atol=atol,
            btol=btol,
            rtol=1e-12,
            conlim=conlim,
            iter_lim=iter_lim,
        )
    else:
        vfilter = None

    if len(train_targets_col) == 0:
        if dataframe_out is not None:
            save_dataframe(
                downconvert_lf(df_stage_4),
                dataframe_out,
            )
        final_model = SGRModel_v1(
            parser=parse_str,
            colormap=colormap,
            partition=partitioner,
            vfilter=vfilter,
            predictors={},
        )
        return final_model

    train_models: dict[str, ModelPredict] = {}

    for tt_name, tt in zip(train_targets, train_targets_col, strict=True):
        if vector_df[tt].has_nulls():
            print(
                f"Warning: missing values for column `{tt}`, covariance filter "
                "may not work properly for predictions of this property."
            )
            train_df = vector_df.filter(polars.col(tt).is_not_null())
        else:
            train_df = vector_df
        w_vector = numpy.asarray(train_df["final_weight"], dtype=numpy.float64)
        target_vector = numpy.asarray(train_df[tt])
        vector_gen = (
            scipy.sparse.csr_array(
                (csrval["data"], csrval["indices"], csrval["indptr"]),
                shape=(1, num_factors),
            )
            for csrval in train_df["csrvec"]
        )
        training_matrix = get_training_matrix(vector_gen)

        coefs, _ = linreg(
            A=training_matrix,
            y=target_vector,
            sqweights=numpy.sqrt(w_vector),
            get_covar=False,
            atol=0,
            btol=0,
            conlim=0,
            iter_lim=iter_lim,
        )

        model_pred = ModelPredictLinear(coefs=coefs)
        train_models[tt_name] = model_pred

    if dataframe_out is not None:
        final_frame = df_stage_4
        for train_name, trained_model in train_models.items():
            final_frame = trained_model.predict_df(
                final_frame.lazy(), train_name
            )
        save_dataframe(
            downconvert_lf(final_frame),
            dataframe_out,
        )

    # assemble model
    final_model = SGRModel_v1(
        parser=parse_str,
        colormap=colormap,
        partition=partitioner,
        vfilter=vfilter,
        predictors=train_models,
    )

    return final_model


def train_model_smiles_mol(
    smiles: Sequence[str],
    weights: Sequence[float] | None = None,
    groups: Sequence[int] | None = None,
    train_targets: dict[str, Sequence[float | int | None]] | None = None,
    stereo_pi: bool = True,
    extra_ring: bool = True,
    extra_conj: bool = True,
    feat_ring: int = 0,
    feat_conj: int = 0,
    trim: bool = True,
    min_count: int = 1,
    slimit: int | None = None,
    max_ssize: int | None = 1,
    train_iter: int = 5,
    prune_frac: float = 0.2,
    plimit: int | None = 64,
    max_iter: int | None = 65536,
    max_heap: int | None = 65536,
    wllimit: float = -100.0,
    n_jobs: int | None = None,
    progress: bool = False,
    covar_filter: bool = True,
    min_part_weight: float = _NP_SQRT_F64EPS,
    atol: int | float = 1e-6,
    btol: int | float = 1e-6,
    conlim: int = 0,
    iter_lim: int = 10000,
    dataframe_out: pathlib.Path | str | None = None,
) -> SGRModel:
    """Train SMILES-based SGR model."""
    # test to make sure weights and groups are equal
    if weights is not None and len(weights) != len(smiles):
        raise ValueError(
            f"Number of weights ({len(weights)}) must equal number of SMILES "
            f"({len(smiles)})"
        )
    if groups is not None and len(groups) != len(smiles):
        raise ValueError(
            f"Number of groups ({len(groups)}) must equal number of SMILES "
            f"({len(smiles)})"
        )

    # check validity of training targets and organize them
    if train_targets is not None:
        for target_name, target_col in train_targets.items():
            if len(target_col) != len(smiles):
                raise ValueError(
                    f'Number of target values for "{target_name}" '
                    "({len(target_col)}) must "
                    f"equal number of SMILES ({len(smiles)})"
                )

        train_col_tags = {t_key: t_key for t_key in train_targets}
        train_cols = {
            tt: polars.Series(
                values=train_targets[tt_old], dtype=PTYPE_PRED_COL
            )
            for tt, tt_old in train_col_tags.items()
        }
    else:
        train_cols = {}
        train_col_tags = {}

    # build initial dataframe
    group_col = tuple(range(len(smiles))) if groups is None else groups
    weight_col = 1.0 if weights is None else weights
    initial_cols = {
        "group": group_col,
        "weight": weight_col,
        "data": smiles,
    }
    initial_cols.update(train_cols)  # type: ignore[arg-type,unused-ignore]
    df_input = polars.LazyFrame(
        initial_cols,
        schema={
            "group": PTYPE_GROUP,
            "weight": PTYPE_WEIGHT,
            "data": PTYPE_DATA,
        },
    )

    return train_model_smiles_mol_df(
        df=df_input,
        smiles="data",
        weights="weight",
        groups="group",
        train_targets=list(train_col_tags),
        stereo_pi=stereo_pi,
        extra_ring=extra_ring,
        extra_conj=extra_conj,
        feat_ring=feat_ring,
        feat_conj=feat_conj,
        trim=trim,
        min_count=min_count,
        slimit=slimit,
        max_ssize=max_ssize,
        train_iter=train_iter,
        prune_frac=prune_frac,
        plimit=plimit,
        max_iter=max_iter,
        max_heap=max_heap,
        wllimit=wllimit,
        n_jobs=n_jobs,
        progress=progress,
        covar_filter=covar_filter,
        min_part_weight=min_part_weight,
        atol=atol,
        btol=btol,
        conlim=conlim,
        iter_lim=iter_lim,
        dataframe_out=dataframe_out,
    )
