"""Primary command-line interface for `sgrpy`."""

import argparse
import importlib.util
import math
import pathlib
import sys
from collections.abc import Callable
from typing import Never

import numpy

from sgrpy.iotypes import FileType, load_dataframe, save_dataframe
from sgrpy.model import (
    downconvert_lf,
    initialize_lf,
    load_model,
    save_model,
    train_model_smiles_mol_df,
)

_NP_SQRT_F64EPS: numpy.float64 = numpy.sqrt(numpy.finfo(numpy.float64).eps)


def unpack_error(e: Exception) -> Never:
    args = e.args
    match len(args):
        case 0:
            raise e
        case 1:
            sys.exit(args[0])
        case _:
            sys.exit("\n".join(args))


def map_layer_feat(in_val: str | None) -> int:
    match in_val:
        case None:
            return 0
        case "core":
            return 1
        case "extended":
            return 2
        case _:
            raise ValueError(f'Invalid feature specifier "{in_val}"')


def cli_train(args: argparse.Namespace) -> None:
    # perform check for RDKit
    method: str | None = args.mode
    if method is not None and (
        method in ("smiles-mol", "smiles-rxn")
        and importlib.util.find_spec("rdkit") is None
    ):
        sys.exit(
            "Please make sure `rdkit` is in your Python environment "
            f'before running in mode "{method}".'
        )

    # gather data
    filetype_key: str | None = args.format
    data_path: pathlib.Path = args.data
    try:
        filetype = (
            FileType.from_str(filetype_key)
            if filetype_key is not None
            else None
        )
        all_data = load_dataframe(path=data_path, filetype=filetype)
    except ValueError as e:
        unpack_error(e)

    # get column names
    data_col: str = args.data_column
    weight_col: str | None = args.weight_column
    group_col: str | None = args.group_column
    extra_feat_col: str | None = args.extra_feat_column

    # get runtime parameters
    stereo_pi = bool(args.stereo_pi)
    extra_ring = True if args.extra_ring is None else args.extra_ring
    extra_conj = True if args.extra_ring is None else args.extra_ring
    feat_ring = map_layer_feat(args.feat_ring)
    feat_conj = map_layer_feat(args.feat_conj)
    trim = bool(args.trim)
    min_count = int(args.min_count)
    if min_count < 1:
        sys.exit(f"`--min-count` must be positive (was {min_count})")
    slimit = int(args.slimit) if args.slimit is not None else None
    if slimit is not None and slimit < 1:
        slimit = None
    max_ssize = int(args.max_ssize) if args.max_ssize is not None else None
    if max_ssize is not None and max_ssize < 1:
        max_ssize = None
    train_iter = int(args.train_iter)
    if train_iter < 1:
        sys.exit(f"`--train-iter` must be positive (was {train_iter})")
    prune_frac = float(args.prune_frac)
    if prune_frac >= 1.0 or prune_frac < 0.0 or not math.isfinite(prune_frac):
        sys.exit(
            "`--prune-frac` must be between 0.0 inclusive and 1.0 exclusive "
            f"(was {prune_frac})"
        )
    plimit = int(args.plimit) if args.plimit is not None else None
    if plimit is not None and plimit < 1:
        plimit = None
    max_iter = int(args.max_iter) if args.max_iter is not None else None
    if max_iter is not None and max_iter < 1:
        max_iter = None
    max_heap = int(args.max_heap) if args.max_heap is not None else None
    if max_heap is not None and max_heap < 1:
        max_heap = None
    wllimit = float(args.wllimit)
    n_jobs = int(args.num_cores) if args.num_cores is not None else None
    if n_jobs is not None and n_jobs < 1:
        n_jobs = None
    progress = bool(args.progress)
    disable_covar = bool(args.disable_covar)
    table_out_path: pathlib.Path | None = args.data_output
    target_cols = () if args.target_column is None else args.target_column
    atol = float(args.atol)
    btol = float(args.btol)
    conlim = int(args.conlim)
    min_part_weight = float(args.min_rel_weight)
    iter_lim = int(args.fit_iter_lim)

    # if partitions not enabled, disable partitioning
    enable_part = bool(args.enable_part)
    if not enable_part:
        max_ssize = 1

    match args.mode:
        case "smiles-mol":
            trained_model = train_model_smiles_mol_df(
                df=all_data,
                smiles=data_col,
                weights=weight_col,
                groups=group_col,
                train_targets=target_cols,
                extra_feat_col=extra_feat_col,
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
                covar_filter=(not disable_covar),
                min_part_weight=min_part_weight,
                atol=atol,
                btol=btol,
                conlim=conlim,
                iter_lim=iter_lim,
                dataframe_out=table_out_path,
            )
        case _:
            sys.exit(f'Unimplemented operation mode "{args.mode}"')

    # save model to path
    out_path: pathlib.Path = args.outfile
    save_model(trained_model, out_path)


def cli_map(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    lf_loaded = load_dataframe(args.datafile)
    data_column: str = args.data_column
    group_column: str | None = args.group_column
    weight_column: str | None = args.weight_column
    extra_column: str | None = args.extra_feat_column
    target_col: list[str] | None = (
        None if args.target_column is None else list(args.target_column)
    )

    lf_select = initialize_lf(
        lf_loaded,
        data_column=data_column,
        group_column=group_column,
        weight_column=weight_column,
        extern_column=extra_column,
        target_columns=target_col,
        extern_from_str=True,
    )

    processed = model_loaded.process_df(lf_select, predictions=target_col)

    processed_out = downconvert_lf(processed)

    save_dataframe(processed_out, path=args.outfile, filetype=args.out_format)


def cli_eval_info(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    target: str = args.input
    disable_filter = bool(args.disable_filter)
    debug = bool(args.debug)
    strict = not bool(args.nostrict)
    extern_feat = None if args.extern_feat is None else str(args.extern_feat)
    result = model_loaded.calc_info(
        data=target,
        extern_feat=extern_feat,
        disable_filter=disable_filter,
        strict=strict,
    )
    if result is None and debug:
        cli_debug(args)
    else:
        print(result)


def cli_eval_filter(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    target: str = args.input
    disable_filter = bool(args.disable_filter)
    debug = bool(args.debug)
    strict = not bool(args.nostrict)
    extern_feat = None if args.extern_feat is None else str(args.extern_feat)
    result = model_loaded.est_filter(
        data=target,
        extern_feat=extern_feat,
        disable_filter=disable_filter,
        strict=strict,
    )
    if not result and debug:
        cli_debug(args)
    else:
        print(int(result))


def cli_eval_predict(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    target: str = args.input
    disable_filter = bool(args.disable_filter)
    strict = not bool(args.nostrict)
    quantity = str(args.quantity)
    debug = bool(args.debug)
    extern_feat = None if args.extern_feat is None else str(args.extern_feat)
    result = model_loaded.calc_est(
        data=target,
        prop=quantity,
        extern_feat=extern_feat,
        disable_filter=disable_filter,
        strict=strict,
    )
    if result is None and debug:
        cli_debug(args)
    else:
        print(result)


def cli_eval_label(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    target: str = args.input
    note_unmapped = bool(args.note_unmapped)
    if note_unmapped:
        model_labels = frozenset(model_loaded.get_labels())
        model_features = frozenset(model_loaded.get_features())
    else:
        model_labels = frozenset()
        model_features = frozenset()
    any_output = False
    for i, (weight, ugraph, extras) in enumerate(
        model_loaded.get_ugraphs(target)
    ):
        any_output = True
        print(f"Labeled Graph #{i}; weight = {weight}")
        if ugraph is None and len(extras) == 0:
            print("  String parse failure!! No graph or features detected!!")
            continue

        print("  Edges:")
        if ugraph is None:
            print("    No graph detected.")
        else:
            for bond in ugraph.get_bonds():
                print(f"    {bond.src} - {bond.trg}")

        print("  Labels:")
        if ugraph is None:
            print("    No graph detected.")
        else:
            for label in ugraph.get_labels():
                label_note = f"label({label})"
                print("    ", end="")
                if note_unmapped and label_note not in model_labels:
                    print("* ", end="")
                print(label_note)

        print("  Features:")
        if len(extras) == 0:
            print("    No extra features detected.")
        else:
            for feature in extras:
                print("    ", end="")
                if note_unmapped and feature not in model_features:
                    print("* ", end="")
                print(feature)

    if not any_output:
        print("String parse failure!! No graphs or features detected!!")


def cli_debug(args: argparse.Namespace) -> None:
    model_loaded = load_model(args.model)
    target: str = args.input
    no_err = True
    extern_feat = None if args.extern_feat is None else str(args.extern_feat)
    print(f'Started debugging string "{target}"...')
    for errmsg in model_loaded.debug(target, extern_feat=extern_feat):
        no_err = False
        print(errmsg)
    if no_err:
        print("No issues parsing this input.")


def add_eval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "model",
        type=pathlib.Path,
        help="Path of model file.",
    )
    parser.add_argument(
        "input", type=str, help="Input string for model evaluation."
    )
    parser.add_argument(
        "--extern-feat",
        default=None,
        type=str,
        help="External features in `extern_feat` string form.",
    )
    parser.add_argument(
        "--disable-filter",
        action="store_true",
        default=False,
        help="Disable filter step (covariance filter only).",
    )
    parser.add_argument(
        "--nostrict",
        action="store_true",
        default=False,
        help=(
            "Remove strictness checks (e.g. when multiple colorings "
            "are possible for the same SMILES due to ambiguity, and one of the "
            "the colorings is filtered out, aggregate values for one which "
            "passed the filter)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Instead of printing `None`, when a value fails to be calculated, "
            "print the result of `sgrcli debug`"
        ),
    )


def build_argparser() -> argparse.ArgumentParser:
    # top-level parser
    parser = argparse.ArgumentParser(
        prog="sgrcli",
        description="General interface for SGR-Py library.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    subparsers = parser.add_subparsers(required=True)

    # training subparser
    parser_train = subparsers.add_parser(
        "train", help="Train a model from observed data."
    )
    parser_train.set_defaults(func=cli_train)
    parser_train.add_argument(
        "data",
        type=pathlib.Path,
        help="Path of input data file.",
    )
    parser_train.add_argument(
        "outfile",
        type=pathlib.Path,
        help="Destination of model save file.",
    )
    parser_train.add_argument(
        "--data-output",
        type=pathlib.Path,
        help="Optional output of full training DataFrame.",
    )
    parser_train.add_argument(
        "--disable-covar",
        action="store_true",
        help=(
            "Deactivate covariance filter (not recommended if training a "
            "linear model)."
        ),
    )
    parser_train_input = parser_train.add_argument_group(
        "Input", "Arguments describing format of input"
    )
    parser_train_input.add_argument(
        "--mode",
        choices=["raw", "smiles-mol", "smiles-rxn"],
        type=str,
        default="smiles-mol",
        help=(
            'Input type (default: "smiles-mol"). Value of "raw" indicates raw '
            'UGraph strings, "smiles-mol" indicates molecules in SMILES form, '
            '"smiles-rxn" indicates reactions in reaction SMILES form. This '
            "value must correspond with any models provided."
        ),
    )
    parser_train_input.add_argument(
        "--format",
        choices=["csv", "tsv", "parquet"],
        default=None,
        help=(
            "Format of input data (default: inferred from file suffix). Must "
            "have column headers."
        ),
    )
    parser_train_input.add_argument(
        "--data-column",
        default="smiles",
        type=str,
        help='Which column contains relevant input string (default: "smiles").',
    )
    parser_train_input.add_argument(
        "--group-column",
        default=None,
        type=str,
        help="Column containing groups, for counting purposes (default: None).",
    )
    parser_train_input.add_argument(
        "--weight-column",
        default=None,
        type=str,
        help="Column containing datapoint weights (default: None).",
    )
    parser_train_input.add_argument(
        "--target-column",
        action="append",
        default=None,
        type=str,
        help="Column containing values to train predictor (default: None). "
        "Can be repeated for additional targets.",
    )
    parser_train_input.add_argument(
        "--extra-feat-column",
        default=None,
        type=str,
        help="Column containing extra features in JSON form (default: None).",
    )
    parser_train_encode = parser_train.add_argument_group(
        "Encoding", "Arguments describing encoding parameters"
    )
    parser_train_encode.add_argument(
        "--stereo-pi",
        default=True,
        type=bool,
        help=(
            "Whether to add cis- nodes across double bonds to account for "
            "cis/trans stereochemistry (default: True).  Only applies to "
            '"rdkit-mol" and "rdkit-rxn" operation modes.'
        ),
    )
    parser_train_encode.add_argument(
        "--extra-ring",
        action=argparse.BooleanOptionalAction,
        help=(
            "Include local ring information in atom encoding (default: on). "
            'Only applies to "rdkit-mol" and "rdkit-rxn" operation modes.'
        ),
    )
    parser_train_encode.add_argument(
        "--extra-conj",
        action=argparse.BooleanOptionalAction,
        help=(
            "Include local conjugation information in atom encoding "
            '(default: on).  Only applies to "rdkit-mol" and "rdkit-rxn" '
            "operation modes."
        ),
    )
    parser_train_encode.add_argument(
        "--feat-ring",
        choices=["core", "extended"],
        default=None,
        type=str,
        help=(
            "Include enumeration of ring features (default: None). "
            'Only applies to "rdkit-mol" and "rdkit-rxn" operation modes.'
        ),
    )
    parser_train_encode.add_argument(
        "--feat-conj",
        choices=["core", "extended"],
        default=None,
        type=str,
        help=(
            "Include enumeration of conjugation features (default: None). "
            'Only applies to "rdkit-mol" and "rdkit-rxn" operation modes.'
        ),
    )
    parser_train_encode.add_argument(
        "--trim",
        default=True,
        type=bool,
        help=(
            "Trim atom-center groups for terminal heavy atoms (default: True). "
            'Only applies to "rdkit-mol" and "rdkit-rxn" operation modes.'
        ),
    )
    parser_train_encode.add_argument(
        "--min-count",
        default=1,
        type=int,
        help=(
            "Minimum number of datapoint groups each derived group must be "
            "present in (default: 1).  Recommended value for similarity models "
            "is 1, for regression models 2."
        ),
    )
    parser_train_partition = parser_train.add_argument_group(
        "Partitioning", "Arguments describing partition parameters"
    )
    parser_train_partition.add_argument(
        "--enable-part",
        action="store_true",
        help="Enable partitioning of molecule.",
    )
    parser_train_partition.add_argument(
        "--max-ssize",
        default=1,
        type=int,
        help=(
            "Maximum size of seed groups for fragmentation (default: 1). "
            "Value of 1 indicates size-1 groups only (no partitioning). "
            "Zero or negative values indicate no limit. "
            "When large, symmetric molecules (ex. buckminsterfullerene) are "
            "present, it is recommended this be set to a reasonable value."
        ),
    )
    parser_train_partition.add_argument(
        "--slimit",
        default=None,
        type=int,
        help=(
            "Cutoff for total seed groups when max-ssize is greater than 1 "
            "(default: None).  Recommended value is a multiple of dataset size."
        ),
    )
    parser_train_partition.add_argument(
        "--train-iter",
        default=5,
        type=int,
        help=("Number of training iterations per pruning call (default: 5)."),
    )
    parser_train_partition.add_argument(
        "--prune-frac",
        default=0.2,
        type=int,
        help=(
            "Proportion of fragments to trim during each pruning call "
            "(default: 0.2)."
        ),
    )
    parser_train_partition.add_argument(
        "--plimit",
        default=64,
        type=int,
        help=(
            "Maximum number of partitions to generate (default: 64).  If "
            "negative or zero, no limit."
        ),
    )
    parser_train_partition.add_argument(
        "--max-iter",
        default=65536,
        type=int,
        help=(
            "Maximum number of partition iterations (default: 65536).  If "
            "negative or zero, no limit.  When this is the bottleneck, runtime "
            "is roughly proportional to this value."
        ),
    )
    parser_train_partition.add_argument(
        "--max-heap",
        default=65536,
        type=int,
        help=(
            "Maximum number of partition iterations to be cached "
            "(default: 65536).  If negative or zero, no limit.  When this is "
            "the bottleneck, memory usage is roughly proportional to this "
            "value."
        ),
    )
    parser_train_partition.add_argument(
        "--wllimit",
        default=-100.0,
        type=float,
        help=(
            "Lower limit of logweight for fragments (default: -100.0).  Used "
            "to prevent `-inf` values from making certain fragment weights "
            "unrecoverable."
        ),
    )
    parser_train_partition.add_argument(
        "--num-cores",
        default=None,
        type=int,
        help="Number of cores to use for partitioning (default: all).",
    )
    parser_train_partition.add_argument(
        "--progress",
        action="store_true",
        help="Visualize progress of partitioning on command line.",
    )
    parser_train_linear = parser_train.add_argument_group(
        "Linear",
        "Arguments describing covariance and linear regression parameters",
    )
    parser_train_linear.add_argument(
        "--min-rel-weight",
        default=_NP_SQRT_F64EPS,
        help=(
            "Minimum relative weight to include during linear fitting "
            "(default: sqrt(f64.eps))."
        ),
        type=float,
    )
    parser_train_linear.add_argument(
        "--atol",
        default=0.0,
        help=(
            "Absolute tolerance argument to `scipy.sparse.linalg.lsqr` for "
            "fitting routine (default: 0.0)."
        ),
        type=float,
    )
    parser_train_linear.add_argument(
        "--btol",
        default=0.0,
        help=(
            "Relative tolerance argument to `scipy.sparse.linalg.lsqr` for "
            "fitting routine (default: 0.0)."
        ),
        type=float,
    )
    parser_train_linear.add_argument(
        "--conlim",
        default=0.0,
        help=(
            "Condition number argument to `scipy.sparse.linalg.lsqr` for "
            "fitting routine (default: 0.0)."
        ),
        type=float,
    )
    parser_train_linear.add_argument(
        "--fit-iter-lim",
        default=10000,
        help=(
            "Maximum iterations of `scipy.sparse.linalg.lsqr` for "
            "fitting routine (default: 10000)."
        ),
        type=int,
    )

    # mapping subparser
    parser_map = subparsers.add_parser(
        "map", help="Use trained model to map many datapoints."
    )
    parser_map.set_defaults(func=cli_map)
    parser_map.add_argument(
        "model",
        type=pathlib.Path,
        help="Path of model file.",
    )
    parser_map.add_argument(
        "datafile", type=pathlib.Path, help="Path of input data file."
    )
    parser_map.add_argument(
        "outfile", type=pathlib.Path, help="Destination of output data file."
    )
    parser_map.add_argument(
        "--format",
        choices=["csv", "tsv", "parquet"],
        default=None,
        help=(
            "Format of input data (default: inferred from file suffix). Must "
            "have column headers."
        ),
    )
    parser_map.add_argument(
        "--out-format",
        choices=["csv", "tsv", "parquet"],
        default=None,
        help=(
            "Format of output data (default: inferred from file suffix). Must "
            "have column headers."
        ),
    )
    parser_map.add_argument(
        "--data-column",
        default="data",
        type=str,
        help='Which column contains relevant input string (default: "data").',
    )
    parser_map.add_argument(
        "--group-column",
        default=None,
        type=str,
        help="Column containing groups, for reference (default: None).",
    )
    parser_map.add_argument(
        "--weight-column",
        default=None,
        type=str,
        help="Column containing datapoint weights (default: None).",
    )
    parser_map.add_argument(
        "--extra-feat-column",
        default=None,
        type=str,
        help="Column containing extra features in JSON form (default: None).",
    )
    parser_map.add_argument(
        "--target-column",
        default=None,
        action="append",
        type=str,
        help="Column containing values to predict (default: None). Can be "
        "repeated.",
    )

    # eval subparser
    parser_eval = subparsers.add_parser(
        "eval", help="Use trained model to evaluate a single datapoint."
    )
    subparsers_eval = parser_eval.add_subparsers(required=True)
    parser_eval_label = subparsers_eval.add_parser(
        "label", help="Get graph labels of datapoint."
    )
    parser_eval_label.set_defaults(func=cli_eval_label)
    parser_eval_label.add_argument(
        "model",
        type=pathlib.Path,
        help="Path of model file.",
    )
    parser_eval_label.add_argument(
        "input", type=str, help="Input string for model evaluation."
    )
    parser_eval_label.add_argument(
        "--note-unmapped",
        action="store_true",
        help=(
            "Add an asterisk (*) in front of labels with no valid mapping "
            "index within the given model (default: all labels listed)."
        ),
    )
    parser_eval_filter = subparsers_eval.add_parser(
        "filter", help="Determine whether datapoint will be filtered by model."
    )
    parser_eval_filter.set_defaults(func=cli_eval_filter)
    add_eval_options(parser_eval_filter)
    parser_eval_info = subparsers_eval.add_parser(
        "info",
        help="Calculate information of datapoint (units of negative nats).",
    )
    parser_eval_info.set_defaults(func=cli_eval_info)
    add_eval_options(parser_eval_info)
    parser_eval_predict = subparsers_eval.add_parser(
        "predict", help="Predict some quantity of the datapoint."
    )
    parser_eval_predict.set_defaults(func=cli_eval_predict)
    add_eval_options(parser_eval_predict)
    parser_eval_predict.add_argument(
        "quantity", type=str, help="Quantity to predict."
    )

    # debugging subparser
    parser_debug = subparsers.add_parser(
        "debug",
        help="Explore why a molecule might have been rejected by a model.",
    )
    parser_debug.set_defaults(func=cli_debug)
    parser_debug.add_argument(
        "model",
        type=pathlib.Path,
        help="Path of model file.",
    )
    parser_debug.add_argument(
        "input", type=str, help="Input string for model evaluation."
    )
    parser_debug.add_argument(
        "--extern-feat",
        default=None,
        type=str,
        help="External features in `extern_feat` string form.",
    )

    return parser


def main(sys_args: list[str] | None = None) -> None:
    if sys_args is None:
        sys_args = sys.argv[1:]
    parser = build_argparser()
    args = parser.parse_args(sys_args)

    arg_func: Callable[[argparse.Namespace], None] = args.func
    arg_func(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
