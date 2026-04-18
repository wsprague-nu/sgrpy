"""Compatibility function replacing `convert_benson`."""

from collections.abc import Iterable

import rdkit.Chem.rdchem

from sgrpy.graph import Bond, CanonType, TGraph

from ._benson_v3 import AtomGraph, FeatureFormat
from ._benson_v3 import ConvertResult as ConvertResult_v3
from ._benson_v4 import (
    AtomCore,
    AtomLabel,
    BensonCore,
    BensonLabel,
    BondExtension,
    BondFeatures,
    BondType,
    EdgeData,
    HybridType,
    map_benson_v4,
)
from ._structures import AtomCore as AtomCore_v3
from ._structures import AtomLabel as AtomLabel_v3
from ._structures import BondExtension as BondExtension_v3
from ._structures import BondFeatures as BondFeatures_v3
from ._structures import BondType as BondType_v3
from ._structures import CisLabel as CisLabel_v3
from ._structures import Hybridization as HybridType_v3


def __convert_hyb(hybrid: HybridType) -> HybridType_v3:
    match hybrid:
        case HybridType.S:
            return HybridType_v3.S
        case HybridType.SP:
            return HybridType_v3.SP
        case HybridType.SP2:
            return HybridType_v3.SP2
        case HybridType.SP3:
            return HybridType_v3.SP3
        case HybridType.SP3D:
            return HybridType_v3.SP3D
        case HybridType.SP3D2:
            return HybridType_v3.SP3D2
        case _:
            raise ValueError(f"Unimplemented hybridization {hybrid}")


def __convert_core(core: AtomCore) -> AtomCore_v3:
    return AtomCore_v3(
        p=core.p,
        v=core.v,
        d=core.d,
        c=core.c,
        rad=core.rad,
        hyb=__convert_hyb(core.hyb),
        a=core.a,
        n=core.n,
    )


def __convert_bt(bt: BondType) -> BondType_v3:
    match bt:
        case BondType.SINGLE:
            return BondType_v3.SINGLE
        case BondType.DOUBLE:
            return BondType_v3.DOUBLE
        case BondType.TRIPLE:
            return BondType_v3.TRIPLE
        case BondType.AROMATIC:
            return BondType_v3.AROMATIC
        case _:
            raise ValueError(f"Unimplemented bond type {bt}")


def __convert_bf(feat: BondFeatures) -> BondFeatures_v3:
    bt = BondFeatures_v3(bond_type=__convert_bt(feat.bond_type))
    return bt


def __convert_ext(ext: BondExtension) -> BondExtension_v3:
    core = __convert_core(ext.trg)
    bf = __convert_bf(ext.bond)
    return BondExtension_v3(bf, core)


def __convert_label(label: BensonLabel) -> AtomLabel_v3 | CisLabel_v3:
    main_label = label.label
    if isinstance(main_label, AtomLabel):
        new_core = __convert_core(main_label.invariants)
        new_neighbors = sorted(__convert_ext(n) for n in main_label._neighbors)
        new_label = AtomLabel_v3(
            invariants=new_core,
            h=main_label.h,
            _neighbors=tuple(new_neighbors),
            _extra=main_label._extra,
        )
        return new_label
    atom1 = __convert_core(main_label._atom1)
    atom2 = __convert_core(main_label._atom2)
    atom3 = __convert_core(main_label._atom3)
    atom4 = __convert_core(main_label._atom4)
    orient_1 = (atom1, atom2, atom3, atom4)
    orient_2 = (atom4, atom3, atom2, atom1)
    orient_1 = min(orient_1, orient_2)
    return CisLabel_v3(
        _atom1=orient_1[0],
        _atom2=orient_1[1],
        _atom3=orient_1[2],
        _atom4=orient_1[3],
        _extra=main_label._extra,
    )


def _convert_vert_label(
    label: BensonCore | BensonLabel,
) -> str:
    if isinstance(label, BensonCore):
        return label.as_str()
    return label.label.as_str()


def _convert_feat_graph(
    feature: TGraph[BensonCore, EdgeData] | TGraph[BensonLabel, EdgeData],
) -> str:
    canonical_feature, _ = feature.canonize(CanonType.F)
    vertex_label = ",".join(
        _convert_vert_label(label) for label in canonical_feature.get_labels()
    )
    edge_label = ",".join(
        f"{bond.src},{bond.trg},{bond.color.as_str()}"
        for bond in canonical_feature.get_bonds()
    )
    return ";".join((edge_label, vertex_label))


def map_benson_v4_compat_v3(
    mol: rdkit.Chem.rdchem.Mol,
    trim_ends: bool = True,
    node_cis: bool = False,
    ext_ring: bool = False,
    ext_conj: bool = False,
    feat_ring: FeatureFormat | int = FeatureFormat.NONE,
    feat_conj: FeatureFormat | int = FeatureFormat.NONE,
) -> Iterable[ConvertResult_v3]:
    """Convert RDKit molecule to pre-UGraph form plus optional annotations.

    If this operation fails, no results will be returned.

    Parameters
    ----------
    mol : Mol
        RDKit molecule to be converted.
    trim_ends : bool
        Trim terminal atoms, excluding diatomic cases (default: True).
    node_cis : bool
        Add nodes representing cis relationships (default: False).  In case of
        ambiguity, multiple results will be returned.
    ext_ring : bool
        Include extended ring information in node labels (default: False).
    ext_conj : bool
        Include extended conjugation information in node labels (default:
        False).
    feat_ring : FeatureFormat
        Include ring feature information in `extras` JSON (default: NONE).
    feat_conj : FeatureFormat
        Include conjugation feature information in `extras` JSON (default:
        NONE).

    Returns
    -------
    Iterable[ConvertResult_v3]
        Iterable of results.  Length zero if no results obtained.
    """
    for result_v4 in map_benson_v4(
        mol=mol,
        node_cis=node_cis,
        trim_ends=trim_ends,
        ext_ring=ext_ring,
        ext_conj=ext_conj,
        feat_ring=int(feat_ring),
        feat_conj=int(feat_conj),
    ):
        tgraph = result_v4.graph
        all_nodes = tuple(__convert_label(v) for v in tgraph.get_labels())
        bonds = tuple(Bond.new(b.src, b.trg) for b in tgraph.get_bonds())
        extra_dict: dict[str, list[str]] | None = {
            key: [_convert_feat_graph(g) for g in value]
            for key, value in result_v4.extras.items()
        }
        if extra_dict is not None and len(extra_dict) == 0:
            extra_dict = None
        ag = AtomGraph(nodes=all_nodes, bonds=bonds)
        result = ConvertResult_v3(graph=ag, extras=extra_dict)
        yield result
