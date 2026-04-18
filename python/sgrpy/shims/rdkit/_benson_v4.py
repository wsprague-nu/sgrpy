"""RDKit to TGraph conversion module."""

import dataclasses
import enum
import itertools
from collections.abc import Iterable
from typing import final

import igraph
import rdkit.Chem.rdchem

from sgrpy.graph import CanonType, IndexMapping, TBond, TGraph
from sgrpy.iotypes import CompareHash

__all__ = [
    "AtomCore",
    "AtomLabel",
    "BensonLabel",
    "BondExtension",
    "BondFeatures",
    "BondStereoError",
    "BondType",
    "BondTypeError",
    "CisLabel",
    "ConvertResult_v4",
    "EdgeData",
    "HybridType",
    "HybridizationError",
    "ParsingError",
    "VertexData",
    "map_benson_core_v4",
    "map_benson_v4",
    "map_core_rdkit_v4",
]


class ParsingError(Exception):
    """Raised when error with parsing RDKit molecule."""


@final
class HybridizationError(ParsingError):
    """Raised when unknown hybridization type is passed."""


@final
class HybridType(enum.IntEnum):
    """Type of hybridization (adapted directly from RDKit)."""

    S = enum.auto()
    """S (spherical) hybridization.  Unsure if this is ever used."""
    SP = enum.auto()
    """SP (linear) hybridization."""
    SP2 = enum.auto()
    """SP2 (trigonal planar) hybridization."""
    SP3 = enum.auto()
    """SP3 (tetrahedral) hybridization."""
    SP2D = enum.auto()
    """SP2D (square planaer) hybridization."""
    SP3D = enum.auto()
    """SP3D (trigonal bipyramidal/square pyramidal) hybridization."""
    SP3D2 = enum.auto()
    """SP3D2 (octahedral) hybridization."""

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "HybridType":
        rdkit_hybrid: rdkit.Chem.rdchem.HybridizationType = (
            atom.GetHybridization()
        )
        match rdkit_hybrid:
            case rdkit.Chem.rdchem.HybridizationType.S:
                return HybridType.S
            case rdkit.Chem.rdchem.HybridizationType.SP:
                return HybridType.SP
            case rdkit.Chem.rdchem.HybridizationType.SP2:
                return HybridType.SP2
            case rdkit.Chem.rdchem.HybridizationType.SP3:
                return HybridType.SP3
            case rdkit.Chem.rdchem.HybridizationType.SP3D:
                return HybridType.SP3D
            case rdkit.Chem.rdchem.HybridizationType.SP3D2:
                return HybridType.SP3D2
            case _:
                raise HybridizationError(
                    f"Unimplemented hybridization: {atom.GetHybridization()}"
                )


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AtomCore:
    """Atom core invariants.

    Attributes
    ----------
    p : int
        Proton number.
    v : int
        Valence number.
    d : int
        Total degree.
    c : int
        Formal charge.
    rad : int
        Number of radical electrons.
    hyb : HybridType
        Hybridization of atom.
    a : bool
        Whether atom is aromatic.
    n : int
        Isotope number (0 indicates averaged).
    """

    p: int
    v: int
    d: int
    c: int
    rad: int
    hyb: HybridType
    a: bool
    n: int

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "AtomCore":
        p: int = atom.GetAtomicNum()
        v: int = atom.GetTotalValence()
        d: int = atom.GetTotalDegree()
        c: int = atom.GetFormalCharge()
        rad: int = atom.GetNumRadicalElectrons()
        a: bool = atom.GetIsAromatic()
        n: int = atom.GetIsotope()
        hyb = HybridType.from_atom(atom)
        return AtomCore(p=p, v=v, d=d, c=c, rad=rad, hyb=hyb, a=a, n=n)

    def as_str(self) -> str:
        return "(" + ",".join(str(x) for x in dataclasses.astuple(self)) + ")"


@final
class BondTypeError(ParsingError):
    """Raised when unknown bond type is passed."""


@final
class BondType(enum.IntEnum):
    """Type of bond (adapted directly from RDKit)."""

    SINGLE = enum.auto()
    """Single bond."""
    DOUBLE = enum.auto()
    """Double bond."""
    TRIPLE = enum.auto()
    """Triple bond."""
    AROMATIC = enum.auto()
    """Aromatic bond."""

    @classmethod
    def from_bond(cls, bond: rdkit.Chem.rdchem.Bond) -> "BondType":
        bond_type: rdkit.Chem.rdchem.BondType = bond.GetBondType()
        match bond_type:
            case rdkit.Chem.rdchem.BondType.SINGLE:
                return BondType.SINGLE
            case rdkit.Chem.rdchem.BondType.DOUBLE:
                return BondType.DOUBLE
            case rdkit.Chem.rdchem.BondType.TRIPLE:
                return BondType.TRIPLE
            case rdkit.Chem.rdchem.BondType.AROMATIC:
                return BondType.AROMATIC
            case _:
                raise BondTypeError(f"Unimplemented bond type: {bond_type}")


@final
class BondStereoError(ParsingError):
    """Raised when unrecognized bond stereochemistry is passed."""


@final
class BondStereo(enum.IntEnum):
    """Bond stereochemistry specifier."""

    STEREOE = enum.auto()
    """E (entgegen) type specifier."""
    STEREOZ = enum.auto()
    """Z (zusammen) type specifier."""
    STEREONONE = enum.auto()
    """Bond is symmetrical, no E/Z specification possible."""
    STEREOANY = enum.auto()
    """Bond is ambiguously specified."""

    @classmethod
    def from_bond(cls, bond: rdkit.Chem.rdchem.Bond) -> "BondStereo":
        bond_stereo: rdkit.Chem.rdchem.BondStereo = bond.GetStereo()
        match bond_stereo:
            case rdkit.Chem.rdchem.BondStereo.STEREOE:
                return BondStereo.STEREOE
            case rdkit.Chem.rdchem.BondStereo.STEREOZ:
                return BondStereo.STEREOZ
            case rdkit.Chem.rdchem.BondStereo.STEREONONE:
                return BondStereo.STEREONONE
            case rdkit.Chem.rdchem.BondStereo.STEREOANY:
                return BondStereo.STEREOANY
            case _:
                raise BondStereoError(
                    f"Unimplemented bond stereo {bond.GetStereo()}"
                )


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class BondFeatures:
    """Bond invariant dataclass.

    Attributes
    ----------
    bond_type : BondType
        Type of bond.
    in_ring : bool | None
        Whether bond is in ring.  None if ring information not relevant.
    is_conj : bool | None
        Whether bond is conjugated.  None if conjugation information not
        relevant.
    """

    bond_type: BondType
    in_ring: bool | None
    is_conj: bool | None

    @classmethod
    def from_bond(
        cls, bond: rdkit.Chem.rdchem.Bond, ring_info: bool, conj_info: bool
    ) -> "BondFeatures":
        bond_type = BondType.from_bond(bond)
        in_ring: bool | None = bond.IsInRing() if ring_info else None
        is_conj: bool | None = bond.GetIsConjugated() if conj_info else None
        return BondFeatures(
            bond_type=bond_type, in_ring=in_ring, is_conj=is_conj
        )

    def as_str(self) -> str:
        bond_part = f"{str(self.bond_type)}"
        if self.in_ring is True:
            bond_part = bond_part + "R"
        if self.is_conj is True:
            bond_part = bond_part + "C"
        return bond_part


@final
@dataclasses.dataclass(frozen=True, slots=True)
class BensonCore:
    atomcore: AtomCore | None

    def as_str(self) -> str:
        return "cis" if self.atomcore is None else self.atomcore.as_str()

    def __lt__(self, __value: "BensonCore") -> bool:
        match (self.atomcore is None, __value.atomcore is None):
            case True, _:
                return False
            case False, True:
                return True
            case False, False:
                return self.atomcore < __value.atomcore  # type: ignore[operator]
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, slots=True)
class VertexData:
    atomcore: AtomCore | None
    h: int

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "VertexData":
        atomcore = AtomCore.from_atom(atom)
        h: int = atom.GetTotalNumHs()
        return VertexData(atomcore, h)

    @classmethod
    def new_cis(cls) -> "VertexData":
        return VertexData(None, 0)

    def to_core(self) -> BensonCore:
        return BensonCore(self.atomcore)

    def __lt__(self, __value: "VertexData") -> bool:
        match (self.atomcore is None, __value.atomcore is None):
            case True, True:
                return self.h < __value.h
            case True, False:
                return False
            case False, True:
                return True
            case False, False:
                return (self.atomcore, self.h) < (__value.atomcore, __value.h)
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, slots=True)
class EdgeData:
    bondcore: BondFeatures | bool
    # True if cis-to-double, False if cis-to-neighbor

    @classmethod
    def from_bond(
        cls, bond: rdkit.Chem.rdchem.Bond, ring_info: bool, conj_info: bool
    ) -> "EdgeData":
        self_label = BondFeatures.from_bond(bond, ring_info, conj_info)

        return EdgeData(bondcore=self_label)

    @classmethod
    def new_cis(cls, is_double: bool) -> "EdgeData":
        return EdgeData(bondcore=is_double)

    def clear_ring(self) -> "EdgeData":
        if isinstance(self.bondcore, bool) or self.bondcore.in_ring is None:
            return self
        newcore = dataclasses.replace(self.bondcore, in_ring=None)
        retval = EdgeData(newcore)
        return retval

    def clear_conj(self) -> "EdgeData":
        if isinstance(self.bondcore, bool) or self.bondcore.is_conj is None:
            return self
        newcore = dataclasses.replace(self.bondcore, is_conj=None)
        retval = EdgeData(newcore)
        return retval

    def as_str(self) -> str:
        if isinstance(self.bondcore, bool):
            return "CT" if self.bondcore is True else "CF"
        return self.bondcore.as_str()

    def __lt__(self, __value: "EdgeData") -> bool:
        match (
            isinstance(self.bondcore, bool),
            isinstance(__value.bondcore, bool),
        ):
            case True, True:
                return self.bondcore < __value.bondcore  # type: ignore[operator]
            case True, False:
                return False
            case False, True:
                return True
            case False, False:
                return self.bondcore < __value.bondcore  # type: ignore[operator]
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, slots=True)
class VertexSelect:
    select: BensonCore | None

    def as_str(self) -> str:
        if self.select is None:
            return "*"
        return self.select.as_str()

    def __lt__(self, __value: "VertexSelect") -> bool:
        match (self.select is None, __value.select is None):
            case _, True:
                return False
            case True, False:
                return True
            case False, False:
                return self.select < __value.select  # type: ignore[operator]
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, slots=True)
class _CisLabelIdx:
    """Cis label index mappings."""

    start_n: int
    start_b: int
    end_b: int
    end_n: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.start_n, self.start_b, self.end_b, self.end_n)

    # def as_tuple_sort(self) -> tuple[int, int, int, int]:
    #     self_tup = self.as_tuple()
    #     rev_tup = (self_tup[3], self_tup[2], self_tup[1], self_tup[0])
    #     return min(self_tup, rev_tup)


def __check_bond_ring_lt_8(bond: rdkit.Chem.rdchem.Bond) -> bool:
    if not bond.IsInRing():
        return False
    return any(bond.IsInRingSize(i) for i in range(8))


def _get_cis_labels_int_spec(
    bond: rdkit.Chem.rdchem.Bond,
) -> tuple[tuple[_CisLabelIdx, ...], ...] | None:
    # get bond type
    bond_type = BondType.from_bond(bond)

    # reject non-pi bonds (others not implemented)
    if bond_type != BondType.DOUBLE:
        return None

    # get start atom indices
    start_atom: rdkit.Chem.rdchem.Atom = bond.GetBeginAtom()
    end_atom: rdkit.Chem.rdchem.Atom = bond.GetEndAtom()
    start_idx: int = start_atom.GetIdx()
    end_idx: int = end_atom.GetIdx()
    start_neigh: tuple[int, ...] = tuple(
        a.GetIdx() for a in start_atom.GetNeighbors() if a.GetIdx() != end_idx
    )
    end_neigh: tuple[int, ...] = tuple(
        a.GetIdx() for a in end_atom.GetNeighbors() if a.GetIdx() != start_idx
    )

    # if either end is terminal, no cis can be assigned
    if (
        len(start_neigh) == 0
        or len(end_neigh) == 0
        or start_atom.GetHybridization()
        == rdkit.Chem.rdchem.HybridizationType.SP
        or end_atom.GetHybridization() == rdkit.Chem.rdchem.HybridizationType.SP
        # or __check_bond_ring_lt_8(bond)
        # or __check_radical_error(start_atom)
        # or __check_radical_error(end_atom)
    ):
        return None

    # case where either is greater than 2
    if len(start_neigh) > 2 or len(end_neigh) > 2:  # noqa: PLR2004
        # complex double bonds not handled
        return None

    stereo_type = BondStereo.from_bond(bond)

    # reject ring stereo if below a certain size, or ambiguous above a size
    if bond.IsInRing():
        match stereo_type, __check_bond_ring_lt_8(bond):
            case BondStereo.STEREONONE | BondStereo.STEREOANY, True:
                return None
            case BondStereo.STEREOE | BondStereo.STEREOZ, True:
                raise BondStereoError(
                    "Stereo specification for ring <8 atoms currently invalid"
                )
            case BondStereo.STEREONONE | BondStereo.STEREOANY, False:
                raise BondStereoError(
                    "Disambiguation of stereochemical specifiers for large "
                    "rings not currently supported"
                )
            case BondStereo.STEREOZ | BondStereo.STEREOE, False:
                pass
            case _:  # this should be unreachable
                raise BondStereoError("Invalid configuration")

    # make sure end neighbors are shorter in length
    # leaves start=1, end=1; start=2, end=1; start=2, end=2 cases
    if len(start_neigh) < len(end_neigh):
        temp_neigh = start_neigh
        start_neigh = end_neigh
        end_neigh = temp_neigh

    stereo_atoms: tuple[int, int]
    match len(start_neigh), len(end_neigh), stereo_type:
        # start=1, end=1; trans bond therefore no cis node
        case 1, _, BondStereo.STEREOE:
            return None
        # start=1, end=1; cis bond therefore a single cis node
        case 1, _, BondStereo.STEREOZ:
            relation_cis = _CisLabelIdx(
                start_neigh[0], start_idx, end_idx, end_neigh[0]
            )
            return ((relation_cis,),)
        # start=1, end=1; undefined stereo so return both possibilities
        case 1, _, BondStereo.STEREONONE | BondStereo.STEREOANY:
            relation_trans = ()
            relation_cis = _CisLabelIdx(
                start_neigh[0], start_idx, end_idx, end_neigh[0]
            )
            return (relation_trans, (relation_cis,))
        # start=2, end=1; STEREOZ maps the cis atoms so this is easy
        case 2, 1, BondStereo.STEREOZ:
            stereo_atoms = tuple(bond.GetStereoAtoms())  # type: ignore[assignment,unused-ignore]
            cis_start = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end = end_neigh[0]
            relation_cis = _CisLabelIdx(cis_start, start_idx, end_idx, cis_end)
            return ((relation_cis,),)
        # start=2, end=1; STEREOE maps the trans atoms so a swap is needed
        case 2, 1, BondStereo.STEREOE:
            stereo_atoms = tuple(bond.GetStereoAtoms())  # type: ignore[assignment,unused-ignore]
            cis_start = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end = end_neigh[0]
            relation_cis = _CisLabelIdx(cis_start, start_idx, end_idx, cis_end)
            return ((relation_cis,),)
        # start=2, end=1; undefined stereo so return both possibilities
        case 2, 1, BondStereo.STEREONONE | BondStereo.STEREOANY:
            cis_start1 = start_neigh[0]
            cis_start2 = start_neigh[1]  # type: ignore[unused-ignore]
            cis_end1 = end_neigh[0]
            rel1 = _CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = _CisLabelIdx(cis_start2, start_idx, end_idx, cis_end1)
            return ((rel1,), (rel2,))
        # start=2, end=2; stereo atoms define cis
        case 2, 2, BondStereo.STEREOZ:
            stereo_atoms = bond.GetStereoAtoms()  # type: ignore[assignment,unused-ignore]
            cis_start1 = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end1 = [sn for sn in end_neigh if sn in stereo_atoms][0]
            cis_start2 = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end2 = [sn for sn in end_neigh if sn not in stereo_atoms][0]
            rel1 = _CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = _CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel1, rel2),)
        # start=2, end=2; stereo atoms define trans
        case 2, 2, BondStereo.STEREOE:
            stereo_atoms = bond.GetStereoAtoms()  # type: ignore[assignment,unused-ignore]
            cis_start1 = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end1 = [sn for sn in end_neigh if sn not in stereo_atoms][0]
            cis_start2 = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end2 = [sn for sn in end_neigh if sn in stereo_atoms][0]
            rel1 = _CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = _CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel1, rel2),)
        # start=2, end=2; stereo atoms undefined so return both possibilities
        case 2, 2, BondStereo.STEREONONE | BondStereo.STEREOANY:
            cis_start1 = start_neigh[0]
            cis_end1 = end_neigh[0]
            cis_end2 = end_neigh[1]  # type: ignore[unused-ignore]
            cis_start2 = start_neigh[1]  # type: ignore[unused-ignore]
            rel11 = _CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel12 = _CisLabelIdx(cis_start1, start_idx, end_idx, cis_end2)
            rel21 = _CisLabelIdx(cis_start2, start_idx, end_idx, cis_end1)
            rel22 = _CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel11, rel22), (rel12, rel21))
        case _:
            raise NotImplementedError(
                f"Case not implemented for neighbor count ({len(start_neigh)}, "
                f"{len(end_neigh)}) with stereo type {stereo_type}"
            )


def _get_cis_labels_int(
    mol: rdkit.Chem.rdchem.Mol,
) -> Iterable[tuple[_CisLabelIdx, ...]]:
    mol_bonds: Iterable[rdkit.Chem.rdchem.Bond] = mol.GetBonds()  # type: ignore[no-untyped-call,unused-ignore]
    all_cis_relations = (
        g
        for g in (_get_cis_labels_int_spec(bond) for bond in mol_bonds)
        if g is not None
    )

    combo: tuple[tuple[_CisLabelIdx, ...], ...]
    for combo in itertools.product(*tuple(all_cis_relations)):
        all_labels = itertools.chain.from_iterable(combo)
        yield tuple(all_labels)


def map_core_rdkit_v4(
    mol: rdkit.Chem.rdchem.Mol,
    node_cis: bool = False,
    bond_ring: bool = False,
    bond_conj: bool = False,
) -> Iterable[tuple[TGraph[VertexData, EdgeData], IndexMapping]]:
    """Map RDKit molecule to labeled graph(s).

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to be mapped.
    node_cis : bool
        Whether to include cis nodes to distinguish stereochemistry (default:
        False).  If there is ambiguity in stereochemistry, this may produce
        multiple results.  Resulting graphs are not necessarily canonically
        unique.
    bond_ring : bool
        Whether to label bonds with `in_ring` information (default: False).
    bond_conj : bool
        Whether to label bonds with `is_conj` information (default: False).

    Raises
    ------
    ParsingError
        Raised if there is some kind of incompatibility between the RDKit
        feature and the labeling algorithm.
    """
    _mol_atoms: Iterable[rdkit.Chem.rdchem.Atom] = mol.GetAtoms()  # type: ignore[no-untyped-call,unused-ignore]
    vertices = [VertexData.from_atom(ra) for ra in _mol_atoms]
    _mol_bonds: Iterable[rdkit.Chem.rdchem.Bond] = mol.GetBonds()  # type: ignore[no-untyped-call,unused-ignore]
    bonds = [
        TBond.new(
            rb.GetBeginAtomIdx(),
            rb.GetEndAtomIdx(),
            EdgeData.from_bond(rb, bond_ring, bond_conj),
        )
        for rb in _mol_bonds
    ]
    _mol_atoms = mol.GetAtoms()  # type: ignore[no-untyped-call,unused-ignore]
    map_tup: list[int] = [ra.GetIdx() for ra in _mol_atoms]
    map_base = IndexMapping.from_seq(map_tup)

    # if no cis-mapping, then simply return graph
    if not node_cis:
        tgraph = TGraph.from_bonds(bonds, vertices)
        yield tgraph, map_base
        return

    # invert map to perform index mapping (not permutation)
    map_base_inv_tup = map_base.inv().as_tuple()

    # iterate through each set of possible cis labels
    for cis_set in _get_cis_labels_int(mol):
        cisnode_list = [VertexData.new_cis() for _ in cis_set]
        map_cis = map_base

        # build all edges and add to edge list
        cisedge_list: list[TBond[EdgeData]] = []
        for cis_i, cisnode_dat in enumerate(cis_set, start=len(vertices)):
            start_n = map_base_inv_tup[cisnode_dat.start_n]
            start_b = map_base_inv_tup[cisnode_dat.start_b]
            end_b = map_base_inv_tup[cisnode_dat.end_b]
            end_n = map_base_inv_tup[cisnode_dat.end_n]
            assert start_n is not None
            assert start_b is not None
            assert end_b is not None
            assert end_n is not None
            start_nb = TBond.new(start_n, cis_i, EdgeData.new_cis(False))
            start_bb = TBond.new(start_b, cis_i, EdgeData.new_cis(True))
            end_bb = TBond.new(end_b, cis_i, EdgeData.new_cis(True))
            end_nb = TBond.new(end_n, cis_i, EdgeData.new_cis(False))
            cisedge_list.extend([start_nb, start_bb, end_bb, end_nb])

            # extend mapping to account for new cis node with no corresponding
            # atom in original mol
            map_cis = map_cis.append(None)

        # build graph and return
        tgraph = TGraph.from_bonds(
            bonds + cisedge_list, vertices + cisnode_list
        )
        yield tgraph, map_cis


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class BondExtension:
    bond: BondFeatures
    trg: AtomCore

    @classmethod
    def from_bond(
        cls,
        bond: rdkit.Chem.rdchem.Bond,
        src_idx: int,
        ring_info: bool,
        conj_info: bool,
    ) -> "BondExtension":
        bond_features = BondFeatures.from_bond(bond, ring_info, conj_info)
        end_idx: int = bond.GetEndAtomIdx()
        trg_atom: rdkit.Chem.rdchem.Atom = (
            bond.GetBeginAtom() if src_idx == end_idx else bond.GetEndAtom()
        )
        trg_features = AtomCore.from_atom(trg_atom)
        return BondExtension(bond_features, trg_features)

    def without_extra(self) -> "BondExtension":
        return dataclasses.replace(
            self, bond=BondFeatures(self.bond.bond_type, None, None)
        )

    def as_str(self) -> str:
        return f"({self.bond.as_str()},{self.trg.as_str()})"


@final
@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AtomLabel:
    invariants: AtomCore
    h: int
    _neighbors: tuple[BondExtension, ...]
    _extra: str | None

    @classmethod
    def new(
        cls, core: AtomCore, h: int, neighbors: Iterable[BondExtension]
    ) -> "AtomLabel":
        neighbor_sort = tuple(sorted(n.without_extra() for n in neighbors))
        return AtomLabel(
            invariants=core, h=h, _neighbors=neighbor_sort, _extra=None
        )

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "AtomLabel":
        self_label = AtomCore.from_atom(atom)
        atom_idx: int = atom.GetIdx()
        all_neighbors = tuple(
            sorted(
                BondExtension.from_bond(bond, atom_idx, False, False)
                for bond in atom.GetBonds()
            )
        )
        h = atom.GetTotalNumHs()
        return AtomLabel(
            invariants=self_label, h=h, _neighbors=all_neighbors, _extra=None
        )

    def is_terminal(self) -> bool:
        return (
            self.h == 0
            and len(self._neighbors) == 1
            and self._neighbors[0].trg.d > 1
        )

    def append_extra(self, extra: str) -> "AtomLabel":
        if self._extra is None:
            return dataclasses.replace(self, _extra=extra)
        return dataclasses.replace(self, _extra=f"{self._extra},{extra}")

    def as_str(self) -> str:
        neighbor_str = ",".join(f"{n.as_str()}" for n in self._neighbors)
        if self._extra is None:
            return f"{self.invariants.as_str()},{self.h},({neighbor_str})"
        return (
            f"{self.invariants.as_str()},{self.h},({neighbor_str}),"
            f"extra({self._extra})"
        )


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class CisLabel:
    _atom1: AtomCore
    _atom2: AtomCore
    _atom3: AtomCore
    _atom4: AtomCore
    _extra: str | None

    @classmethod
    def from_cores(
        cls, atom1: AtomCore, atom2: AtomCore, atom3: AtomCore, atom4: AtomCore
    ) -> "CisLabel":
        orient_1 = (atom1, atom2, atom3, atom4)
        orient_2 = (atom4, atom3, atom2, atom1)
        orient = min(orient_1, orient_2)
        return CisLabel(
            _atom1=orient[0],
            _atom2=orient[1],
            _atom3=orient[2],
            _atom4=orient[3],
            _extra=None,
        )

    @property
    def invariant(self) -> str:
        return "cis"

    def as_str(self) -> str:
        if self._extra is None:
            return (
                f"cis,({self._atom1.as_str()}),({self._atom2.as_str()}),"
                f"({self._atom3.as_str()}),({self._atom4.as_str()})"
            )
        return (
            f"cis,({self._atom1.as_str()}),({self._atom2.as_str()}),"
            f"({self._atom3.as_str()}),({self._atom4.as_str()}),"
            f"extra({self._extra})"
        )

    def append_extra(self, ann: str) -> "CisLabel":
        if self._extra is None:
            return dataclasses.replace(self, _extra=ann)
        return dataclasses.replace(self, _extra=f"{self._extra},{ann}")


@final
@dataclasses.dataclass(frozen=True, slots=True)
class BensonLabel:
    label: AtomLabel | CisLabel

    @classmethod
    def from_vertlabel(
        cls,
        vertex: int,
        tgraph: TGraph[VertexData, EdgeData],
    ) -> "BensonLabel":
        all_vertices = tuple(tgraph.get_labels())
        self_core = all_vertices[vertex]

        # if self is a cis node, perform cis node assignment
        if self_core.atomcore is None:
            cis_startn: int | None = None
            cis_startb: int | None = None
            cis_endb: int | None = None
            cis_endn: int | None = None
            enode1: int | None = None
            enode2: int | None = None
            for neighbor_ext in tgraph.neighbors_ext(vertex):
                if neighbor_ext.typ == EdgeData(bondcore=True):
                    if cis_startb is None:
                        cis_startb = neighbor_ext.idx
                    else:
                        assert cis_endb is None
                        cis_endb = neighbor_ext.idx
                elif enode1 is None:
                    enode1 = neighbor_ext.idx
                else:
                    assert enode2 is None
                    enode2 = neighbor_ext.idx
            assert enode1 is not None
            assert enode2 is not None
            assert cis_startb is not None
            assert cis_endb is not None
            if enode1 in tgraph.neighbors(cis_startb):
                cis_startn = enode1
                cis_endn = enode2
            else:
                cis_endn = enode1
                cis_startn = enode2

            cis_startn_core = all_vertices[cis_startn].atomcore
            cis_startb_core = all_vertices[cis_startb].atomcore
            cis_endb_core = all_vertices[cis_endb].atomcore
            cis_endn_core = all_vertices[cis_endn].atomcore

            assert cis_startn_core is not None
            assert cis_startb_core is not None
            assert cis_endb_core is not None
            assert cis_endn_core is not None

            cis_label = CisLabel.from_cores(
                cis_startn_core,
                cis_startb_core,
                cis_endb_core,
                cis_endn_core,
            )

            return BensonLabel(cis_label)

        # self is an atom, proceed normally
        atom_ext: list[BondExtension] = []
        for neighbor in tgraph.neighbors_ext(vertex):
            bond_typ = neighbor.typ
            neighbor_core = all_vertices[neighbor.idx]
            if neighbor_core.atomcore is None:
                continue
            assert not isinstance(bond_typ.bondcore, bool)
            bond_ext = BondExtension(bond_typ.bondcore, neighbor_core.atomcore)
            atom_ext.append(bond_ext)

        atom_label = AtomLabel.new(self_core.atomcore, self_core.h, atom_ext)

        return BensonLabel(label=atom_label)

    def append_extra(self, ann: str) -> "BensonLabel":
        return BensonLabel(label=self.label.append_extra(ann))

    def __lt__(self, __value: "BensonLabel") -> bool:
        match (
            isinstance(self.label, AtomLabel),
            isinstance(__value.label, AtomLabel),
        ):
            case True, True:
                return self.label < __value.label  # type: ignore[operator]
            case True, False:
                return True
            case False, True:
                return False
            case False, False:
                return self.label < __value.label  # type: ignore[operator]
        raise NotImplementedError


@final
@dataclasses.dataclass(frozen=True, slots=True)
class ConvertResult_v4:
    """Conversion result data structure.

    Attributes
    ----------
    graph : TGraph[BensonLabel, EdgeData]
        Labeled graph representing converted graph.
    mapping : IndexMapping
        Index mapping corresponding to the conversion.
    extras : dict[str, list[TGraph[VertexData, EdgeData]]]
        Optional extra structures (e.g., rings).
    """

    graph: TGraph[BensonLabel, EdgeData]
    mapping: IndexMapping
    extras: dict[
        str,
        list[TGraph[BensonCore, EdgeData]]
        | list[TGraph[BensonLabel, EdgeData]],
    ]


def _clearlabels[_T: CompareHash](
    graph: TGraph[_T, EdgeData], clear_ring: bool, clear_conj: bool
) -> TGraph[_T, EdgeData]:
    if clear_ring is False and clear_conj is False:
        return graph
    bondlist: list[TBond[EdgeData]] = []
    for bond in graph.get_bonds():
        new_bond = bond
        if clear_ring:
            new_bond = TBond.new(
                src=bond.src, trg=bond.trg, color=new_bond.color.clear_ring()
            )
        if clear_conj:
            new_bond = TBond.new(
                src=bond.src, trg=bond.trg, color=new_bond.color.clear_conj()
            )
        bondlist.append(new_bond)
    new_tgraph = TGraph.from_bonds(bondlist, tuple(graph.get_labels()))
    return new_tgraph


def _extract_components(graph: igraph.Graph) -> Iterable[tuple[int, ...]]:
    all_components = graph.connected_components()
    retval: Iterable[tuple[int, ...]] = (
        tuple(tup) for tup in all_components if len(tup) > 1
    )
    return retval


def _identify_rings(
    mol: TGraph[VertexData, EdgeData],
) -> Iterable[tuple[int, ...]]:
    edgelist: list[tuple[int, int]] = []
    for bond in mol.get_bonds():
        bcolor = bond.color.bondcore
        if not isinstance(bcolor, bool) and bcolor.in_ring is True:
            edgelist.append((bond.src, bond.trg))

    cis_lookup: dict[int, tuple[int, ...]] = {}
    for i, atom in enumerate(mol.get_labels()):
        if atom.atomcore is not None:
            continue
        neighbors = tuple(mol.neighbors_ext(i))
        cis_lookup[i] = tuple(
            n.idx for n in neighbors if n.typ == EdgeData(True)
        )

    temp_graph = igraph.Graph(n=mol.nof_nodes(), edges=edgelist)

    for tup_group in _extract_components(temp_graph):
        del_cis: list[int] = []
        bigger_tup = tup_group
        for cis_i, cis_neighbors in cis_lookup.items():
            if len(tup_group) >= 4 and cis_neighbors[0] in tup_group:  # noqa: PLR2004
                if all(i in tup_group for i in cis_neighbors[1:]):
                    bigger_tup = bigger_tup + (cis_i,)
                del_cis.append(cis_i)
        yield bigger_tup
        for cis_i in del_cis:
            del cis_lookup[cis_i]


def _identify_conjs(
    mol: TGraph[VertexData, EdgeData],
) -> Iterable[tuple[int, ...]]:
    edgelist: list[tuple[int, int]] = []
    for bond in mol.get_bonds():
        bcolor = bond.color.bondcore
        if not isinstance(bcolor, bool) and bcolor.is_conj is True:
            edgelist.append((bond.src, bond.trg))

    cis_lookup: dict[int, tuple[int, ...]] = {}
    for i, atom in enumerate(mol.get_labels()):
        if atom.atomcore is not None:
            continue
        neighbors = tuple(mol.neighbors_ext(i))
        cis_lookup[i] = tuple(
            n.idx for n in neighbors if n.typ == EdgeData(True)
        )

    temp_graph = igraph.Graph(n=mol.nof_nodes(), edges=edgelist)

    for tup_group in _extract_components(temp_graph):
        del_cis: list[int] = []
        bigger_tup = tup_group
        for cis_i, cis_neighbors in cis_lookup.items():
            if len(tup_group) >= 4 and cis_neighbors[0] in tup_group:  # noqa: PLR2004
                if all(i in tup_group for i in cis_neighbors[1:]):
                    bigger_tup = bigger_tup + (cis_i,)
                del_cis.append(cis_i)
        yield bigger_tup
        for cis_i in del_cis:
            del cis_lookup[cis_i]


def __convert_bondcolor(color: EdgeData) -> str:
    if isinstance(color.bondcore, bool):
        if color.bondcore is True:
            return "CT"
        else:
            return "CF"
    return str(int(color.bondcore.bond_type))


def _convert_structure_label(tgraph: TGraph[VertexSelect, EdgeData]) -> str:
    bond_str = ",".join(
        f"{bond.src},{bond.trg},{__convert_bondcolor(bond.color)}"
        for bond in tgraph.get_bonds()
    )
    vert_str = ",".join(label.as_str() for label in tgraph.get_labels())
    tot_str = f"{bond_str};{vert_str}"
    return tot_str


def map_benson_core_v4(
    mol_graph: TGraph[VertexData, EdgeData],
    trim_ends: bool = True,
    ext_ring: bool = False,
    ext_conj: bool = False,
    feat_ring: int = 0,
    feat_conj: int = 0,
) -> ConvertResult_v4:
    """Map core graph to labeled Benson group graph(s).

    Also included are optional global structural features.  In order to properly
    detect ring and conjugation structures, `mol_graph` must include these
    properties in its `EdgeData` labels.

    Parameters
    ----------
    mol_graph : TGraph[VertexData, EdgeData]
        Valid core graph to be mapped (valid output of `map_core_rdkit_v4`).
    trim_ends : bool
        Whether to trim terminal heavy atoms (default: True).  Does not prune
        atoms from diatomic molecules.
    ext_ring : bool
        Whether to include atom-local group information in ring atom labels
        (default: False).
    ext_conj : bool
        Whether to include atom-local group information in conjugated atom
        labels (default: False).
    feat_ring : bool
        Whether to include ring structures in `extras` field of result (default:
        0).  Value of 0 is "no rings", value of 1 is "rings made from core
        labels only", value of 2 is "rings made from Benson labels".  Note:
        these structures will not be canonized by default.
    feat_conj : bool
        Whether to include conjugation structures in `extras` field of result
        (default: 0).  Value of 0 is "no conjugation structures", value of 1 is
        "conjugation structures made from core labels only", value of 2 is
        "conjugation structures made from Benson labels".  Note: these
        structures will not be canonized by default.

    Raises
    ------
    ParsingError
        Raised if there is some kind of incompatibility between the requested
        features and the labeling algorithm.
    """
    # step 0: obtain cleared form of mol
    mol_clear = _clearlabels(
        mol_graph,
        clear_ring=not ext_ring,
        clear_conj=not ext_conj,
    )

    extras: dict[
        str,
        list[TGraph[BensonCore, EdgeData]]
        | list[TGraph[BensonLabel, EdgeData]],
    ] = {}

    # step 1: perform direct mapping to tgraph of final type
    #         (obtaining benson groups)
    label_overwrite: dict[str, dict[int, str]] = {}
    vlabels_1 = tuple(
        BensonLabel.from_vertlabel(i, mol_clear)
        for i in range(mol_graph.nof_nodes())
    )
    tgraph_benson: TGraph[BensonLabel, EdgeData] = TGraph.from_bonds(
        tuple(mol_clear.get_bonds()), vlabels_1
    )

    # step 2: identify ring structures
    if ext_ring or feat_ring != 0:
        # step 2.1: get ring structure indices
        ring_structs = list(_identify_rings(mol_graph))

        # step 2.2: if feat_ring, extract ring structure(s) to extras
        if feat_ring == 1:
            ring_extras_1: list[TGraph[BensonCore, EdgeData]] = []
            for ring_struct in ring_structs:
                ring_graph_v = _clearlabels(
                    mol_graph.subset(ring_struct)[0], True, True
                )
                ring_graph = TGraph.from_bonds(
                    list(ring_graph_v.get_bonds()),
                    [v.to_core() for v in ring_graph_v.get_labels()],
                )
                ring_graph_canon = ring_graph.canonize(CanonType.F)[0]
                ring_extras_1.append(ring_graph_canon)
            extras["ring"] = ring_extras_1
        elif feat_ring == 2:  # noqa: PLR2004
            ring_extras_2: list[TGraph[BensonLabel, EdgeData]] = []
            for ring_struct in ring_structs:
                ring_graph_e = _clearlabels(
                    tgraph_benson.subset(ring_struct)[0], True, True
                )
                ring_graph_e_canon = ring_graph_e.canonize(CanonType.F)[0]
                ring_extras_2.append(ring_graph_e_canon)
            extras["ring"] = ring_extras_2

        # step 2.3: if ext_ring, extract ring structure(s) and apply to vertex
        #           labels
        if ext_ring:
            ring_overwrite: dict[int, str] = {}
            for ring_struct in ring_structs:
                ring_graph_lab, ring_map = mol_graph.subset(ring_struct)
                ring_graph_v = _clearlabels(ring_graph_lab, True, True)
                ring_graph = TGraph.from_bonds(
                    list(ring_graph_v.get_bonds()),
                    [v.to_core() for v in ring_graph_v.get_labels()],
                )
                for r_i, o_i in enumerate(ring_map.as_tuple_int()):
                    ring_labels = [
                        VertexSelect(lab) if i != r_i else VertexSelect(None)
                        for i, lab in enumerate(ring_graph.get_labels())
                    ]
                    ring_select = TGraph.from_bonds(
                        list(ring_graph.get_bonds()), ring_labels
                    )
                    ring_select_can, _ = ring_select.canonize(CanonType.F)
                    ring_label = _convert_structure_label(ring_select_can)
                    ring_overwrite[o_i] = ring_label
            label_overwrite["ring"] = ring_overwrite

    # step 3: identify conj structures
    if ext_conj or feat_conj != 0:
        # step 3.1: get conj structure indices
        conj_structs = list(_identify_conjs(mol_graph))

        # step 3.2: if feat_conj, extract conj structure to extras
        if feat_conj == 1:
            conj_extras_1: list[TGraph[BensonCore, EdgeData]] = []
            for conj_struct in conj_structs:
                conj_graph_v = _clearlabels(
                    mol_graph.subset(conj_struct)[0], True, True
                )
                conj_graph = TGraph.from_bonds(
                    list(conj_graph_v.get_bonds()),
                    [v.to_core() for v in conj_graph_v.get_labels()],
                )
                conj_graph_canon = conj_graph.canonize(CanonType.F)[0]
                conj_extras_1.append(conj_graph_canon)
            extras["conj"] = conj_extras_1
        elif feat_conj == 2:  # noqa: PLR2004
            conj_extras_2 = []
            for conj_struct in conj_structs:
                conj_graph_e = _clearlabels(
                    tgraph_benson.subset(conj_struct)[0], True, True
                )
                conj_graph_e_canon = conj_graph_e.canonize(CanonType.F)[0]
                conj_extras_2.append(conj_graph_e_canon)
            extras["conj"] = conj_extras_2

        # step 3.3: if ext_conj, extract conj structures and apply to vertex
        #           labels
        if ext_conj:
            conj_overwrite: dict[int, str] = {}
            for conj_struct in conj_structs:
                conj_graph_lab, conj_map = mol_graph.subset(conj_struct)
                conj_graph_v = _clearlabels(conj_graph_lab, True, True)
                conj_graph = TGraph.from_bonds(
                    list(conj_graph_v.get_bonds()),
                    [v.to_core() for v in conj_graph_v.get_labels()],
                )
                for c_i, o_i in enumerate(conj_map.as_tuple_int()):
                    conj_labels = [
                        VertexSelect(lab) if i != c_i else VertexSelect(None)
                        for i, lab in enumerate(conj_graph.get_labels())
                    ]
                    conj_select = TGraph.from_bonds(
                        list(conj_graph.get_bonds()), conj_labels
                    )
                    conj_select_can, _ = conj_select.canonize(CanonType.F)
                    conj_label = _convert_structure_label(conj_select_can)
                    conj_overwrite[o_i] = conj_label
            label_overwrite["conj"] = conj_overwrite

    # step 4: add label overwrites, if they exist
    if len(label_overwrite) > 0:
        benson_labels = list(tgraph_benson.get_labels())
        for extra_type, extra_data in label_overwrite.items():
            for v_i, strval in extra_data.items():
                new_data = f"{extra_type}({strval})"
                old_benson_label = benson_labels[v_i]
                new_benson_label = old_benson_label.append_extra(new_data)
                benson_labels[v_i] = new_benson_label
        tgraph_benson = TGraph.from_bonds(
            list(tgraph_benson.get_bonds()), benson_labels
        )

    # step 5: if trim is enabled, trim molecules and adjust mapping accordingly
    if trim_ends:
        accept_i: list[int] = []
        for i, label in enumerate(tgraph_benson.get_labels()):
            if isinstance(label.label, AtomLabel) and label.label.is_terminal():
                continue
            accept_i.append(i)
        tgraph_benson, mapping = tgraph_benson.subset(accept_i)
    else:
        mapping = IndexMapping.identity(tgraph_benson.nof_nodes())

    # step 6: return final components
    retval = ConvertResult_v4(
        graph=tgraph_benson, mapping=mapping, extras=extras
    )
    return retval


def map_benson_v4(
    mol: rdkit.Chem.rdchem.Mol,
    node_cis: bool = False,
    trim_ends: bool = True,
    ext_ring: bool = False,
    ext_conj: bool = False,
    feat_ring: int = 0,
    feat_conj: int = 0,
) -> Iterable[ConvertResult_v4]:
    """Map RDKit molecule to labeled Benson group graph(s).

    Also included are optional global structural features.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule to be mapped.
    node_cis : bool
        Whether to include cis nodes to distinguish stereochemistry (default:
        False).  If there is ambiguity in stereochemistry, this may produce
        multiple results.  Resulting graphs are not necessarily canonically
        unique.
    trim_ends : bool
        Whether to trim terminal heavy atoms (default: True).  Does not prune
        atoms from diatomic molecules.
    ext_ring : bool
        Whether to include atom-local group information in ring atom labels
        (default: False).
    ext_conj : bool
        Whether to include atom-local group information in conjugated atom
        labels (default: False).
    feat_ring : bool
        Whether to include ring structures in `extras` field of result (default:
        0).  Value of 0 is "no rings", value of 1 is "rings made from core
        labels only", value of 2 is "rings made from Benson labels".  Note:
        these structures will not be canonized by default.
    feat_conj : bool
        Whether to include conjugation structures in `extras` field of result
        (default: 0).  Value of 0 is "no conjugation structures", value of 1 is
        "conjugation structures made from core labels only", value of 2 is
        "conjugation structures made from Benson labels".  Note: these
        structures will not be canonized by default.

    Raises
    ------
    ParsingError
        Raised if there is some kind of incompatibility between the requested
        features and the labeling algorithm.
    """
    coremaps = map_core_rdkit_v4(
        mol,
        node_cis=node_cis,
        bond_ring=ext_ring or feat_ring != 0,
        bond_conj=ext_conj or feat_conj != 0,
    )
    for coremap, mapping_core in coremaps:
        res = map_benson_core_v4(
            coremap,
            trim_ends=trim_ends,
            ext_ring=ext_ring,
            ext_conj=ext_conj,
            feat_ring=feat_ring,
            feat_conj=feat_conj,
        )
        res_final = dataclasses.replace(
            res, mapping=res.mapping.compose(mapping_core)
        )
        yield res_final
