"""Molecule feature structures."""

import dataclasses
import enum

import rdkit.Chem.rdchem


class Hybridization(enum.IntEnum):
    S = enum.auto()
    SP = enum.auto()
    SP2 = enum.auto()
    SP3 = enum.auto()
    SP3D = enum.auto()
    SP3D2 = enum.auto()

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "Hybridization":
        match atom.GetHybridization():
            case rdkit.Chem.rdchem.HybridizationType.S:
                return Hybridization.S
            case rdkit.Chem.rdchem.HybridizationType.SP:
                return Hybridization.SP
            case rdkit.Chem.rdchem.HybridizationType.SP2:
                return Hybridization.SP2
            case rdkit.Chem.rdchem.HybridizationType.SP3:
                return Hybridization.SP3
            case rdkit.Chem.rdchem.HybridizationType.SP3D:
                return Hybridization.SP3D
            case rdkit.Chem.rdchem.HybridizationType.SP3D2:
                return Hybridization.SP3D2
        raise NotImplementedError(
            f"Unimplemented hybridization type {atom.GetHybridization()}"
        )


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AtomCore:
    """Atom core invariants dataclass."""

    p: int  # proton number
    v: int  # valence
    d: int  # total degree
    c: int  # formal charge
    rad: int  # num radical electrons
    hyb: Hybridization  # hybridization of atom
    a: bool  # is atom aromatic
    n: int  # isotope number

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "AtomCore":
        p: int = atom.GetAtomicNum()
        v: int = atom.GetTotalValence()
        d: int = atom.GetTotalDegree()
        c: int = atom.GetFormalCharge()
        rad: int = atom.GetNumRadicalElectrons()
        a: bool = atom.GetIsAromatic()
        n: int = atom.GetIsotope()
        hyb = Hybridization.from_atom(atom)
        if p == 1:
            raise NotImplementedError("Explicit hydrogens not supported")
        return AtomCore(p=p, v=v, d=d, c=c, rad=rad, hyb=hyb, a=a, n=n)

    def as_str(self) -> str:
        return ",".join(str(x) for x in dataclasses.astuple(self))


class BondType(enum.IntEnum):
    SINGLE = enum.auto()
    DOUBLE = enum.auto()
    TRIPLE = enum.auto()
    AROMATIC = enum.auto()

    @classmethod
    def from_bond(cls, bond: rdkit.Chem.rdchem.Bond) -> "BondType":
        match bond.GetBondType():
            case rdkit.Chem.rdchem.BondType.SINGLE:
                return BondType.SINGLE
            case rdkit.Chem.rdchem.BondType.DOUBLE:
                return BondType.DOUBLE
            case rdkit.Chem.rdchem.BondType.TRIPLE:
                return BondType.TRIPLE
            case rdkit.Chem.rdchem.BondType.AROMATIC:
                return BondType.AROMATIC

        raise NotImplementedError(
            f"Unimplemented bond type {bond.GetBondType()}"
        )


class BondStereo(enum.IntEnum):
    STEREOE = enum.auto()
    STEREOZ = enum.auto()
    STEREONONE = enum.auto()
    STEREOANY = enum.auto()

    @classmethod
    def from_bond(cls, bond: rdkit.Chem.rdchem.Bond) -> "BondStereo":
        match bond.GetStereo():
            case rdkit.Chem.rdchem.BondStereo.STEREOE:
                return BondStereo.STEREOE
            case rdkit.Chem.rdchem.BondStereo.STEREOZ:
                return BondStereo.STEREOZ
            case rdkit.Chem.rdchem.BondStereo.STEREONONE:
                return BondStereo.STEREONONE
            case rdkit.Chem.rdchem.BondStereo.STEREOANY:
                return BondStereo.STEREOANY

        raise NotImplementedError(
            f"Unimplemented bond stereo {bond.GetStereo()}"
        )


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class BondFeatures:
    """Bond invariant dataclass."""

    bond_type: BondType  # type of bond
    # conjugated: bool  # whether bond is conjugated
    # ring_size: int  # size of smallest ring containing bond
    # carbenium_neighbor: bool
    # radical_neighbor: bool

    @classmethod
    def from_bond(
        cls, bond: rdkit.Chem.rdchem.Bond, src: int
    ) -> "BondFeatures":
        bond_type = BondType.from_bond(bond)
        # conjugated: bool = bond.GetIsConjugated()

        return BondFeatures(bond_type=bond_type)

    def as_str(self) -> str:
        return f"{str(self.bond_type)},"
        # return ",".join(str(x) for x in dataclasses.astuple(self))


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class BondExtension:
    bond: BondFeatures
    trg: AtomCore

    @classmethod
    def from_bond(
        cls, bond: rdkit.Chem.rdchem.Bond, src: int
    ) -> "BondExtension":
        self_features = BondFeatures.from_bond(bond, src)
        end_idx = bond.GetEndAtomIdx()
        target_atom: rdkit.Chem.rdchem.Atom = (
            bond.GetBeginAtom() if src == end_idx else bond.GetEndAtom()
        )
        target_features = AtomCore.from_atom(target_atom)
        return BondExtension(self_features, target_features)

    def as_str(self) -> str:
        return f"({self.bond.as_str()}),({self.trg.as_str()})"


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class AtomLabel:
    invariants: AtomCore
    h: int  # number of hydrogens
    _neighbors: tuple[BondExtension, ...]
    _extra: str | None = None

    @classmethod
    def from_atom(cls, atom: rdkit.Chem.rdchem.Atom) -> "AtomLabel":
        self_label = AtomCore.from_atom(atom)
        atom_idx = atom.GetIdx()
        all_neighbors = tuple(
            sorted(
                BondExtension.from_bond(bond, atom_idx)
                for bond in atom.GetBonds()
            )
        )
        h = atom.GetTotalNumHs()
        return AtomLabel(invariants=self_label, h=h, _neighbors=all_neighbors)

    def as_str(self) -> str:
        neighbor_str = ",".join(f"({n.as_str()})" for n in self._neighbors)
        if self._extra is None:
            return f"({self.invariants.as_str()}),{self.h},({neighbor_str})"
        return (
            f"({self.invariants.as_str()}),{self.h},({neighbor_str}),"
            f"extra({self._extra})"
        )

    def is_terminal(self) -> bool:
        return (
            self.h == 0
            and len(self._neighbors) == 1
            and self._neighbors[0].trg.d > 1
        )

    def no_extra(self) -> "AtomLabel":
        return dataclasses.replace(self, _extra=None)

    def append_extra(self, ann: str) -> "AtomLabel":
        if self._extra is None:
            return dataclasses.replace(self, _extra=ann)
        return dataclasses.replace(self, _extra=f"{self._extra},{ann}")


@dataclasses.dataclass(frozen=True, order=True, slots=True)
class CisLabel:
    _atom1: AtomCore
    _atom2: AtomCore
    _atom3: AtomCore
    _atom4: AtomCore
    _extra: str | None = None

    @classmethod
    def from_cores(
        cls, atom1: AtomCore, atom2: AtomCore, atom3: AtomCore, atom4: AtomCore
    ) -> "CisLabel":
        orient_1 = (atom1, atom2, atom3, atom4)
        orient_2 = (atom4, atom3, atom2, atom1)
        orient_1 = min(orient_1, orient_2)
        return CisLabel(
            _atom1=orient_1[0],
            _atom2=orient_1[1],
            _atom3=orient_1[2],
            _atom4=orient_1[3],
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


class RDKitTranslationError(Exception):
    """Raised when molecule fails to be parsed by RDKit methods."""
