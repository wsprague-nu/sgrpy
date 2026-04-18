"""RDKit to UGraph conversion functions."""

# mypy: allow-untyped-calls

import dataclasses
import enum
import itertools
from collections.abc import Collection, Container, Iterable, Mapping, Sequence

import igraph
import rdkit.Chem.rdchem

from sgrpy.graph import Bond, UGraph, canonize_ugraph

from ._structures import (
    AtomLabel,
    BondStereo,
    BondType,
    CisLabel,
    RDKitTranslationError,
)


class FeatureFormat(enum.IntEnum):
    NONE = 0
    CORE = 1
    EXT = 2


@dataclasses.dataclass(frozen=True, slots=True)
class AtomGraph:
    nodes: tuple[AtomLabel | CisLabel, ...]
    bonds: tuple[Bond, ...]

    def to_ugraph(self) -> UGraph:
        labels = (node.as_str() for node in self.nodes)
        ugraph = UGraph.from_bonds(self.bonds, labels=labels)
        return ugraph

    def canon_str(self) -> str:
        ugc = canonize_ugraph(self.to_ugraph())
        return ugc.to_sugraph().as_str()


@dataclasses.dataclass(frozen=True, slots=True)
class ConvertResult:
    graph: AtomGraph
    extras: dict[str, list[str]] | None

    def to_ugraph(self) -> UGraph:
        return self.graph.to_ugraph()


def assert_compact_idx(mol: rdkit.Chem.rdchem.Mol) -> None:
    total_atoms: int = mol.GetNumAtoms()
    atom_indices: list[int] = [atom.GetIdx() for atom in mol.GetAtoms()]
    if sorted(atom_indices) != list(range(total_atoms)):
        raise RDKitTranslationError(
            "Atom indices are not compact (mapped to range starting from 0): "
            f"{sorted(atom_indices)}"
        )


def get_atom_data(
    mol: rdkit.Chem.rdchem.Mol, trim_ends: bool
) -> tuple[list[AtomLabel], list[int]]:
    atom_labels: list[AtomLabel] = []
    term_atoms: list[int] = []
    atom: rdkit.Chem.rdchem.Atom
    for atom in mol.GetAtoms():
        label = AtomLabel.from_atom(atom)
        atom_labels.append(label)
        if trim_ends and label.is_terminal():
            idx = atom.GetIdx()
            term_atoms.append(idx)
    return atom_labels, term_atoms


def get_bond_data(
    mol: rdkit.Chem.rdchem.Mol, ring_info: bool, conj_info: bool
) -> tuple[list[Bond], list[int], list[int]]:
    bond_list: list[Bond] = []
    ring_list: list[int] = []
    conj_list: list[int] = []
    rdbond: rdkit.Chem.rdchem.Bond
    for i, rdbond in enumerate(mol.GetBonds()):
        src: int = rdbond.GetBeginAtomIdx()
        trg: int = rdbond.GetEndAtomIdx()
        bond = Bond.new(src, trg)
        bond_list.append(bond)
        if ring_info and rdbond.IsInRing():
            ring_list.append(i)
        if conj_info and rdbond.GetIsConjugated():
            conj_list.append(i)

    return bond_list, ring_list, conj_list


def __check_bond_ring_lt_8(bond: rdkit.Chem.rdchem.Bond) -> bool:
    if not bond.IsInRing():
        return False
    return any(bond.IsInRingSize(i) for i in range(8))


@dataclasses.dataclass(frozen=True, slots=True)
class CisLabelIdx:
    start_n: int
    start_b: int
    end_b: int
    end_n: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.start_n, self.start_b, self.end_b, self.end_n)

    def as_tuple_sort(self) -> tuple[int, int, int, int]:
        self_tup = self.as_tuple()
        rev_tup = (self_tup[3], self_tup[2], self_tup[1], self_tup[0])
        return min(self_tup, rev_tup)


def get_cis_labels_int_spec(
    bond: rdkit.Chem.rdchem.Bond,
) -> tuple[tuple[CisLabelIdx, ...], ...] | None:
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
                raise NotImplementedError(
                    "Stereo specification for ring <8 atoms currently invalid"
                )
            case BondStereo.STEREONONE | BondStereo.STEREOANY, False:
                raise NotImplementedError(
                    "Disambiguation of stereochemical specifiers for large "
                    "rings not currently supported"
                )
            case BondStereo.STEREOZ | BondStereo.STEREOE, False:
                pass
            case _:  # this should be unreachable
                raise NotImplementedError("Invalid configuration")

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
            relation_cis = CisLabelIdx(
                start_neigh[0], start_idx, end_idx, end_neigh[0]
            )
            return ((relation_cis,),)
        # start=1, end=1; undefined stereo so return both possibilities
        case 1, _, BondStereo.STEREONONE | BondStereo.STEREOANY:
            relation_trans = ()
            relation_cis = CisLabelIdx(
                start_neigh[0], start_idx, end_idx, end_neigh[0]
            )
            return (relation_trans, (relation_cis,))
        # start=2, end=1; STEREOZ maps the cis atoms so this is easy
        case 2, 1, BondStereo.STEREOZ:
            stereo_atoms = tuple(bond.GetStereoAtoms())  # type: ignore[assignment,unused-ignore]
            cis_start = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end = end_neigh[0]
            relation_cis = CisLabelIdx(cis_start, start_idx, end_idx, cis_end)
            return ((relation_cis,),)
        # start=2, end=1; STEREOE maps the trans atoms so a swap is needed
        case 2, 1, BondStereo.STEREOE:
            stereo_atoms = tuple(bond.GetStereoAtoms())  # type: ignore[assignment,unused-ignore]
            cis_start = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end = end_neigh[0]
            relation_cis = CisLabelIdx(cis_start, start_idx, end_idx, cis_end)
            return ((relation_cis,),)
        # start=2, end=1; undefined stereo so return both possibilities
        case 2, 1, BondStereo.STEREONONE | BondStereo.STEREOANY:
            cis_start1 = start_neigh[0]
            cis_start2 = start_neigh[1]  # type: ignore[unused-ignore]
            cis_end1 = end_neigh[0]
            rel1 = CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = CisLabelIdx(cis_start2, start_idx, end_idx, cis_end1)
            return ((rel1,), (rel2,))
        # start=2, end=2; stereo atoms define cis
        case 2, 2, BondStereo.STEREOZ:
            stereo_atoms = bond.GetStereoAtoms()  # type: ignore[assignment,unused-ignore]
            cis_start1 = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end1 = [sn for sn in end_neigh if sn in stereo_atoms][0]
            cis_start2 = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end2 = [sn for sn in end_neigh if sn not in stereo_atoms][0]
            rel1 = CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel1, rel2),)
        # start=2, end=2; stereo atoms define trans
        case 2, 2, BondStereo.STEREOE:
            stereo_atoms = bond.GetStereoAtoms()  # type: ignore[assignment,unused-ignore]
            cis_start1 = [sn for sn in start_neigh if sn in stereo_atoms][0]
            cis_end1 = [sn for sn in end_neigh if sn not in stereo_atoms][0]
            cis_start2 = [sn for sn in start_neigh if sn not in stereo_atoms][0]
            cis_end2 = [sn for sn in end_neigh if sn in stereo_atoms][0]
            rel1 = CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel2 = CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel1, rel2),)
        # start=2, end=2; stereo atoms undefined so return both possibilities
        case 2, 2, BondStereo.STEREONONE | BondStereo.STEREOANY:
            cis_start1 = start_neigh[0]
            cis_end1 = end_neigh[0]
            cis_end2 = end_neigh[1]  # type: ignore[unused-ignore]
            cis_start2 = start_neigh[1]  # type: ignore[unused-ignore]
            rel11 = CisLabelIdx(cis_start1, start_idx, end_idx, cis_end1)
            rel12 = CisLabelIdx(cis_start1, start_idx, end_idx, cis_end2)
            rel21 = CisLabelIdx(cis_start2, start_idx, end_idx, cis_end1)
            rel22 = CisLabelIdx(cis_start2, start_idx, end_idx, cis_end2)
            return ((rel11, rel22), (rel12, rel21))
        case _:
            raise NotImplementedError(
                f"Case not implemented for neighbor count ({len(start_neigh)}, "
                f"{len(end_neigh)}) with stereo type {stereo_type}"
            )


def get_cis_labels_int(
    mol: rdkit.Chem.rdchem.Mol,
) -> Iterable[tuple[CisLabelIdx, ...]]:
    all_cis_relations = (
        g
        for g in (get_cis_labels_int_spec(bond) for bond in mol.GetBonds())
        if g is not None
    )

    combo: tuple[tuple[CisLabelIdx, ...], ...]
    for combo in itertools.product(*tuple(all_cis_relations)):
        all_labels = itertools.chain.from_iterable(combo)
        yield tuple(all_labels)


def label_cis(idx: CisLabelIdx, atom_labels: Sequence[AtomLabel]) -> CisLabel:
    atom1 = atom_labels[idx.start_n].invariants
    atom2 = atom_labels[idx.start_b].invariants
    atom3 = atom_labels[idx.end_b].invariants
    atom4 = atom_labels[idx.end_n].invariants
    cislabel = CisLabel.from_cores(atom1, atom2, atom3, atom4)
    return cislabel


def get_cis_labels(
    mol: rdkit.Chem.rdchem.Mol, atom_labels: Sequence[AtomLabel]
) -> Iterable[tuple[tuple[CisLabel, CisLabelIdx], ...]]:
    for cis_set in get_cis_labels_int(mol):
        yield tuple((label_cis(cidx, atom_labels), cidx) for cidx in cis_set)


def add_cis_labels(
    mol: rdkit.Chem.rdchem.Mol,
    atom_list: Sequence[AtomLabel],
    bond_list: Sequence[Bond],
) -> Iterable[
    tuple[list[AtomLabel | CisLabel], list[Bond], dict[Bond, tuple[int, ...]]]
]:
    num_atoms = len(atom_list)
    num_bonds = len(bond_list)
    for cislabel_set in get_cis_labels(mol, atom_list):
        atom_list_new: list[AtomLabel | CisLabel] = list(atom_list)
        bond_list_new = list(bond_list)
        cislabel_new: list[CisLabelIdx] = []
        trigger_dict: dict[Bond, tuple[int, ...]] = {}
        for node_i, (cislabel, cislabel_idx) in enumerate(cislabel_set):
            atom_list_new.append(cislabel)
            cislabel_new.append(cislabel_idx)
            bond1 = Bond.new(num_atoms + node_i, cislabel_idx.start_n)
            bond2 = Bond.new(num_atoms + node_i, cislabel_idx.start_b)
            bond3 = Bond.new(num_atoms + node_i, cislabel_idx.end_b)
            bond4 = Bond.new(num_atoms + node_i, cislabel_idx.end_n)
            bond_list_new.extend((bond1, bond2, bond3, bond4))
            trigger_bond = Bond.new(cislabel_idx.start_b, cislabel_idx.end_b)
            bond2_id = num_bonds + node_i * 4 + 1
            bond3_id = bond2_id + 1
            trigger_dict[trigger_bond] = (bond2_id, bond3_id)
        yield atom_list_new, bond_list_new, trigger_dict


def aug_trigger_bonds(
    bond_list: Sequence[Bond],
    trigger_dict: Mapping[Bond, tuple[int, ...]],
    bonds_used: Collection[Bond],
) -> frozenset[Bond]:
    cur_bonds = frozenset(bonds_used)
    if len(trigger_dict) == 0:
        return cur_bonds
    return cur_bonds.union(
        itertools.chain.from_iterable(
            (bond_list[b_i] for b_i in b)
            for b in (trigger_dict.get(bond) for bond in cur_bonds)
            if b is not None
        )
    )


def remap_bonds(bonds: Iterable[Bond]) -> tuple[list[Bond], list[int]]:
    """Remap bonds to make resulting graph(s) compact."""
    new_bond_list: list[Bond] = []
    id_map: dict[int, int] = {}
    for bond in bonds:
        if bond.src not in id_map:
            id_map[bond.src] = len(id_map)
        if bond.trg not in id_map:
            id_map[bond.trg] = len(id_map)
        new_bond = Bond.new(id_map[bond.src], id_map[bond.trg])
        new_bond_list.append(new_bond)
    id_unmap: list[int] = [0 for _ in range(len(id_map))]
    for id_orig, id_new in id_map.items():
        id_unmap[id_new] = id_orig
    return new_bond_list, id_unmap


def locate_structures(
    bonds: Iterable[Bond],
) -> tuple[list[UGraph], list[list[int]]]:
    """Locate structure mappings.

    Note: returned UGraph does not have any labels applied!
    """
    bonds_remapped, id_map = remap_bonds(bonds)
    graph = igraph.Graph(
        edges=[(bond.src, bond.trg) for bond in bonds_remapped],
        vertex_attrs={"name": id_map},
    )
    vcluster = graph.connected_components()

    result_map: list[list[int]] = []
    result_ug: list[UGraph] = []
    for sg in vcluster.subgraphs():
        result_map_graph: list[int] = []
        # create ugraph from bonds
        num_verts = len(sg.vs)
        ug = UGraph.from_tuples(
            sg.get_edgelist(), labels=["" for _ in range(num_verts)]
        )
        result_ug.append(ug)

        # build mapping for individual nodes
        result_map_graph = [int(vertex["name"]) for vertex in sg.vs]
        result_map.append(result_map_graph)

    return result_ug, result_map


def annotate_structure(
    ugraph: UGraph,
    labels: Sequence[str],
    special: int | None,
) -> UGraph:
    """Return canonical UGraph structure after annotating."""
    if special is not None:
        labels = list(labels)
        labels[special] = "!"
    new_ug = UGraph.from_bonds(ugraph.get_bonds(), labels)
    new_ugc = canonize_ugraph(new_ug)
    return new_ugc


def get_label_str(label: AtomLabel | CisLabel, detail: FeatureFormat) -> str:
    match detail, label:
        case FeatureFormat.NONE, _:
            return ""
        case FeatureFormat.EXT, _:
            return label.as_str()
        case FeatureFormat.CORE, AtomLabel():
            return label.invariants.as_str()
        case FeatureFormat.CORE, CisLabel():
            return "cis"
    raise ValueError("Something went wrong")


def extract_struc(
    atom_list: Sequence[AtomLabel | CisLabel],
    ugraphs: Sequence[UGraph],
    struc_map: Sequence[Sequence[int]],
    detail: FeatureFormat,
) -> Iterable[str]:
    """Return layered structure features."""
    for mapping, ug in zip(struc_map, ugraphs, strict=True):
        labels_str = [get_label_str(atom_list[i], detail) for i in mapping]
        new_ug = annotate_structure(ug, labels_str, None)
        new_ug_str = new_ug.to_sugraph().as_str()
        yield new_ug_str


def relabel_nodes_struc(
    atom_list: Sequence[AtomLabel | CisLabel],
    ugraphs: Sequence[UGraph],
    struc_map: list[list[int]],
) -> Iterable[tuple[int, str]]:
    """Return new labels for certain nodes."""
    for mapping, ug in zip(struc_map, ugraphs, strict=True):
        labels_str = [
            get_label_str(atom_list[i], FeatureFormat.CORE) for i in mapping
        ]
        for sub_i, orig_i in enumerate(mapping):
            new_ug = annotate_structure(ug, labels_str, sub_i)
            new_ug_str = new_ug.to_sugraph().as_str()
            yield orig_i, new_ug_str


def trim_atoms(agraph: AtomGraph, trim_atoms: Container[int]) -> AtomGraph:
    new_labels = (
        lab for i, lab in enumerate(agraph.nodes) if i not in trim_atoms
    )
    atom_idxs = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(
            i for i in range(len(agraph.nodes)) if i not in trim_atoms
        )
    }
    new_bonds = (
        Bond.new(src=atom_idxs[bond.src], trg=atom_idxs[bond.trg])
        for bond in agraph.bonds
        if bond.src not in trim_atoms and bond.trg not in trim_atoms
    )
    new_ug = AtomGraph(tuple(new_labels), tuple(new_bonds))
    return new_ug


def construct_result(
    atom_list: Sequence[AtomLabel | CisLabel],
    bond_list: Sequence[Bond],
    trigger_dict: Mapping[Bond, tuple[int, ...]],
    term_atoms: Sequence[int],
    ring_bonds: Sequence[int],
    conj_bonds: Sequence[int],
    ext_ring: bool,
    ext_conj: bool,
    feat_ring: FeatureFormat,
    feat_conj: FeatureFormat,
) -> ConvertResult:
    atom_list_mod: list[tuple[int, str]] = []
    return_json: None | dict[str, list[str]] = None
    # step 1: detect and assign rings
    if len(ring_bonds) > 0 and (ext_ring or feat_ring != FeatureFormat.NONE):
        ring_bond_exp = frozenset(bond_list[i] for i in ring_bonds)

        # add trigger bonds
        ring_bond_exp = aug_trigger_bonds(
            bond_list, trigger_dict, ring_bond_exp
        )

        # locate relevant structure assignments
        blank_ugs, ring_map = locate_structures(ring_bond_exp)

        # if ring feature flag is set, extract canonical rings
        if feat_ring != FeatureFormat.NONE:
            ring_feat_list = list(
                f"ring({ring_str})"
                for ring_str in extract_struc(
                    atom_list, blank_ugs, ring_map, feat_ring
                )
            )
            if return_json is None:
                return_json = {"ring": ring_feat_list}
            else:
                return_json["ring"] = ring_feat_list

        # if using local ring features, extract feature mods
        if ext_ring:
            ring_feat_gen = (
                (i, f"ring({s})")
                for i, s in relabel_nodes_struc(atom_list, blank_ugs, ring_map)
            )
            atom_list_mod.extend(ring_feat_gen)

    # step 2: assign and detect conjugation
    if len(conj_bonds) > 0 and (ext_conj or feat_conj != FeatureFormat.NONE):
        conj_bond_exp = frozenset(bond_list[i] for i in conj_bonds)

        # add trigger bonds
        conj_bond_exp = aug_trigger_bonds(
            bond_list, trigger_dict, conj_bond_exp
        )

        # locate relevant structure assignments
        blank_ugs, conj_map = locate_structures(conj_bond_exp)

        # if ring feature flag is set, extract canonical conjs
        if feat_conj != FeatureFormat.NONE:
            conj_feat_list = list(
                f"conj({conj_str})"
                for conj_str in extract_struc(
                    atom_list, blank_ugs, conj_map, feat_conj
                )
            )
            if return_json is None:
                return_json = {"conj": conj_feat_list}
            else:
                return_json["conj"] = conj_feat_list

        # if using local conj features, extract feature mods
        if ext_conj:
            conj_feat_gen = (
                (i, f"conj({s})")
                for i, s in relabel_nodes_struc(atom_list, blank_ugs, conj_map)
            )
            atom_list_mod.extend(conj_feat_gen)

    # step 3: annotate UGraph and create final stage
    final_atom_list = atom_list
    if len(atom_list_mod) > 0:
        final_atom_list = list(atom_list)
        for mod_i, mod in atom_list_mod:
            final_atom_list[mod_i] = final_atom_list[mod_i].append_extra(mod)
    final_ag = AtomGraph(nodes=tuple(final_atom_list), bonds=tuple(bond_list))

    # step 4: trim nodes
    if len(term_atoms) > 0:
        final_ag = trim_atoms(final_ag, term_atoms)

    return ConvertResult(graph=final_ag, extras=return_json)


def convert_mol_general(
    mol: rdkit.Chem.rdchem.Mol,
    trim_ends: bool = True,
    node_cis: bool = False,
    ext_ring: bool = False,
    ext_conj: bool = False,
    feat_ring: FeatureFormat | int = FeatureFormat.NONE,
    feat_conj: FeatureFormat | int = FeatureFormat.NONE,
) -> Iterable[ConvertResult]:
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
    Iterable[ConvertResult]
        Iterable of results.  Length zero if no results obtained.
    """
    # convert to featureflag
    if isinstance(feat_ring, int):
        feat_ring = FeatureFormat(feat_ring)
    if isinstance(feat_conj, int):
        feat_conj = FeatureFormat(feat_conj)

    # check to make sure indices of atoms are properly compact
    assert_compact_idx(mol)

    atom_list, term_atoms = get_atom_data(mol, trim_ends)

    bond_list, ring_bonds, conj_bonds = get_bond_data(
        mol,
        ring_info=(ext_ring or feat_ring != FeatureFormat.NONE),
        conj_info=(ext_conj or feat_conj != FeatureFormat.NONE),
    )

    items_iter = (
        add_cis_labels(mol, atom_list, bond_list)
        if node_cis
        else ((atom_list, bond_list, {}),)
    )
    for atoms_all, bonds_all, trigger_dict in items_iter:
        yield construct_result(
            atoms_all,
            bonds_all,
            trigger_dict,
            term_atoms=term_atoms,
            ring_bonds=ring_bonds,
            conj_bonds=conj_bonds,
            ext_ring=ext_ring,
            ext_conj=ext_conj,
            feat_ring=feat_ring,
            feat_conj=feat_conj,
        )
