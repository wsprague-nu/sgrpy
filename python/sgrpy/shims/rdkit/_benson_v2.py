"""Implementation of Benson-type shim from RDKit molecule to UGraph V2.

Notes
-----
Implements group encoding (atom invariants with neighbor bonds and atom
invariants). Goes further than Benson groups by including ring size information.
Also has an optional extension to include "cis" label node for double bond
stereochem.
"""

# mypy: allow-untyped-calls

import collections.abc
import dataclasses
import itertools

import igraph
import rdkit.Chem.rdchem

import sgrpy

from ._structures import AtomCore, AtomLabel, BondStereo, BondType, CisLabel


def get_ring_complexes(
    ring_bonds: list[tuple[int, int]],
) -> collections.abc.Iterable[igraph.Graph]:
    graph: igraph.Graph = igraph.Graph.TupleList(ring_bonds)

    comp: list[int]
    for comp in graph.components():
        if len(comp) < 3:  # noqa: PLR2004
            continue
        sg = graph.induced_subgraph(comp)
        yield sg


def no_extras(label: AtomLabel | CisLabel) -> AtomLabel | CisLabel:
    if isinstance(label, CisLabel):
        return label
    return label.no_extra()


def __check_bond_ring_lt_8(bond: rdkit.Chem.rdchem.Bond) -> bool:
    if not bond.IsInRing():
        return False
    return any(bond.IsInRingSize(i) for i in range(8))


def __check_radical_error(atom: rdkit.Chem.rdchem.Atom) -> bool:
    """Check for radical hybridization error."""
    atomic_num: int = atom.GetAtomicNum()
    degree: int = atom.GetDegree()
    totaldegree: int = atom.GetTotalDegree()
    hyb: rdkit.Chem.rdchem.HybridizationType = atom.GetHybridization()
    rad: int = atom.GetNumRadicalElectrons()
    return (atomic_num, degree, totaldegree, hyb, rad) == (
        6,
        2,
        2,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        1,
    )


def get_cis_labels(
    mol: rdkit.Chem.rdchem.Mol,
) -> collections.abc.Iterable[tuple[tuple[int, int, int, int], ...]]:
    cis_relations: list[tuple[tuple[tuple[int, int, int, int], ...], ...]] = []
    for bond in mol.GetBonds():
        bond_type = BondType.from_bond(bond)
        if bond_type != BondType.DOUBLE:
            continue
        start_atom: rdkit.Chem.rdchem.Atom = bond.GetBeginAtom()
        end_atom: rdkit.Chem.rdchem.Atom = bond.GetEndAtom()
        start_idx: int = start_atom.GetIdx()
        end_idx: int = end_atom.GetIdx()
        start_neigh: tuple[int, ...] = tuple(
            a.GetIdx()
            for a in start_atom.GetNeighbors()
            if a.GetIdx() != end_idx
        )
        end_neigh: tuple[int, ...] = tuple(
            a.GetIdx()
            for a in end_atom.GetNeighbors()
            if a.GetIdx() != start_idx
        )
        # if either end is terminal or bond is in small ring, no cis is possible
        if (
            len(start_neigh) == 0
            or len(end_neigh) == 0
            or start_atom.GetHybridization()
            == rdkit.Chem.rdchem.HybridizationType.SP
            or end_atom.GetHybridization()
            == rdkit.Chem.rdchem.HybridizationType.SP
            or __check_bond_ring_lt_8(bond)
            # or __check_radical_error(start_atom)
            # or __check_radical_error(end_atom)
        ):
            continue

        if len(start_neigh) > 2 or len(end_neigh) > 2:  # noqa: PLR2004
            # complex double bonds not handled
            continue

        stereo_type = BondStereo.from_bond(bond)
        # make sure end neighbors are shorter in length
        if len(start_neigh) < len(end_neigh):
            temp_neigh = start_neigh
            start_neigh = end_neigh
            end_neigh = temp_neigh

        # case where both have size 1
        if len(start_neigh) == 1:
            match stereo_type:
                # trans, no cis; continue
                case BondStereo.STEREOE:
                    pass
                # cis, not trans; add simple cis
                case BondStereo.STEREOZ:
                    relation_cis = (
                        (start_neigh[0], start_idx, end_idx, end_neigh[0]),
                    )
                    cis_relations.append((relation_cis,))
                # none specified, include trans and cis options
                case BondStereo.STEREONONE | BondStereo.STEREOANY:
                    relation_trans = ()
                    relation_cis = (
                        (start_neigh[0], start_idx, end_idx, end_neigh[0]),
                    )
                    cis_relations.append((relation_trans, relation_cis))
                case _:
                    raise NotImplementedError(
                        f"Case not implemented for {stereo_type}"
                    )
        # case where end has size 1
        elif len(start_neigh) == 2 and len(end_neigh) == 1:  # noqa: PLR2004
            match stereo_type:
                # cis are mapped atoms, all is well
                case BondStereo.STEREOZ:
                    stereo_atoms = tuple(bond.GetStereoAtoms())
                    stereo_start = stereo_atoms[0]
                    stereo_end = stereo_atoms[1]
                    cis_relations.append(
                        (((stereo_start, start_idx, end_idx, stereo_end),),)
                    )
                # trans to mapped atoms, do a swap
                case BondStereo.STEREOE:
                    stereo_atoms = tuple(bond.GetStereoAtoms())
                    stereo_start = tuple(
                        sn for sn in start_neigh if sn not in stereo_atoms
                    )[0]
                    stereo_end = end_neigh[0]
                    cis_relations.append(
                        (((stereo_start, start_idx, end_idx, stereo_end),),)
                    )

                # stereo none, include both possibilities
                case BondStereo.STEREONONE | BondStereo.STEREOANY:
                    cis_relations.append(
                        (
                            (
                                (
                                    start_neigh[0],
                                    start_idx,
                                    end_idx,
                                    end_neigh[0],
                                ),
                            ),
                            (
                                (
                                    start_neigh[1],
                                    start_idx,
                                    end_idx,
                                    end_neigh[0],
                                ),
                            ),
                        ),
                    )
                case _:
                    raise NotImplementedError(
                        f"Case not implemented for {stereo_type}"
                    )
        elif len(end_neigh) == 2:  # noqa: PLR2004
            match stereo_type:
                # case of nondescript: generate two
                case BondStereo.STEREONONE:
                    first_relation = (
                        (start_neigh[0], start_idx, end_idx, end_neigh[0]),
                        (start_neigh[1], start_idx, end_idx, end_neigh[1]),
                    )
                    second_relation = (
                        (start_neigh[1], start_idx, end_idx, end_neigh[0]),
                        (start_neigh[0], start_idx, end_idx, end_neigh[1]),
                    )
                    cis_relations.append((first_relation, second_relation))
                case BondStereo.STEREOE:
                    stereo_atoms = bond.GetStereoAtoms()
                    cis_start1 = tuple(
                        sn for sn in start_neigh if sn in stereo_atoms
                    )[0]
                    cis_end1 = tuple(
                        sn for sn in end_neigh if sn not in stereo_atoms
                    )[0]
                    cis_start2 = tuple(
                        sn for sn in start_neigh if sn not in stereo_atoms
                    )[0]
                    cis_end2 = tuple(
                        sn for sn in end_neigh if sn in stereo_atoms
                    )[0]
                    relation = (
                        (cis_start1, start_idx, end_idx, cis_end1),
                        (cis_start2, start_idx, end_idx, cis_end2),
                    )
                    cis_relations.append((relation,))
                case BondStereo.STEREOZ:
                    stereo_atoms = bond.GetStereoAtoms()
                    cis_start1 = tuple(
                        sn for sn in start_neigh if sn in stereo_atoms
                    )[0]
                    cis_end1 = tuple(
                        sn for sn in end_neigh if sn in stereo_atoms
                    )[0]
                    cis_start2 = tuple(
                        sn for sn in start_neigh if sn not in stereo_atoms
                    )[0]
                    cis_end2 = tuple(
                        sn for sn in end_neigh if sn not in stereo_atoms
                    )[0]
                    relation = (
                        (cis_start1, start_idx, end_idx, cis_end1),
                        (cis_start2, start_idx, end_idx, cis_end2),
                    )
                    cis_relations.append((relation,))
                case _:
                    raise NotImplementedError(
                        f"Case not implemented for {stereo_type}"
                    )
        else:
            raise RuntimeError("If-else logic failure")

    combo: tuple[tuple[tuple[int, int, int, int], ...], ...]
    for combo in itertools.product(*tuple(cis_relations)):
        all_pairs = tuple(
            (x[0], x[1], x[2], x[3])
            for x in (tuple(c) for c in itertools.chain.from_iterable(combo))
        )
        yield all_pairs


def _get_dependent_annotations(
    ring_structures: collections.abc.Sequence[igraph.Graph],
    atom_map: collections.abc.Mapping[int, int],
    atomlabels: collections.abc.Sequence[AtomCore],
) -> collections.abc.Sequence[str | None]:
    ring_labels: list[str | None] = [None for _ in atomlabels]
    for ringstruc in ring_structures:
        atom_idxs = tuple(atom_map[node["name"]] for node in ringstruc.vs)
        atom_labels = tuple(atomlabels[i].as_str() for i in atom_idxs)
        bond_tuples = ringstruc.to_tuple_list()
        for cur_idx, fin_idx in enumerate(atom_idxs):
            cur_labels = list(atom_labels)
            cur_labels[cur_idx] = "!"
            cur_ug = sgrpy.graph.UGraph.from_tuples(bond_tuples, cur_labels)
            cur_uge = sgrpy.graph.UGraphEquiv.from_ugraph(cur_ug).to_ugraph()
            cur_ring_ann = cur_uge.to_sugraph().as_str()
            ring_labels[fin_idx] = cur_ring_ann
    return ring_labels


def get_invariant_annotations(
    complexes: collections.abc.Sequence[igraph.Graph],
    atomlabels: list[AtomLabel | CisLabel],
) -> list[str | None]:
    label_list: list[str | None] = [None for _ in atomlabels]
    for ext_complex in complexes:
        atom_idxs: tuple[int, ...] = tuple(
            node["name"] for node in ext_complex.vs
        )
        atom_labels = tuple(
            "cis"
            if isinstance(atomlabel, CisLabel)
            else atomlabel.invariants.as_str()
            for atomlabel in (atomlabels[i] for i in atom_idxs)
        )
        bond_tuples = ext_complex.to_tuple_list()
        for cur_idx, fin_idx in enumerate(atom_idxs):
            cur_labels = list(atom_labels)
            cur_labels[cur_idx] = "!"
            cur_ug = sgrpy.graph.UGraph.from_tuples(bond_tuples, cur_labels)
            cur_uge = sgrpy.graph.UGraphEquiv.from_ugraph(cur_ug).to_ugraph()
            cur_ring_ann = cur_uge.to_sugraph().as_str()
            label_list[fin_idx] = cur_ring_ann

    return label_list


def get_complex_graphs(
    complexes: collections.abc.Sequence[igraph.Graph],
    atomlabels: list[AtomLabel | CisLabel],
    use_core: bool,
) -> list[str]:
    label_list: list[str] = []
    for ext_complex in complexes:
        atom_idxs: tuple[int, ...] = tuple(
            node["name"] for node in ext_complex.vs
        )
        atom_labels = tuple(
            "cis"
            if isinstance(atomlabel, CisLabel)
            else (
                atomlabel.invariants.as_str()
                if use_core
                else no_extras(atomlabel).as_str()
            )
            for atomlabel in (atomlabels[i] for i in atom_idxs)
        )
        bond_tuples = ext_complex.to_tuple_list()
        cur_labels = list(atom_labels)
        cur_ug = sgrpy.graph.UGraph.from_tuples(bond_tuples, cur_labels)
        cur_uge = sgrpy.graph.UGraphEquiv.from_ugraph(cur_ug).to_ugraph()
        cur_ring_ann = cur_uge.to_sugraph().as_str()
        label_list.append(cur_ring_ann)

    return label_list


def _trim_ugraph(
    ugraph: sgrpy.graph.UGraph, trim_set: collections.abc.Collection[int]
) -> sgrpy.graph.UGraph:
    n_items = len(ugraph.get_labels())
    vert_map = {
        i_orig: i
        for i, i_orig in enumerate(
            j for j in range(n_items) if j not in trim_set
        )
    }
    labels = (
        label
        for i, label in enumerate(ugraph.get_labels())
        if i not in trim_set
    )
    bonds = (
        (vert_map[bond.src], vert_map[bond.trg])
        for bond in ugraph.get_bonds()
        if bond.src not in trim_set and bond.trg not in trim_set
    )
    new_ugraph = sgrpy.graph.UGraph.from_tuples(bonds=bonds, labels=labels)
    return new_ugraph


def find_ring_bonds(
    mol: rdkit.Chem.rdchem.Mol, atom_map: dict[int, int]
) -> frozenset[sgrpy.graph.Bond]:
    raw_indices = (
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.IsInRing()
    )
    mapped_indices = (
        (atom_map[begin_idx], atom_map[end_idx])
        for begin_idx, end_idx in raw_indices
    )
    bond_gen = (
        sgrpy.graph.Bond.new(begin_idx, end_idx)
        for begin_idx, end_idx in mapped_indices
    )
    return frozenset(bond_gen)


def find_conj_bonds(
    mol: rdkit.Chem.rdchem.Mol, atom_map: dict[int, int]
) -> frozenset[sgrpy.graph.Bond]:
    raw_indices = (
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetIsConjugated()
    )
    mapped_indices = (
        (atom_map[begin_idx], atom_map[end_idx])
        for begin_idx, end_idx in raw_indices
    )
    bond_gen = (
        sgrpy.graph.Bond.new(begin_idx, end_idx)
        for begin_idx, end_idx in mapped_indices
    )
    return frozenset(bond_gen)


def get_complexes(
    bond_set: collections.abc.Iterable[sgrpy.graph.Bond],
) -> collections.abc.Iterable[igraph.Graph]:
    graph: igraph.Graph = igraph.Graph.TupleList(
        [bond.as_tuple() for bond in bond_set]
    )

    comp: list[int]
    for comp in graph.components():
        sg = graph.induced_subgraph(comp)
        yield sg


def _get_rings(
    mol: rdkit.Chem.rdchem.Mol,
    cislabels: bool,
    use_core: bool,
) -> list[list[str]]:
    """Return ring list."""
    bonds: set[sgrpy.graph.Bond] = set()
    trim_ends = False

    # create atom map
    atom: rdkit.Chem.rdchem.Atom
    bond: rdkit.Chem.rdchem.Bond
    atom_map: dict[int, int] = {}
    atom_str_labels: list[str] = []
    atom_labels: list[AtomLabel] = []
    term_atoms: set[int] = set()

    # iterate through atoms
    for i, atom in enumerate(mol.GetAtoms()):
        atom_map[atom.GetIdx()] = i
        atom_label = AtomLabel.from_atom(atom)
        atom_str_labels.append(atom_label.as_str())
        atom_labels.append(atom_label)
        if trim_ends and atom_label.is_terminal():
            term_atoms.add(i)
            continue

    # iterate through bonds
    for bond in mol.GetBonds():
        src: int = atom_map[bond.GetBeginAtomIdx()]
        if src in term_atoms:
            continue
        trg: int = atom_map[bond.GetEndAtomIdx()]
        if trg in term_atoms:
            continue
        sgrpy_bond = sgrpy.graph.Bond.new(src, trg)
        bonds.add(sgrpy_bond)

    # detect ring bonds if enabled
    ring_bonds: frozenset[sgrpy.graph.Bond] = find_ring_bonds(mol, atom_map)

    ring_communities = tuple(
        get_ring_complexes(
            list(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in mol.GetBonds()
                if (bond.IsInRing())
            )
        )
    )
    ringlabels_ext = _get_dependent_annotations(
        ring_communities,
        atom_map,
        tuple(a.invariants for a in atom_labels),
    )
    for i, ringlabel_ext in enumerate(ringlabels_ext):
        if ringlabel_ext is None:
            continue
        atom_labels[i] = dataclasses.replace(
            atom_labels[i],
            _extra=ringlabel_ext,
        )
        atom_str_labels[i] = atom_labels[i].as_str()

    all_cis_labels: collections.abc.Iterable[
        tuple[tuple[int, int, int, int], ...]
    ] = get_cis_labels(mol) if cislabels else ((),)
    graph_list: list[list[str]] = []

    for cis_labels in all_cis_labels:
        labels_copy: list[AtomLabel | CisLabel] = list(atom_labels)
        bond_copy = set(bonds)
        ring_bonds_copy = set(ring_bonds)
        # should be a unique mapping for each of these iterations
        for cis_start_a, cis_start_db, cis_end_db, cis_end_a in cis_labels:
            cis_start = atom_map[cis_start_a]
            cis_startdb = atom_map[cis_start_db]
            cis_enddb = atom_map[cis_end_db]
            cis_end = atom_map[cis_end_a]
            atomlabel1 = atom_labels[cis_start]
            atomlabel2 = atom_labels[cis_start_db]
            atomlabel3 = atom_labels[cis_end_db]
            atomlabel4 = atom_labels[cis_end]
            atomcore1 = atomlabel1.invariants
            atomcore2 = atomlabel2.invariants
            atomcore3 = atomlabel3.invariants
            atomcore4 = atomlabel4.invariants
            cislabel = CisLabel.from_cores(
                atomcore1, atomcore2, atomcore3, atomcore4
            )
            cis_index = len(labels_copy)
            labels_copy.append(cislabel)
            bond1 = sgrpy.graph.Bond.new(cis_start, cis_index)
            bond2 = sgrpy.graph.Bond.new(cis_index, cis_end)
            bond3 = sgrpy.graph.Bond.new(cis_startdb, cis_index)
            bond4 = sgrpy.graph.Bond.new(cis_enddb, cis_index)
            bond_copy.add(bond1)
            bond_copy.add(bond2)
            bond_copy.add(bond3)
            bond_copy.add(bond4)

            # check for participation in ring or conjugated systems
            main_cis_bond = sgrpy.graph.Bond.new(cis_start_db, cis_end_db)
            if main_cis_bond in ring_bonds_copy:
                ring_bonds_copy.add(bond3)
                ring_bonds_copy.add(bond4)

        # get ring complexes
        ring_complexes = list(get_complexes(ring_bonds_copy))
        all_rings = get_complex_graphs(
            ring_complexes, labels_copy, use_core=use_core
        )
        graph_list.append(all_rings)
        continue
    return graph_list


def _get_conjs(
    mol: rdkit.Chem.rdchem.Mol,
    cislabels: bool,
    use_core: bool,
) -> list[list[str]]:
    """Return conj list."""
    trim_ends = False
    bonds: set[sgrpy.graph.Bond] = set()

    # create atom map
    atom: rdkit.Chem.rdchem.Atom
    bond: rdkit.Chem.rdchem.Bond
    atom_map: dict[int, int] = {}
    atom_str_labels: list[str] = []
    atom_labels: list[AtomLabel] = []
    term_atoms: set[int] = set()

    # iterate through atoms
    for i, atom in enumerate(mol.GetAtoms()):
        atom_map[atom.GetIdx()] = i
        atom_label = AtomLabel.from_atom(atom)
        atom_str_labels.append(atom_label.as_str())
        atom_labels.append(atom_label)
        if trim_ends and atom_label.is_terminal():
            term_atoms.add(i)
            continue

    # iterate through bonds
    for bond in mol.GetBonds():
        src: int = atom_map[bond.GetBeginAtomIdx()]
        if src in term_atoms:
            continue
        trg: int = atom_map[bond.GetEndAtomIdx()]
        if trg in term_atoms:
            continue
        sgrpy_bond = sgrpy.graph.Bond.new(src, trg)
        bonds.add(sgrpy_bond)

    # detect conjugated bonds if enabled
    conj_bonds: frozenset[sgrpy.graph.Bond] = find_conj_bonds(mol, atom_map)

    ring_communities = tuple(
        get_ring_complexes(
            list(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                for bond in mol.GetBonds()
                if bond.GetIsConjugated()
            )
        )
    )
    ringlabels_ext = _get_dependent_annotations(
        ring_communities,
        atom_map,
        tuple(a.invariants for a in atom_labels),
    )
    for i, ringlabel_ext in enumerate(ringlabels_ext):
        if ringlabel_ext is None:
            continue
        atom_labels[i] = dataclasses.replace(
            atom_labels[i],
            _extra=ringlabel_ext,
        )
        atom_str_labels[i] = atom_labels[i].as_str()

    all_cis_labels: collections.abc.Iterable[
        tuple[tuple[int, int, int, int], ...]
    ] = get_cis_labels(mol) if cislabels else ((),)
    graph_list: list[list[str]] = []

    for cis_labels in all_cis_labels:
        labels_copy: list[AtomLabel | CisLabel] = list(atom_labels)
        bond_copy = set(bonds)
        conj_bonds_copy = set(conj_bonds)
        # should be a unique mapping for each of these iterations
        for cis_start_a, cis_start_db, cis_end_db, cis_end_a in cis_labels:
            cis_start = atom_map[cis_start_a]
            cis_startdb = atom_map[cis_start_db]
            cis_enddb = atom_map[cis_end_db]
            cis_end = atom_map[cis_end_a]
            atomlabel1 = atom_labels[cis_start]
            atomlabel2 = atom_labels[cis_start_db]
            atomlabel3 = atom_labels[cis_end_db]
            atomlabel4 = atom_labels[cis_end]
            atomcore1 = atomlabel1.invariants
            atomcore2 = atomlabel2.invariants
            atomcore3 = atomlabel3.invariants
            atomcore4 = atomlabel4.invariants
            cislabel = CisLabel.from_cores(
                atomcore1, atomcore2, atomcore3, atomcore4
            )
            cis_index = len(labels_copy)
            labels_copy.append(cislabel)
            bond1 = sgrpy.graph.Bond.new(cis_start, cis_index)
            bond2 = sgrpy.graph.Bond.new(cis_index, cis_end)
            bond3 = sgrpy.graph.Bond.new(cis_startdb, cis_index)
            bond4 = sgrpy.graph.Bond.new(cis_enddb, cis_index)
            bond_copy.add(bond1)
            bond_copy.add(bond2)
            bond_copy.add(bond3)
            bond_copy.add(bond4)

            # check for participation in ring or conjugated systems
            main_cis_bond = sgrpy.graph.Bond.new(cis_start_db, cis_end_db)
            if main_cis_bond in conj_bonds_copy:
                conj_bonds_copy.add(bond3)
                conj_bonds_copy.add(bond4)

        # get ring complexes
        # ring_complexes = list(get_complexes(ring_bonds_copy))
        # ring_info = get_invariant_annotations(ring_complexes, labels_copy)

        # get conj complexes
        conj_complexes = list(get_complexes(conj_bonds_copy))
        all_conjs = get_complex_graphs(
            conj_complexes, labels_copy, use_core=use_core
        )
        graph_list.append(all_conjs)
        continue
    return graph_list


def convert_benson(
    mol: rdkit.Chem.rdchem.Mol,
    cislabels: bool = False,
    trim_ends: bool = True,
    extended_ringlabels: bool = False,
    extended_conjlabels: bool = False,
) -> None | tuple[sgrpy.graph.UGraph, ...]:
    """Convert RDKit molecule to UGraph form.

    Parameters
    ----------
    mol : Mol
       RDKit molecule to be converted.
    stereo: bool
        Include cis labels across double bonds.

    Returns
    -------
    None | tuple[UGraph, ...]
       If conversion failed, return None; else returns a tuple of possible graph
       conversions (multiple are sometimes possible).
    """
    bonds: set[sgrpy.graph.Bond] = set()

    # create atom map
    atom: rdkit.Chem.rdchem.Atom
    bond: rdkit.Chem.rdchem.Bond
    atom_map: dict[int, int] = {}
    atom_str_labels: list[str] = []
    atom_labels: list[AtomLabel] = []
    term_atoms: set[int] = set()

    # iterate through atoms
    for i, atom in enumerate(mol.GetAtoms()):
        atom_map[atom.GetIdx()] = i
        atom_label = AtomLabel.from_atom(atom)
        atom_str_labels.append(atom_label.as_str())
        atom_labels.append(atom_label)
        if trim_ends and atom_label.is_terminal():
            term_atoms.add(i)
            continue

    # iterate through bonds
    for bond in mol.GetBonds():
        src: int = atom_map[bond.GetBeginAtomIdx()]
        if src in term_atoms:
            continue
        trg: int = atom_map[bond.GetEndAtomIdx()]
        if trg in term_atoms:
            continue
        sgrpy_bond = sgrpy.graph.Bond.new(src, trg)
        bonds.add(sgrpy_bond)

    # detect ring bonds if enabled
    ring_bonds: frozenset[sgrpy.graph.Bond] = (
        find_ring_bonds(mol, atom_map) if extended_ringlabels else frozenset()
    )

    # detect conjugated bonds if enabled
    conj_bonds: frozenset[sgrpy.graph.Bond] = (
        find_conj_bonds(mol, atom_map) if extended_conjlabels else frozenset()
    )

    if extended_ringlabels or extended_conjlabels:
        ring_communities = tuple(
            get_ring_complexes(
                list(
                    (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    for bond in mol.GetBonds()
                    if (extended_ringlabels and bond.IsInRing())
                    or (extended_conjlabels and bond.GetIsConjugated())
                )
            )
        )
        ringlabels_ext = _get_dependent_annotations(
            ring_communities,
            atom_map,
            tuple(a.invariants for a in atom_labels),
        )
        for i, ringlabel_ext in enumerate(ringlabels_ext):
            if ringlabel_ext is None:
                continue
            atom_labels[i] = dataclasses.replace(
                atom_labels[i],
                _extra=ringlabel_ext,
            )
            atom_str_labels[i] = atom_labels[i].as_str()

    all_cis_labels: collections.abc.Iterable[
        tuple[tuple[int, int, int, int], ...]
    ] = get_cis_labels(mol) if cislabels else ((),)
    graph_list: list[sgrpy.graph.UGraph] = []

    for cis_labels in all_cis_labels:
        labels_copy: list[AtomLabel | CisLabel] = list(atom_labels)
        bond_copy = set(bonds)
        ring_bonds_copy = set(ring_bonds)
        conj_bonds_copy = set(conj_bonds)
        # should be a unique mapping for each of these iterations
        for cis_start_a, cis_start_db, cis_end_db, cis_end_a in cis_labels:
            cis_start = atom_map[cis_start_a]
            cis_startdb = atom_map[cis_start_db]
            cis_enddb = atom_map[cis_end_db]
            cis_end = atom_map[cis_end_a]
            atomlabel1 = atom_labels[cis_start]
            atomlabel2 = atom_labels[cis_start_db]
            atomlabel3 = atom_labels[cis_end_db]
            atomlabel4 = atom_labels[cis_end]
            atomcore1 = atomlabel1.invariants
            atomcore2 = atomlabel2.invariants
            atomcore3 = atomlabel3.invariants
            atomcore4 = atomlabel4.invariants
            cislabel = CisLabel.from_cores(
                atomcore1, atomcore2, atomcore3, atomcore4
            )
            cis_index = len(labels_copy)
            labels_copy.append(cislabel)
            bond1 = sgrpy.graph.Bond.new(cis_start, cis_index)
            bond2 = sgrpy.graph.Bond.new(cis_index, cis_end)
            bond3 = sgrpy.graph.Bond.new(cis_startdb, cis_index)
            bond4 = sgrpy.graph.Bond.new(cis_enddb, cis_index)
            bond_copy.add(bond1)
            bond_copy.add(bond2)
            bond_copy.add(bond3)
            bond_copy.add(bond4)

            # check for participation in ring or conjugated systems
            main_cis_bond = sgrpy.graph.Bond.new(cis_start_db, cis_end_db)
            if main_cis_bond in ring_bonds_copy:
                ring_bonds_copy.add(bond3)
                ring_bonds_copy.add(bond4)
            if main_cis_bond in conj_bonds_copy:
                conj_bonds_copy.add(bond3)
                conj_bonds_copy.add(bond4)

        # get ring complexes
        ring_complexes = list(get_complexes(ring_bonds_copy))
        ring_info = get_invariant_annotations(ring_complexes, labels_copy)

        # get conj complexes
        conj_complexes = list(get_complexes(conj_bonds_copy))
        conj_info = get_invariant_annotations(conj_complexes, labels_copy)

        for i, (cur_ring_info, cur_conj_info) in enumerate(
            zip(ring_info, conj_info, strict=True)
        ):
            match cur_ring_info, cur_conj_info:
                case None, None:
                    continue
                case None, str():
                    labels_copy[i] = dataclasses.replace(
                        labels_copy[i], _extra=f"conj({cur_conj_info})"
                    )
                case str(), None:
                    labels_copy[i] = dataclasses.replace(
                        labels_copy[i], _extra=f"ring({cur_ring_info})"
                    )
                case str(), str():
                    labels_copy[i] = dataclasses.replace(
                        labels_copy[i],
                        _extra=f"ring({cur_ring_info}),conj({cur_conj_info})",
                    )

        new_graph = sgrpy.graph.UGraph.from_bonds(
            sorted(bond_copy), (lab.as_str() for lab in labels_copy)
        )
        if trim_ends:
            new_graph = _trim_ugraph(new_graph, trim_set=term_atoms)
        graph_list.append(new_graph)
    return tuple(graph_list)
