"""Test extended Benson shim."""

import pathlib

import rdkit.Chem.rdchem
import rdkit.Chem.rdmolfiles

import sgrpy

from .._benson_v2 import convert_benson, get_cis_labels


def __mol_from_smiles(smiles: str) -> rdkit.Chem.rdchem.Mol:
    mol = rdkit.Chem.rdmolfiles.MolFromSmiles(smiles)
    canon_smiles = rdkit.Chem.rdmolfiles.MolToSmiles(mol)
    canon_mol = rdkit.Chem.rdmolfiles.MolFromSmiles(canon_smiles)
    return canon_mol


def __attempt_cisconversion(
    smiles: str,
) -> None | tuple[sgrpy.graph.UGraph, ...]:
    mol = __mol_from_smiles(smiles)

    result = convert_benson(mol, cislabels=True)

    return result


def __cislabels(smiles: str) -> tuple[tuple[tuple[int, ...], ...], ...]:
    mol = __mol_from_smiles(smiles)
    result = get_cis_labels(mol)
    return_result = tuple(
        sorted(tuple(sorted(tuple((z for z in y)) for y in x)) for x in result)
    )
    return return_result


def test_cis_labels() -> None:
    """Test to make sure cis labeling scheme works as intended."""
    assert __cislabels(r"CCC") == ((),)
    assert __cislabels(r"CC=O") == ((),)
    assert __cislabels(r"CCC(C)=NC") == (((1, 2, 4, 5),), ((3, 2, 4, 5),))
    assert __cislabels(r"CC/C(C)=N/C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"CC/C(C)=N\C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"CC\C(C)=N/C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"CC\C(C)=N\C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"C/C(CC)=N/C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"C/C(CC)=N\C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"C\C(CC)=N/C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"C\C(CC)=N\C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"OC(Cl)=C(F)C") == (
        ((0, 1, 3, 4), (2, 1, 3, 5)),
        ((0, 1, 3, 5), (2, 1, 3, 4)),
    )
    assert __cislabels(r"O/C(Cl)=C(/F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"O/C(Cl)=C(\F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
    assert __cislabels(r"O\C(Cl)=C(/F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
    assert __cislabels(r"O\C(Cl)=C(\F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"Cl/C(O)=C(/F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
    assert __cislabels(r"Cl/C(O)=C(\F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"Cl\C(O)=C(/F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"Cl\C(O)=C(\F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
    assert __cislabels(r"C/C=C/C") == ((),)
    assert __cislabels(r"C/C=C\C") == (((0, 1, 2, 3),),)
    assert __cislabels(r"CC=CC") == ((), ((0, 1, 2, 3),))
    assert __cislabels(r"CC=CC(C)(C)C=CC") == (
        (),
        ((0, 1, 2, 3),),
        ((0, 1, 2, 3), (3, 6, 7, 8)),
        ((3, 6, 7, 8),),
    )
    assert __cislabels(r"C/C=C/C(C)(C)/C=C/C") == ((),)
    assert __cislabels(r"C/C=C\C(C)(C)/C=C\C") == (
        ((0, 1, 2, 3), (3, 6, 7, 8)),
    )
    assert __cislabels(r"C/C=C/C(C)(C)/C=C\C") == (((0, 1, 2, 3),),)
    assert __cislabels(r"C/C=C\C(C)(C)/C=C/C") == (((0, 1, 2, 3),),)
    assert __cislabels(r"C=C=CC") == ((),)
    assert __cislabels(r"C1CCCCC1") == ((),)
    assert __cislabels(r"C1CCCCC=C1") == ((),)
    assert __cislabels(r"CC[C]=CC") == ((), ((0, 1, 2, 3),))


def test_cis_conversion() -> None:
    """Test to make sure cis conversion scheme works as intended."""
    with open(pathlib.Path(__file__).parent / "benson_simple.txt", "r") as fin:
        tests = tuple(
            (i, o) for i, o in (line.strip().split("\t") for line in fin)
        )
    for in_val, out_val in tests:
        result_val = __attempt_cisconversion(in_val)
        assert result_val is not None
        result_uges = tuple(
            sgrpy.graph.UGraphEquiv.from_ugraph(result_val_t).to_ugraph()
            for result_val_t in result_val
        )
        result_uge_strs = tuple(
            result_uge.to_sugraph().as_str() for result_uge in result_uges
        )
        out_vals = out_val.lstrip("('").rstrip("',)").split("', '")
        out_ugs = tuple(
            sgrpy.graph.SUGraph.from_str(out_str).to_ugraph()
            for out_str in out_vals
        )
        out_uge = tuple(
            sgrpy.graph.UGraphEquiv.from_ugraph(out_ug).to_ugraph()
            for out_ug in out_ugs
        )
        out_uge_str = tuple(
            out_uge_s.to_sugraph().as_str() for out_uge_s in out_uge
        )
        sorted_results = sorted(result_uge_strs)
        sorted_out = sorted(out_uge_str)
        assert sorted_results == sorted_out


def test_ring_detection_1() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = "Cc1ccccc1C"
    canon_mol = __mol_from_smiles(smiles)
    ug = convert_benson(canon_mol, extended_ringlabels=True)
    end_label = "(6,4,4,0,0,4,False,0),3,(((1,),(6,4,3,0,0,3,True,0)))"
    ring_label = (
        "(6,4,3,0,0,3,True,0),1,(((4,),(6"
        ",4,3,0,0,3,True,0)),((4,),(6,4,3,0,0,3,True,0))),extra(ring(1,3,2,3,"
        '0,4,1,4,0,5,2,5;["!","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0","'
        '6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0"'
        "]))"
    )
    ring_end_label = (
        "(6,4,3,0,0,3,True,0),0,(((1,),(6,4,4,0,0,4,False,0)),((4,),(6"
        ",4,3,0,0,3,True,0)),((4,),(6,4,3,0,0,3,True,0))),extra(ring(1,3,2,3,"
        '0,4,1,4,0,5,2,5;["!","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0","'
        '6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0","6,4,3,0,0,3,True,0"'
        "]))"
    )
    assert ug is not None
    assert len(ug) == 1
    labels = ug[0].get_labels()
    assert labels[0] == end_label
    assert labels[1] == ring_end_label
    assert labels[2] == ring_label
    assert labels[3] == ring_label
    assert labels[4] == ring_label
    assert labels[5] == ring_label
    assert labels[6] == ring_end_label
    assert labels[7] == end_label


def test_ring_detection_2() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = "C1CC2CCC1C2"
    canon_mol = __mol_from_smiles(smiles)
    ug = convert_benson(canon_mol, extended_ringlabels=True)
    assert ug is not None
    assert len(ug) == 1
    bridge_label = (
        "(6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),"
        "(6"
        ',4,4,0,0,4,False,0))),extra(ring(1,2,3,4,0,5,1,5,3,5,0,6,2,6,4,6;["!","6'
        ',4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,'
        '0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,'
        'False,0"]))'
    )
    junction_label = (
        "(6,4,4,0,0,4,False,0),1,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6"
        ",4,4,0,0,4,False,0)),((1,),(6,4,4,0,0,4,False,0))),extra(ring(0,1,0,"
        '2,1,3,2,4,0,5,3,6,4,6,5,6;["!","6,4,4,0,0,4,False,0","6,4,4,0,0,4,'
        'False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0'
        ',0,4,False,0","6,4,4,0,0,4,False,0"]))'
    )
    loop_label = (
        "(6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False,0)),((1,),(6"
        ',4,4,0,0,4,False,0))),extra(ring(1,3,0,4,1,5,2,5,4,5,0,6,2,6,3,6;["!","6'
        ',4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,'
        '0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,'
        'False,0"]))'
    )
    labels = ug[0].get_labels()
    assert labels[0] == loop_label
    assert labels[1] == loop_label
    assert labels[2] == junction_label
    assert labels[3] == loop_label
    assert labels[4] == loop_label
    assert labels[5] == junction_label
    assert labels[6] == bridge_label


def test_ring_detection_3() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = r"C1CCC/C=C\CC1"
    canon_mol = __mol_from_smiles(smiles)
    ug = convert_benson(canon_mol, cislabels=True, extended_ringlabels=True)
    db_label = (
        "(6,4,3,0,0,3,False,0),1,(((1,),(6,4,4,0,0,4,False,0))"
        ",((2,),(6,4,3,0,0,3,False,0))),extra(ring(0,1,0,2,2,3,3,"
        '4,4,5,5,6,1,7,6,7,0,8,1,8;["!","6,4,3,0,0,3,False,0","6,4,'
        '4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,'
        'False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6'
        ',4,4,0,0,4,False,0","cis"]))'
    )
    alpha_label = (
        "(6,4,4,0,0,4,False,0),2,(((1,),(6,4,3,0,0,3,False"
        ",0)),((1,),(6,4,4,0,0,4,False,0))),extra(ring(0,2,1,"
        '2,1,3,3,4,4,5,5,6,0,7,6,7,1,8,2,8;["!","6,4,3,0,0,3,Fa'
        'lse,0","6,4,3,0,0,3,False,0","6,4,4,0,0,4,False,0",'
        '"6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,'
        '0,0,4,False,0","6,4,4,0,0,4,False,0","cis"]))'
    )
    beta_label = (
        "(6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,False"
        ",0)),((1,),(6,4,4,0,0,4,False,0))),extra(ring(1,2,3,"
        '4,1,5,4,5,0,6,3,6,0,7,2,7,1,8,2,8;["!","6,4,3,0,0,3,Fa'
        'lse,0","6,4,3,0,0,3,False,0","6,4,4,0,0,4,False,0",'
        '"6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,'
        '0,0,4,False,0","6,4,4,0,0,4,False,0","cis"]))'
    )
    gamma_label = (
        "(6,4,4,0,0,4,False,0),2,(((1,),(6,4,4,0,0,4,"
        "False,0)),((1,),(6,4,4,0,0,4,False,0))),extra(r"
        'ing(1,2,1,4,2,5,3,5,0,6,4,6,0,7,3,7,1,8,2,8;["!","6,'
        '4,3,0,0,3,False,0","6,4,3,0,0,3,False,0","6,4,'
        '4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,'
        '0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,'
        '0,4,False,0","cis"]))'
    )
    cis_label = (
        "cis,(6,4,4,0,0,4,False,0),(6,4,3,0,0,3,False,0),(6,4,"
        "3,0,0,3,False,0),(6,4,4,0,0,4,False,0),extra(ring(0,1"
        ',0,2,1,2,3,5,4,6,5,6,1,7,3,7,2,8,4,8;["!","6,4,3,0,0,3,F'
        'alse,0","6,4,3,0,0,3,False,0","6,4,4,0,0,4,False,0","'
        '6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0'
        ',4,False,0","6,4,4,0,0,4,False,0","6,4,4,0,0,4,False,0"]))'
    )
    assert ug is not None
    assert len(ug) == 1
    labels = ug[0].get_labels()
    assert len(labels) == 9  # noqa: PLR2004
    assert labels[0] == db_label
    assert labels[1] == db_label
    assert labels[2] == alpha_label
    assert labels[3] == beta_label
    assert labels[4] == gamma_label
    assert labels[5] == gamma_label
    assert labels[6] == beta_label
    assert labels[7] == alpha_label
    assert labels[8] == cis_label
