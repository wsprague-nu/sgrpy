"""Test Benson V3 code."""

import pathlib

import pytest
import rdkit.Chem.Draw

import sgrpy

from .._benson_v3 import convert_mol_general, get_cis_labels_int
from .._load_smiles import load_smiles


def __cislabels(
    smiles: str,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    mol = load_smiles(smiles)
    if mol is None:
        return ()
    result = tuple(
        sorted(
            tuple(sorted(z.as_tuple_sort() for z in y))
            for y in get_cis_labels_int(mol)
        )
    )
    return result


def __attempt_cisconversion(
    smiles: str,
) -> None | tuple[sgrpy.graph.UGraph, ...]:
    mol = load_smiles(smiles)

    if mol is None:
        return None

    result = convert_mol_general(mol, node_cis=True)

    final_result = (
        result if result is None else tuple(y.to_ugraph() for y in result)
    )

    return final_result


def __draw_output(smiles: str, path: str) -> None:
    mol = load_smiles(smiles)
    if mol is None:
        return
    drawer = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.drawOptions().addAtomIndices = True  # type: ignore[assignment,unused-ignore]
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    image = drawer.GetDrawingText()
    with open(path, "w") as fout:
        fout.write(image)


def test_cis_labels_v3() -> None:
    """Test to make sure cis labeling scheme works as intended."""
    assert __cislabels(r"CCC") == ((),)
    assert __cislabels(r"CC=O") == ((),)
    assert __cislabels(r"CCC(C)=NC") == (((1, 2, 4, 5),), ((3, 2, 4, 5),))
    assert __cislabels(r"CC/C(C)=N/C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"CC/C(C)=N\C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"CC\C(C)=N/C") == (((1, 2, 4, 5),),)
    assert __cislabels(r"CC\C(C)=N\C") == (((3, 2, 4, 5),),)
    assert __cislabels(r"C/C(CC)=N/C") == (((2, 1, 4, 5),),)
    assert __cislabels(r"C/C(CC)=N\C") == (((0, 1, 4, 5),),)
    assert __cislabels(r"C\C(CC)=N/C") == (((0, 1, 4, 5),),)
    assert __cislabels(r"C\C(CC)=N\C") == (((2, 1, 4, 5),),)
    assert __cislabels(r"OC(Cl)=C(F)C") == (
        ((0, 1, 3, 4), (2, 1, 3, 5)),
        ((0, 1, 3, 5), (2, 1, 3, 4)),
    )
    assert __cislabels(r"O/C(Cl)=C(/F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
    assert __cislabels(r"O/C(Cl)=C(\F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"O\C(Cl)=C(/F)C") == (((0, 1, 3, 4), (2, 1, 3, 5)),)
    assert __cislabels(r"O\C(Cl)=C(\F)C") == (((0, 1, 3, 5), (2, 1, 3, 4)),)
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
    assert __cislabels(r"C/C=C/C(C)(C)/C=C\C") == (((3, 6, 7, 8),),)
    assert __cislabels(r"C/C=C\C(C)(C)/C=C/C") == (((0, 1, 2, 3),),)
    assert __cislabels(r"C=C=CC") == ((),)
    assert __cislabels(r"C1CCCCC1") == ((),)
    assert __cislabels(r"C1CCCCC=C1") == ((),)
    assert __cislabels(r"CC[C]=CC") == ((), ((1, 2, 3, 4),))
    assert __cislabels(r"C1CCCCC/C=C/1") == ((),)
    assert __cislabels(r"C1CCCCC/C=C\1") == (((0, 7, 6, 5),),)

    with pytest.raises(NotImplementedError):
        __cislabels(r"C1CCCCCC=C1")


def test_cis_conversion_v3() -> None:
    """Test to make sure V3 cis labeling scheme works as intended."""
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


def test_ring_detection_v3_1() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = "Cc1ccccc1C"
    canon_mol = load_smiles(smiles)
    assert canon_mol is not None
    ugc = convert_mol_general(canon_mol, ext_ring=True)
    assert ugc is not None
    ug = tuple(y.to_ugraph() for y in ugc)
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


def test_ring_detection_v3_2() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = "C1CC2CCC1C2"
    canon_mol = load_smiles(smiles)
    assert canon_mol is not None
    ugc = convert_mol_general(canon_mol, ext_ring=True)
    assert ugc is not None
    ug = tuple(y.to_ugraph() for y in ugc)
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


def test_ring_detection_v3_3() -> None:
    """Test to make sure ring detection scheme works as intended."""
    smiles = r"C1CCC/C=C\CC1"
    canon_mol = load_smiles(smiles)
    assert canon_mol is not None

    ugc = convert_mol_general(canon_mol, node_cis=True, ext_ring=True)
    assert ugc is not None
    ug = tuple(y.to_ugraph() for y in ugc)
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
    assert labels[0] == gamma_label
    assert labels[1] == gamma_label
    assert labels[2] == beta_label
    assert labels[3] == alpha_label
    assert labels[4] == db_label
    assert labels[5] == db_label
    assert labels[6] == alpha_label
    assert labels[7] == beta_label
    assert labels[8] == cis_label
