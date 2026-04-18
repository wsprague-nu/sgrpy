"""Test Benson v4 routines."""

import dataclasses
import json
import pathlib

import numpy
import rdkit.Chem.rdChemReactions
import rdkit.Chem.rdmolops

from sgrpy.graph import CanonType, TGraph
from sgrpy.iotypes import JSON

from .._benson_v4 import (
    BensonCore,
    BensonLabel,
    ConvertResult_v4,
    EdgeData,
    ParsingError,
    map_benson_core_v4,
    map_core_rdkit_v4,
)
from .._load_smiles import load_smiles

__SELF_DIR = pathlib.Path(__file__).parent

# TODO: add test cases for rings, conjugation, and when cis nodes are included
# (should have all four other nodes detected as part of structure)


def __write_json(data: JSON) -> None:
    with open("random.json", "w") as fout:
        json.dump(data, fout)


def __graph_to_json(
    graph: TGraph[BensonCore, EdgeData] | TGraph[BensonLabel, EdgeData],
) -> JSON:
    labels = list(
        lab.label.as_str() if isinstance(lab, BensonLabel) else lab.as_str()
        for lab in graph.get_labels()
    )
    edges: list[list[int | str]] = list(
        [edge.src, edge.trg, edge.color.as_str()] for edge in graph.get_bonds()
    )
    return {"labels": labels, "edges": edges}


def __replace_result(result: ConvertResult_v4) -> ConvertResult_v4:
    tgraph = result.graph
    tgraph_can, newmap = tgraph.canonize(CanonType.F)
    return dataclasses.replace(
        result, graph=tgraph_can, mapping=newmap.compose(result.mapping)
    )


def __convert_to_json(result: ConvertResult_v4) -> JSON:
    labels = list(lab.label.as_str() for lab in result.graph.get_labels())
    edges: list[list[int | str]] = list(
        [edge.src, edge.trg, edge.color.as_str()]
        for edge in result.graph.get_bonds()
    )
    extras = {
        ex_key: [__graph_to_json(tg) for tg in ex_val]
        for ex_key, ex_val in result.extras.items()
    }
    return {"labels": labels, "edges": edges, "extras": extras}


def test_conversions() -> None:
    """Test molecule conversions."""
    rng = numpy.random.default_rng(5795)

    with open(__SELF_DIR / "v4_data.json") as fin:
        test_data = json.load(fin)

    for test_entry in test_data["examples"]:
        smiles = test_entry["smiles"]
        node_cis = test_entry["params"]["node_cis"]
        ext_ring = test_entry["params"]["ext_ring"]
        ext_conj = test_entry["params"]["ext_conj"]
        feat_ring = test_entry["params"]["feat_ring"]
        feat_conj = test_entry["params"]["feat_conj"]
        trim_ends = test_entry["params"]["trim_ends"]
        test_results = test_entry["data"]

        mol = load_smiles(smiles)
        if mol is None:
            assert test_results is None
            continue

        for _ in range(8):
            nof_atoms: int = mol.GetNumHeavyAtoms()
            shuff_arr = numpy.arange(nof_atoms, dtype=numpy.uintp)
            rng.shuffle(shuff_arr)
            newmol = (
                rdkit.Chem.rdmolops.RenumberAtoms(mol, shuff_arr.tolist())
                if len(shuff_arr) > 0
                else mol
            )

            coremaps = map_core_rdkit_v4(
                newmol,
                node_cis=node_cis,
                bond_ring=ext_ring or feat_ring != 0,
                bond_conj=ext_conj or feat_conj != 0,
            )
            try:
                converted = sorted(
                    (
                        __replace_result(
                            map_benson_core_v4(
                                coremap[0],
                                trim_ends=trim_ends,
                                ext_ring=ext_ring,
                                ext_conj=ext_conj,
                                feat_ring=feat_ring,
                                feat_conj=feat_conj,
                            )
                        )
                        for coremap in coremaps
                    ),
                    key=lambda x: x.graph,
                )
            except ParsingError:
                assert test_results is None
                continue

            for convres, test_compare in zip(
                converted, test_results, strict=True
            ):
                jsonres = __convert_to_json(convres)
                assert jsonres == test_compare
