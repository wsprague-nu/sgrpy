"""Function for loading SMILES."""

from typing import Any

import rdkit
import rdkit.Chem.rdchem
import rdkit.Chem.rdmolfiles
import rdkit.Chem.rdmolops


def load_smiles(smiles: str) -> rdkit.Chem.rdchem.Mol | None:
    with rdkit.rdBase.BlockLogs():
        parser_params: Any = rdkit.Chem.rdmolfiles.SmilesParserParams()
        parser_params.sanitize = False
        parser_params.removeHs = False
        parser_params.allowCXSMILES = True
        with rdkit.rdBase.BlockLogs():
            rdkitmol: rdkit.Chem.rdchem.Mol | None = (
                rdkit.Chem.rdmolfiles.MolFromSmiles(
                    smiles, params=parser_params
                )
            )
        if rdkitmol is None:
            return None

        rdkit.Chem.rdmolops.AssignStereochemistry(rdkitmol, force=True)
        rdkit.Chem.rdmolops.FindPotentialStereoBonds(rdkitmol)

        remove_h_params: Any = rdkit.Chem.rdmolops.RemoveHsParameters()
        remove_h_params.removeHydrides = False
        remove_h_params.removeInSGroups = False
        remove_h_params.removeDefiningBondStereo = False
        remove_h_params.showWarnings = False
        atom: rdkit.Chem.rdchem.Atom
        for atom in rdkitmol.GetAtoms():  # type: ignore[no-untyped-call,unused-ignore]
            if atom.GetAtomicNum() == 1 and atom.GetFormalCharge() != 0:
                return None
        rdkitmol = rdkit.Chem.rdmolops.RemoveHs(
            rdkitmol, params=remove_h_params, sanitize=False
        )

        flags = (
            rdkit.Chem.rdmolops.SANITIZE_NONE
            ^ rdkit.Chem.rdmolops.SANITIZE_ADJUSTHS
            ^ rdkit.Chem.rdmolops.SANITIZE_CLEANUP
            ^ rdkit.Chem.rdmolops.SANITIZE_CLEANUPCHIRALITY
            ^ rdkit.Chem.rdmolops.SANITIZE_CLEANUP_ORGANOMETALLICS
            ^ rdkit.Chem.rdmolops.SANITIZE_FINDRADICALS
            ^ rdkit.Chem.rdmolops.SANITIZE_KEKULIZE
            ^ rdkit.Chem.rdmolops.SANITIZE_PROPERTIES
            ^ rdkit.Chem.rdmolops.SANITIZE_SETAROMATICITY
            ^ rdkit.Chem.rdmolops.SANITIZE_SETCONJUGATION
            ^ rdkit.Chem.rdmolops.SANITIZE_SETHYBRIDIZATION
            ^ rdkit.Chem.rdmolops.SANITIZE_SYMMRINGS
        )
        r_val: rdkit.Chem.rdmolops.SanitizeFlags = (
            rdkit.Chem.rdmolops.SanitizeMol(
                rdkitmol, sanitizeOps=flags, catchErrors=True
            )
        )
        if r_val != rdkit.Chem.rdmolops.SANITIZE_NONE:
            return None

    return rdkitmol
