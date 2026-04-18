"""Interface with RDKit."""

__all__ = [
    "ConvertResult_v4",
    "RDKitTranslationError",
    "benson_v4",
    "convert_benson",
    "load_smiles",
    "map_benson_core_v4",
    "map_benson_v4",
    "map_benson_v4_compat_v3",
    "map_core_rdkit_v4",
]

import contextlib as _contextlib

with _contextlib.suppress(ImportError):
    from . import _benson_v4 as benson_v4
    from ._benson_v3 import convert_mol_general as convert_benson
    from ._benson_v4 import (
        ConvertResult_v4,
        map_benson_core_v4,
        map_benson_v4,
        map_core_rdkit_v4,
    )
    from ._benson_v4_compat import map_benson_v4_compat_v3
    from ._load_smiles import load_smiles
    from ._structures import RDKitTranslationError
