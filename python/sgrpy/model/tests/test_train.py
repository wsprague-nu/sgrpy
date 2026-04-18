"""Test training methods."""

import math

from .._model_train import train_model_smiles_mol
from .._models import SGRModel_v1


def test_train_mol() -> None:
    """Test model training on example SMILES strings."""
    smiles = ("CCC", "CC", "CCO", "CCC=CC", "xxy", "CCCCO", "CC=CCCC=CC")
    weights = (1.0, 0.8, 1.0, 2.0, 1.0, 1.0, 1.0)
    groups = (1, 1, 8, 3, 4, 4, 9)

    model = train_model_smiles_mol(
        smiles,
        weights,
        groups,
        min_count=2,
        covar_filter=True,
        max_ssize=None,
    )

    assert isinstance(model, SGRModel_v1)

    mgi = model.calc_info("CCCCO")

    assert (
        mgi is not None and math.isclose(mgi, -6.550198997270105)  # noqa: PLR2004
    )

    result = model.calc_info("CCCCl")
    if result is not None:
        raise ValueError('Value "CCCCl" should have been rejected')

    result = model.calc_info("xxy")
    if result is not None:
        raise ValueError('Value "xxy" should have been rejected')


# def test_train_ds() -> None:
#     ds_path = pathlib.Path(__file__).parent / "example_aug.tsv"
#     df = polars.read_csv(ds_path, separator="\t")

#     smiles = df["SMILES"]
#     weights = df["weight"]
#     groups = df["idx"]

#     model = train_model_smiles_mol(
#         smiles.to_list(),
#         weights.to_list(),
#         groups.to_list(),
#         train_targets={"hf298": df["Value"].to_list()},
#         min_count=2,
#         covar_filter=True,
#         max_ssize=2,
#         dataframe_out="./out_data.tsv",
#     )

#     assert isinstance(model, SGRModel_v1)

#     assert model.graph_information("CCCCO") == -6.626617805279214

#     try:
#         model.graph_information("CCCCl")
#         raise ValueError('Value "CCCCl" should have been rejected')
#     except ModelReject:
#         pass

#     try:
#         model.graph_information("xxy")
#         raise ValueError('Value "xxy" should have been rejected')
#     except ModelReject:
#         pass
