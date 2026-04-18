"""Prediction models."""

import abc
import dataclasses
from typing import Any

import numpy
import polars

from sgrpy.iotypes import JSON, JSONable, json_to_dict
from sgrpy.linear import dict_to_csrvecint, json_to_np, np_to_json

from ._model_types import PTYPE_PRED_COL


class ModelPredict(JSONable):
    """Component which makes a prediction based on a feature vector."""

    @abc.abstractmethod
    def predict(self, vec: numpy.typing.NDArray[Any]) -> float:
        """Make a prediction."""

    @abc.abstractmethod
    def predict_struct(self, struct: dict[str, list[int]]) -> float:
        """Make a prediction based on CSR struct."""

    def predict_df(
        self, lf: polars.LazyFrame, pred_col: str
    ) -> polars.LazyFrame:
        """Make a prediction from a column of feature vectors."""
        lf_pred = lf.with_columns(
            polars.col("csrvec")
            .map_elements(self.predict_struct, return_dtype=PTYPE_PRED_COL)
            .alias(f"pred_{pred_col}")
        )
        return lf_pred


@dataclasses.dataclass(frozen=True, slots=True)
class ModelPredictLinear(ModelPredict):
    coefs: numpy.typing.NDArray[Any]

    def predict(self, vec: numpy.typing.NDArray[Any]) -> float:
        return float(self.coefs @ vec)

    def predict_struct(self, struct: dict[str, list[int]]) -> float:
        size = self.coefs.shape[0]
        vec_sp = dict_to_csrvecint(struct, size)
        vec_np = vec_sp.todense()[0, :]
        return self.predict(vec_np)

    @classmethod
    def _from_json(cls, json: JSON) -> "ModelPredictLinear":
        coefs = json_to_np(json)
        return ModelPredictLinear(coefs=coefs)

    def to_json(self) -> JSON:
        return np_to_json(self.coefs)


def model_predict_from_json(json: JSON) -> ModelPredict:
    json_dict = json_to_dict(json)
    model_type = str(json_dict["type"])
    model_data = json_dict["data"]
    match model_type:
        case "ModelPredictLinear":
            return ModelPredictLinear.from_json(model_data)
        case _:
            raise ValueError(f"Unrecognized prediction model {model_type}")


def model_predict_to_json(model_str: ModelPredict) -> JSON:
    return {"type": type(model_str).__name__, "data": model_str.to_json()}
