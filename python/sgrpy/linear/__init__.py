"""Linear algebra submodule."""

__all__ = [
    "csr_equal",
    "csr_to_json",
    "dict_to_csrvecint",
    "json_to_csr",
    "json_to_np",
    "linreg",
    "np_to_json",
    "scale_csr",
    "scale_gram_csr",
    "scale_gram_sparse",
    "within_kern",
    "within_kern_fast",
]

from ._regression import (
    linreg,
    scale_csr,
    scale_gram_csr,
    scale_gram_sparse,
    within_kern,
    within_kern_fast,
)
from ._utils import (
    csr_equal,
    csr_to_json,
    dict_to_csrvecint,
    json_to_csr,
    json_to_np,
    np_to_json,
)
