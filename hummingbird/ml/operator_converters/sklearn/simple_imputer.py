# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn SimpleImputer.
"""
from .._base_operator import BaseOperator
import numpy as np
from onnxconverter_common.registration import register_converter
import torch


class SimpleImputer(BaseOperator, torch.nn.Module):
    """
    Class implementing SimpleImputer operators in PyTorch.
    """

    def __init__(self, sklearn_imputer, device):
        super(SimpleImputer, self).__init__()
        b_mask = np.logical_not(np.isnan(sklearn_imputer.statistics_))
        i_mask = [i for i in range(len(b_mask)) if b_mask[i]]
        self.transformer = True
        self.do_mask = sklearn_imputer.strategy == "constant" or all(b_mask)
        self.mask = torch.nn.Parameter(torch.LongTensor([] if self.do_mask else i_mask), requires_grad=False)
        self.replace_values = torch.nn.Parameter(
            torch.tensor([sklearn_imputer.statistics_], dtype=torch.float32), requires_grad=False
        )
        self.missing_values = torch.nn.Parameter(
            torch.tensor([sklearn_imputer.missing_values], dtype=torch.float32), requires_grad=False
        )
        self.is_nan = True if (sklearn_imputer.missing_values == "NaN" or np.isnan(sklearn_imputer.missing_values)) else False

    def forward(self, x):
        if self.is_nan:
            result = torch.where(torch.isnan(x), self.replace_values, x)
            if self.do_mask:
                return result
            return torch.index_select(result, 1, self.mask)
        else:
            return torch.where(torch.eq(x, self.missing_values), self.replace_values, x)


def convert_sklearn_simple_imputer(operator, device, extra_config):
    """
    Converter for `sklearn.impute.SimpleImputer`

    Args:
        operator: An operator wrapping a `sklearn.impute.SimpleImputer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    return SimpleImputer(operator.raw_operator, device)


register_converter("SklearnSimpleImputer", convert_sklearn_simple_imputer)
