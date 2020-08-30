# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn MissingIndicator.
"""
from .._base_operator import BaseOperator
import numpy as np
from onnxconverter_common.registration import register_converter
import torch


class MissingIndicator(BaseOperator, torch.nn.Module):
    """
    Class implementing Imputer operators in MissingIndicator.
    """

    def __init__(self, sklearn_missing_indicator, device):
        super(MissingIndicator, self).__init__()
        self.transformer = True
        self.missing_values = torch.nn.Parameter(
            torch.tensor([sklearn_missing_indicator.missing_values], dtype=torch.float32), requires_grad=False
        )
        self.features = sklearn_missing_indicator.features
        self.is_nan = True if (sklearn_missing_indicator.missing_values in ["NaN", None, np.nan]) else False
        self.column_indices = torch.nn.Parameter(torch.LongTensor(sklearn_missing_indicator.features_), requires_grad=False)

    def forward(self, x):
        if self.is_nan:
            if self.features == "all":
                return torch.isnan(x).float()
            else:
                return torch.isnan(torch.index_select(x, 1, self.column_indices)).float()
        else:
            if self.features == "all":
                return torch.eq(x, self.missing_values).float()
            else:
                return torch.eq(torch.index_select(x, 1, self.column_indices), self.missing_values).float()


def convert_sklearn_missing_indicator(operator, device, extra_config):
    """
    Converter for `sklearn.impute.MissingIndicator`
    Args:
        operator: An operator wrapping a `sklearn.impute.MissingIndicator` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy
    Returns:
        A PyTorch model
    """
    return MissingIndicator(operator.raw_operator, device)


register_converter("SklearnMissingIndicator", convert_sklearn_missing_indicator)
