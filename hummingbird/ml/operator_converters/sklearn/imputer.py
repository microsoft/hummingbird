# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn Imputers: SimpleImputer and MissingIndicator
"""
from .._physical_operator import PhysicalOperator
import numpy as np
from onnxconverter_common.registration import register_converter
import torch


class SimpleImputer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing SimpleImputer operators in PyTorch.
    """

    def __init__(self, logical_operator, device):
        super(SimpleImputer, self).__init__(logical_operator)
        sklearn_imputer = logical_operator.raw_operator
        stats = [float(stat) for stat in sklearn_imputer.statistics_ if isinstance(stat, float)]
        b_mask = np.logical_not(np.isnan(stats))
        i_mask = [i for i in range(len(b_mask)) if b_mask[i]]
        self.transformer = True
        self.do_mask = sklearn_imputer.strategy == "constant" or all(b_mask)
        self.mask = torch.nn.Parameter(torch.LongTensor([] if self.do_mask else i_mask), requires_grad=False)
        self.replace_values = torch.nn.Parameter(
            torch.tensor([sklearn_imputer.statistics_], dtype=torch.float32), requires_grad=False
        )

        self.is_nan = True if (sklearn_imputer.missing_values == "NaN" or np.isnan(sklearn_imputer.missing_values)) else False
        if not self.is_nan:
            self.missing_values = torch.nn.Parameter(
                torch.tensor([sklearn_imputer.missing_values], dtype=torch.float32), requires_grad=False
            )

    def forward(self, x):
        if self.is_nan:
            result = torch.where(torch.isnan(x), self.replace_values.expand(x.shape), x)
            if self.do_mask:
                return result
            return torch.index_select(result, 1, self.mask)
        else:
            return torch.where(torch.eq(x, self.missing_values), self.replace_values.expand(x.shape), x)


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
    assert operator is not None, "Cannot convert None operator"

    return SimpleImputer(operator, device)


class MissingIndicator(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Imputer operators in MissingIndicator.
    """

    def __init__(self, logical_operator, device):
        super(MissingIndicator, self).__init__(logical_operator)
        sklearn_missing_indicator = logical_operator.raw_operator
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
    assert operator is not None, "Cannot convert None operator"

    return MissingIndicator(operator, device)


register_converter("SklearnImputer", convert_sklearn_simple_imputer)
register_converter("SklearnSimpleImputer", convert_sklearn_simple_imputer)
register_converter("SklearnMissingIndicator", convert_sklearn_missing_indicator)
