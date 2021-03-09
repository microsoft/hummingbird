# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for Imputers
"""

import torch
import numpy as np

from ._physical_operator import PhysicalOperator
from . import constants


class SimpleImputer(PhysicalOperator, torch.nn.Module):
    """
    Class implementing SimpleImputer operators in PyTorch.
    """

    def __init__(self, logical_operator, device, statistics=None, missing=None, strategy=None):
        super(SimpleImputer, self).__init__(logical_operator)
        sklearn_imputer = logical_operator.raw_operator
        # Pull out the stats field from either the SKL imputer or args
        stats_ = statistics if statistics is not None else sklearn_imputer.statistics_
        # Process the stats into an array
        stats = [float(stat) for stat in stats_]

        missing_values = missing if missing is not None else sklearn_imputer.missing_values
        strategy = strategy if strategy is not None else sklearn_imputer.strategy

        b_mask = np.logical_not(np.isnan(stats))
        i_mask = [i for i in range(len(b_mask)) if b_mask[i]]
        self.transformer = True
        self.do_mask = strategy == "constant" or all(b_mask)
        self.mask = torch.nn.Parameter(torch.LongTensor([] if self.do_mask else i_mask), requires_grad=False)
        self.replace_values = torch.nn.Parameter(torch.tensor([stats_], dtype=torch.float32), requires_grad=False)

        self.is_nan = True if (missing_values == "NaN" or np.isnan(missing_values)) else False
        if not self.is_nan:
            self.missing_values = torch.nn.Parameter(torch.tensor([missing_values], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        if self.is_nan:
            result = torch.where(torch.isnan(x), self.replace_values.expand(x.shape), x)
            if self.do_mask:
                return result
            return torch.index_select(result, 1, self.mask)
        else:
            return torch.where(torch.eq(x, self.missing_values), self.replace_values.expand(x.shape), x)


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
