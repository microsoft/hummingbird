# -------------------------------------------------------------------------
# Copyright (c) 2020 Supun Nakandala. All Rights Reserved.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Base classes for KNeighbors model implementations: (KNeighborsClassifier).
"""

import torch
from ._base_operator import BaseOperator


class KNeighborsClassifierModel(BaseOperator, torch.nn.Module):
    def __init__(self):
        super(KNeighborsClassifierModel, self).__init__()
        self.classification = True

    def forward(self, x):
        return x