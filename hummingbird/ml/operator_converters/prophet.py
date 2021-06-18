# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for Prophet.
"""

import numpy as np
import torch
from datetime import datetime
from onnxconverter_common.registration import register_converter

from ._physical_operator import PhysicalOperator
from . import constants


class Prophet(PhysicalOperator, torch.nn.Module):
    """
    Class implementing Prophet operator in PyTorch.
    """

    def __init__(self, logical_operator, k, m, deltas, floor, start, t_scale, y_scale, changepoints_t, device):
        super(Prophet, self).__init__(logical_operator)
        self.regression = True
        self.k = k
        self.m = m
        self.deltas = torch.nn.Parameter(torch.Tensor(deltas), requires_grad=False)
        self.floor = floor
        self.start = start
        self.t_scale = t_scale
        self.y_scale = y_scale
        self.changepoints_t = torch.nn.Parameter(torch.Tensor(changepoints_t), requires_grad=False)

    def forward(self, x):
        x = torch.sort(x)[0]
        t = (x - self.start) / self.t_scale

        # Linear.
        # Intercept changes
        gammas = -self.changepoints_t * self.deltas
        # Get cumulative slope and intercept at each t
        k_t = self.k * torch.ones_like(t)
        m_t = self.m * torch.ones_like(t)
        for s, t_s in enumerate(self.changepoints_t):
            indx = t >= t_s
            k_t[indx] += self.deltas[s]
            m_t[indx] += gammas[s]
            trend = k_t * t + m_t
        trend = trend * self.y_scale + self.floor
        return trend


def convert_prophet(operator, device=None, extra_config={}):
    """
    Converter for `prophet.Prophet`

    Args:
        operator: An operator wrapping a `prophet.Prophet` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    k = np.nanmean(operator.original_operator.params["k"])
    m = np.nanmean(operator.original_operator.params["m"])
    deltas = np.nanmean(operator.original_operator.params["delta"], axis=0)
    floor = 0
    start = (operator.original_operator.start - datetime(1970, 1, 1)).total_seconds()
    t_scale = operator.original_operator.t_scale.total_seconds()
    y_scale = operator.original_operator.y_scale
    changepoints_t = operator.original_operator.changepoints_t
    growth = operator.original_operator.growth

    assert growth == "linear", "Growth function {} not supported yet.".format(growth)

    return Prophet(operator, k, m, deltas, floor, start, t_scale, y_scale, changepoints_t, device)


register_converter("SklearnProphet", convert_prophet)
