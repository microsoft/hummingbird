# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch
from onnxconverter_common.registration import register_converter


class Normalizer(torch.nn.Module):
    def __init__(self, norm, device):
        super(Normalizer, self).__init__()
        self.norm = norm
        self.regression = False

    def forward(self, x):
        if self.norm == "l1":
            return x / torch.abs(x).sum(1, keepdim=True)
        elif self.norm == "l2":
            return x / torch.pow(torch.pow(x, 2).sum(1, keepdim=True), 0.5)
        elif self.norm == "max":
            return x / torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        else:
            raise RuntimeError("Unsupported norm: {0}".format(self.norm))


def convert_sklearn_normalizer(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.Normalizer`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.Normalizer` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """

    return Normalizer(operator.raw_operator.norm, device)


register_converter("SklearnNormalizer", convert_sklearn_normalizer)
