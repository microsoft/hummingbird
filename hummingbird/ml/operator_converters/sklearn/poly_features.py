# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn PolynomialFeatures.
"""
from .._base_operator import BaseOperator
from onnxconverter_common.registration import register_converter
import torch


class PolynomialFeatures(BaseOperator, torch.nn.Module):
    """
    Class implementing PolynomialFeatures operators in PyTorch.
    """

    def __init__(self, n_features, degree, interaction_only, include_bias, device):
        super(PolynomialFeatures, self).__init__()
        self.transformer = True

        # TODO extend this class to support higher orders
        if degree != 2:
            raise RuntimeError("Only supports degree 2")

        self.n_features = n_features
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        indices = [i for j in range(n_features) for i in range(j * n_features + j, (j + 1) * n_features)]
        self.n_poly_features = len(indices)
        self.n_features = n_features
        self.indices = torch.nn.Parameter(torch.LongTensor(indices), requires_grad=False)

        self.bias = torch.nn.Parameter(torch.FloatTensor([1.0]), requires_grad=False)

    def forward(self, x):
        x_orig = x
        x = x.view(-1, self.n_features, 1) * x.view(-1, 1, self.n_features)
        x = x.view(-1, self.n_features ** 2)
        x = torch.index_select(x, 1, self.indices)
        if self.interaction_only:
            if self.include_bias:
                bias = self.bias.expand(x_orig.size()[0], 1)
                return torch.cat([bias, x], dim=1)
            else:
                return x
        else:
            if self.include_bias:
                bias = self.bias.expand(x_orig.size()[0], 1)
                return torch.cat([bias, x_orig, x], dim=1)
            else:
                return torch.cat([x_orig, x], dim=1)


def convert_sklearn_poly_features(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.PolynomialFeatures`

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.PolynomialFeatures` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    if operator.raw_operator.degree != 2:
        raise NotImplementedError("Hummingbird currently only supports degree 2 for PolynomialFeatures")
    return PolynomialFeatures(
        operator.raw_operator.n_input_features_,
        operator.raw_operator.degree,
        operator.raw_operator.interaction_only,
        operator.raw_operator.include_bias,
        device,
    )


register_converter("SklearnPolynomialFeatures", convert_sklearn_poly_features)
