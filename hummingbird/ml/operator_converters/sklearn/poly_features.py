# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
Converter for scikit-learn PolynomialFeatures.
"""
from .._physical_operator import PhysicalOperator
from onnxconverter_common.registration import register_converter
import torch
import itertools


class PolynomialFeatures(PhysicalOperator, torch.nn.Module):
    """
    Class implementing PolynomialFeatures operators in PyTorch.

    # TODO extend this class to support higher orders
    """

    def __init__(self, operator, n_features, degree=2, interaction_only=False, include_bias=True, device=None):
        super(PolynomialFeatures, self).__init__(operator)
        self.transformer = True
        self.degree = degree
        self.n_features = n_features
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def forward(self, x):
        if self.degree < 0:
            raise ValueError("Degree should be greater than or equal to 0.")

        features = []

        # Move input to GPU if available
        device = x.device

        # Add bias term if include_bias is True
        if self.include_bias:
            bias = torch.ones(x.size()[0], 1, device=device)
            features.append(bias)

        # Generate polynomial features
        for d in range(1, self.degree + 1):
            for combo in itertools.combinations_with_replacement(range(self.n_features), d):
                if self.interaction_only and len(set(combo)) != d:
                    continue
                new_feature = torch.prod(torch.stack([x[:, idx] for idx in combo], dim=1), dim=1, keepdim=True)
                features.append(new_feature)

        return torch.cat(features, dim=1).to(device=device)


def convert_sklearn_poly_features(operator, device, extra_config):
    """
    Converter for `sklearn.preprocessing.PolynomialFeatures`

    Currently this supports only degree 2, and does not support interaction_only

    Args:
        operator: An operator wrapping a `sklearn.preprocessing.PolynomialFeatures` model
        device: String defining the type of device the converted operator should be run on
        extra_config: Extra configuration used to select the best conversion strategy

    Returns:
        A PyTorch model
    """
    assert operator is not None, "Cannot convert None operator"

    if operator.raw_operator.interaction_only:
        raise NotImplementedError("Hummingbird does not currently support interaction_only flag for PolynomialFeatures")

    if operator.raw_operator.degree < 0:
        raise NotImplementedError("Hummingbird not supports negtive degree for PolynomialFeatures")
    return PolynomialFeatures(
        operator,
        operator.raw_operator.n_features_in_,
        operator.raw_operator.degree,
        operator.raw_operator.interaction_only,
        operator.raw_operator.include_bias,
        device,
    )


register_converter("SklearnPolynomialFeatures", convert_sklearn_poly_features)
