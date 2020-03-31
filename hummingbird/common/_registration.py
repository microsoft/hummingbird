# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

# This dictionary defines the converters which can be invoked in the
# conversion framework defined in _topology.py. A key in this dictionary
# is an operator's unique ID (e.g., string and type) while the
# associated value is the callable object used to convert the
# operator specified by the key.
_converter_pool = {}

# This dictionary defines the shape calculators which can be invoked in
# the conversion framework defined in _topology.py. A key in this
# dictionary is an operator's unique ID (e.g., string and type) while
# the associated value is the callable object used to infer the output
# shape(s) for the operator specified by the key.
_shape_calculator_pool = {}

# This dict defines the set of all registered optimizers
_optimizer_passes_pool = {}


def register_converter(operator_name, conversion_function, overwrite=False):
    """
    :param operator_name: A unique operator ID. It is usually a string
                          but you can use a type as well
    :param conversion_function: A callable object
    :param overwrite: By default, we raise an exception if the caller of
                      this function is trying to assign an existing
                      key (i.e., operator_name) a new value
                      (i.e., conversion_function). Set this flag to True
                      to enable overwriting.
    """
    if not overwrite and operator_name in _converter_pool:
        raise ValueError("We do not overwrite registered converter " "by default")

    _converter_pool[operator_name] = conversion_function


def get_converter(operator_name):
    if operator_name not in _converter_pool:
        msg = "Unsupported conversion for operator %s (%d registered)" % (operator_name, len(_converter_pool))
        raise ValueError(msg)
    return _converter_pool[operator_name]
