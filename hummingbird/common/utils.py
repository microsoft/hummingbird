# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from collections import OrderedDict
from .data_types import TensorType


def get_column_index(i, inputs):
    """
    Returns a tuples (variable index, column index in that variable).
    The function has two different behaviours, one when *i* (column index)
    is an integer, another one when *i* is a string (column name).
    If *i* is a string, the function looks for input name with
    this name and returns (index, 0).
    If *i* is an integer, let's assume first we have two inputs
    *I0 = FloatTensorType([None, 2])* and *I1 = FloatTensorType([None, 3])*,
    in this case, here are the results:

    ::

        get_column_index(0, inputs) -> (0, 0)
        get_column_index(1, inputs) -> (0, 1)
        get_column_index(2, inputs) -> (1, 0)
        get_column_index(3, inputs) -> (1, 1)
        get_column_index(4, inputs) -> (1, 2)
    """
    if isinstance(i, int):
        if i == 0:
            # Useful shortcut, skips the case when end is None
            # (unknown dimension)
            return 0, 0
        vi = 0
        pos = 0
        end = (inputs[0].type.shape[1]
               if isinstance(inputs[0].type, TensorType) else 1)
        if end is None:
            raise RuntimeError("Cannot extract a specific column {0} when "
                               "one input ('{1}') has unknown "
                               "dimension.".format(i, inputs[0]))
        while True:
            if pos <= i < end:
                return (vi, i - pos)
            vi += 1
            pos = end
            rel_end = (inputs[vi].type.shape[1]
                       if isinstance(inputs[vi].type, TensorType) else 1)
            if rel_end is None:
                raise RuntimeError("Cannot extract a specific column {0} when "
                                   "one input ('{1}') has unknown "
                                   "dimension.".format(i, inputs[vi]))
            end += rel_end
    else:
        for ind, inp in enumerate(inputs):
            if inp.pytorch_name == i:
                return ind, 0
        raise RuntimeError("Unable to find column name '{0}'".format(i))


def get_column_indices(indices, inputs, multiple):
    """
    Returns the requested graph inpudes based on their
    indices or names. See :func:`get_column_index`.

    :param indices: variables indices or names
    :param inputs: graph inputs
    :param multiple: allows column to come from multiple variables
    :return: a tuple *(variable name, list of requested indices)* if
        *multiple* is False, a dictionary *{ var_index: [ list of
        requested indices ] }*
        if *multiple* is True
    """
    if multiple:
        res = OrderedDict()
        for p in indices:
            ov, pytorch_i = get_column_index(p, inputs)
            if ov not in res:
                res[ov] = []
            res[ov].append(pytorch_i)
        return res
    else:
        pytorch_var = None
        pytorch_is = []
        for p in indices:
            ov, pytorch_i = get_column_index(p, inputs)
            pytorch_is.append(pytorch_i)
            if pytorch_var is None:
                pytorch_var = ov
            elif pytorch_var != ov:
                cols = [pytorch_var, ov]
                raise NotImplementedError(
                    "sklearn-pytorch is not able to merge multiple columns from "
                    "multiple variables ({0}). You should think about merging "
                    "initial types.".format(cols))
        return pytorch_var, pytorch_is
