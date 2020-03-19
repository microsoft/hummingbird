# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Common errors.
"""
_missing_converter = """
It usually means the pipeline being converted contains a
transformer or a predictor with no corresponding converter
implemented in sklearn-pytorch.
"""


class MissingConverter(RuntimeError):
    """
    Raised when there is no registered converter for a machine learning operator.
    """

    def __init__(self, msg):
        super().__init__(msg + _missing_converter)
