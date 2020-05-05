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
transformer or a predictor with no corresponding converter implemented.
Please fill an issue at https://github.com/microsoft/hummingbird.
"""
_missing_backend = """
It usually means the backend is not currently supported.
Please check the spelling or fill an issue at https://github.com/microsoft/hummingbird.
"""
_constant_error = """
It usually means a constant is not available or you are trying to override a constant value.
"""


class MissingConverter(RuntimeError):
    """
    Raised when there is no registered converter for a machine learning operator.
    """

    def __init__(self, msg):
        super().__init__(msg + _missing_converter)


class MissingBackend(RuntimeError):
    """
    Raised when the selected backend is not supported.
    """

    def __init__(self, msg):
        super().__init__(msg + _missing_backend)


class ConstantError(TypeError):
    """
    Raised when a constant is not available or it get overwritten.
    """

    def __init__(self, msg):
        super().__init__(msg + _constant_error)
