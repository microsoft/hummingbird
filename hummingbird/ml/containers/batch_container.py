# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np

from hummingbird.ml._utils import pandas_installed

if pandas_installed():
    from pandas import DataFrame
else:
    DataFrame = None


class BatchContainer:
    def __init__(self, base_container, remainder_model_container=None):
        """
        A wrapper around one or two containers to do batch by batch prediction. The batch size is
        fixed when `base_container` is created. Together with `remainder_model_container`, this class
        enables prediction on a dataset of size `base_container._batch_size` * k +
        `remainder_model_container._batch_size`, where k is any integer. Its `predict` related method
        optionally takes `concatenate_outputs` argument, which when set to True causes the outputs to
        be returned as a list of individual prediction. This avoids an extra allocation of an output array
        and copying of each batch prediction into it.

        Args:
            base_container: One of subclasses of `SklearnContainer`.
            remainder_model_container: An auxiliary container that is used in the last iteration,
            if the test input batch size is not devisible by `base_container._batch_size`.
        """
        assert base_container._batch_size is not None
        self._base_container = base_container
        self._batch_size = base_container._batch_size

        if remainder_model_container:
            assert remainder_model_container._batch_size is not None
            self._remainder_model_container = remainder_model_container
            self._remainder_size = remainder_model_container._batch_size
        else:
            # This is remainder_size == 0 case
            # We repurpose base_container as a remainder_model_container
            self._remainder_model_container = base_container
            self._remainder_size = base_container._batch_size

    def __getattr__(self, name):
        return getattr(self._base_container, name)

    def decision_function(self, *inputs, concatenate_outputs=True):
        return self._predict_common(
            self._base_container.decision_function,
            self._remainder_model_container.decision_function,
            *inputs,
            concatenate_outputs=concatenate_outputs
        )

    def transform(self, *inputs, concatenate_outputs=True):
        return self._predict_common(
            self._base_container.transform,
            self._remainder_model_container.transform,
            *inputs,
            concatenate_outputs=concatenate_outputs
        )

    def score_samples(self, *inputs, concatenate_outputs=True):
        return self._predict_common(
            self._base_container.score_samples,
            self._remainder_model_container.score_samples,
            *inputs,
            concatenate_outputs=concatenate_outputs
        )

    def predict(self, *inputs, concatenate_outputs=True):
        return self._predict_common(
            self._base_container.predict,
            self._remainder_model_container.predict,
            *inputs,
            concatenate_outputs=concatenate_outputs
        )

    def predict_proba(self, *inputs, concatenate_outputs=True):
        return self._predict_common(
            self._base_container.predict_proba,
            self._remainder_model_container.predict_proba,
            *inputs,
            concatenate_outputs=concatenate_outputs
        )

    def _predict_common(self, predict_func, remainder_predict_func, *inputs, concatenate_outputs=True):
        if DataFrame is not None and type(inputs[0]) == DataFrame:
            # Split the dataframe into column ndarrays.
            inputs = inputs[0]
            input_names = list(inputs.columns)
            splits = [inputs[input_names[idx]] for idx in range(len(input_names))]
            inputs = tuple([df.to_numpy().reshape(-1, 1) for df in splits])

        def output_proc(predictions):
            if concatenate_outputs:
                return np.concatenate(predictions)
            return predictions

        is_tuple = isinstance(inputs, tuple)

        if is_tuple:
            total_size = inputs[0].shape[0]
        else:
            total_size = inputs.shape[0]

        if total_size == self._batch_size:
            # A single batch inference case
            return output_proc([predict_func(*inputs)])

        iterations = total_size // self._batch_size
        iterations += 1 if total_size % self._batch_size > 0 else 0
        iterations = max(1, iterations)
        predictions = []

        for i in range(0, iterations):
            start = i * self._batch_size
            end = min(start + self._batch_size, total_size)
            if is_tuple:
                batch = tuple([input[start:end, :] for input in inputs])
            else:
                batch = inputs[start:end, :]

            if i == iterations - 1:
                assert (end - start) == self._remainder_size
                out = remainder_predict_func(*batch)
            else:
                out = predict_func(*batch)

            predictions.append(out)

        return output_proc(predictions)
