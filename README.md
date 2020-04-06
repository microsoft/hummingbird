# Hummingbird

![](https://github.com/microsoft/hummingbird/workflows/Python%20application/badge.svg?branch=develop)

## Introduction
*hummingbird* converts trained traditional machine learning models to [PyTorch](https://pytorch.org/). Once in the PyTorch format, <!--you can further convert to [ONNX](https://github.com/onnx/onnx) or [TorchScript](https://pytorch.org/docs/stable/jit.html), and --> you can run the models on GPU for high performance native scoring. For full details, see [our paper](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf).

Currently we support [these](https://github.com/microsoft/hummingbird/blob/develop/hummingbird/_supported_operators.py#L26) tree-based classifiers and regressors.  These models include
[scikit-learn](https://scikit-learn.org/stable/) models such as  Decision Trees and Random Forest, and also [LightGBM](https://github.com/Microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost) Classifiers/Regressors.

## Installation

This was tested on Python 3.7.
```
python setup.py install
```

## Examples

See the [notebooks](notebooks) section for examples that demonstrate use and speedups.

In general, the syntax is very similar to [skl2onnx](https://github.com/onnx/sklearn-onnx), as hummingbird started as a fork of that project.

```python
import pickle, torch
import numpy as np
from onnxconverter_common.data_types import FloatTensorType
from hummingbird import convert_sklearn

# create some random data
X = np.array(np.random.rand(200000, 28), dtype=np.float32)
X_torch = torch.from_numpy(X)

# use hummingbird to convert your sklearn model to pytorch
model = pickle.load(open("my-skl-model.pkl", "rb"))
pytorch_model = convert_sklearn(model,[("input", FloatTensorType([200000, 28]))])

# Run hummingbird on CPU
pytorch_model.to('cpu')
hum_cpu = pytorch_model(X_torch)

# Run hummingbird on GPU
pytorch_model.to('cuda')
hum_gpu = pytorch_model(X_torch.to('cuda'))
```

# Contributing

Please see the section on [Contributing](CONTRIBUTING.md).

Also, see our [roadmap](wiki/Roadmap-for-Upcoming-Features-and-Support) of planned features. If you have a feature that you want us to add or that you are interested in working on, please open an issue or [contact us](hummingbird-dev@microsoft.com) to help us plan out the future roadmap.

# Authors

* Supun Nakandala
* Matteo Interlandi
* Karla Saur

# License
[MIT License](LICENSE)
