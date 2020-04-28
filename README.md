[![](https://i.imgur.com/0pp9lMS.png?1)](https://github.com/microsoft/hummingbird/)

# Hummingbird

![](https://github.com/microsoft/hummingbird/workflows/Python%20application/badge.svg?branch=develop)

## Introduction
*Hummingbird* converts trained traditional Machine Learning models into [PyTorch](https://pytorch.org/). Once in the PyTorch format, <!--you can further convert to [ONNX](https://github.com/onnx/onnx) or [TorchScript](https://pytorch.org/docs/stable/jit.html), and --> you can run the models on GPU for high performance native scoring. For full details, see [our paper](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf).

Currently we support [these](https://github.com/microsoft/hummingbird/blob/develop/hummingbird/_supported_operators.py#L26) tree-based classifiers and regressors.  These models include
[scikit-learn](https://scikit-learn.org/stable/) models such as  Decision Trees and Random Forest, and also [LightGBM](https://github.com/Microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost) Classifiers/Regressors.

## Installation

This was tested on Python 3.7 on a Unix machine.
```
mkdir hummingbird
cd hummingbird
git clone https://github.com/microsoft/hummingbird.git .
python setup.py install
```

## Examples

See the [notebooks](notebooks) section for examples that demonstrate use and speedups.

In general, the syntax is very similar to [skl2onnx](https://github.com/onnx/sklearn-onnx), as hummingbird started as a fork of that project.

```python
import torch
import numpy as np
import lightgbm as lgb
from hummingbird import convert_lightgbm

# Create some random data for binary classification
num_classes = 2
X = np.array(np.random.rand(100000, 28), dtype=np.float32)
y = np.random.randint(num_classes, size=100000)

# Create and train a model (LightGBM in this case)
model = model = lgb.LGBMClassifier()
model.fit(X, y)

# Use Hummingbird to convert the model to pytorch
pytorch_model = convert_lightgbm(model)

# Run Hummingbird on CPU
pytorch_model.to('cpu')
hb_cpu = pytorch_model(torch.from_numpy(X))

# Run Hummingbird on GPU
pytorch_model.to('cuda')
hb_gpu = pytorch_model(torch.from_numpy(X).to('cuda'))
```

# Contributing

We welcome contributions! Please see the guide on [Contributing](CONTRIBUTING.md).

Also, see our [roadmap](wiki/Roadmap-for-Upcoming-Features-and-Support) of planned features.

# Community

Join our community! *gitter badge here*

 For more formal enquiries, you can [contact us](mailto:hummingbird-dev@microsoft.com).

# Authors

* Supun Nakandala
* Matteo Interlandi
* Karla Saur

# License
[MIT License](LICENSE)
