# Hummingbird

[![PyPI version](https://badge.fury.io/py/hummingbird-ml.svg)](https://badge.fury.io/py/hummingbird-ml)
[![](https://github.com/microsoft/hummingbird/workflows/Build/badge.svg?branch=master)](https://github.com/microsoft/hummingbird/actions)
![](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)
[![coverage](https://codecov.io/gh/microsoft/hummingbird/branch/master/graph/badge.svg)](https://codecov.io/github/microsoft/hummingbird?branch=master)
[![Gitter](https://badges.gitter.im/hummingbird-ml/community.svg)](https://gitter.im/hummingbird-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Introduction
*Hummingbird* is a library for compiling trained traditional ML models into tensor computations. *Hummingbird* allows users to seamlessly leverage neural network frameworks (such as [PyTorch](https://pytorch.org/)) to accelerate traditional ML models. Thanks to *Hummingbird*, users can benefit from: (1) all the current and future optimizations implemented in neural network frameworks; (2) native hardware acceleration; (3) having a unique platform to support for both traditional and neural network models; and have all of this (4) without having to re-engineer their models.

Currently, you can use *Hummingbird* to convert your trained traditional ML models into [PyTorch](https://pytorch.org/). *Hummingbird* [supports](https://github.com/microsoft/hummingbird/wiki/Supported-Operators) a variety of tree-based classifiers and regressors.  These models include
[scikit-learn](https://scikit-learn.org/stable/) Decision Trees and Random Forest, and also [LightGBM](https://github.com/Microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost) Classifiers/Regressors. Support for other neural network backends (e.g., [ONNX](https://onnx.ai/), [TVM](https://docs.tvm.ai/)) and models is on our [roadmap](https://github.com/microsoft/hummingbird/wiki/Roadmap-for-Upcoming-Features-and-Support).

## Installation

Hummingbird was tested on Python >= 3.5 on Linux, Windows and MacOS machines.  It is recommended to use a virtual environment (See: [python3 venv doc](https://docs.python.org/3/tutorial/venv.html) or [Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).)


Install the Hummingbird package in Linux or Mac
```
pip install hummingbird-ml
```

To install the package in Windows instead:
```
pip install hummingbird-ml -f https://download.pytorch.org/whl/torch_stable.html
```

If you require the optional dependencies lightgbm and xgboost, you can use:
```
pip install hummingbird-ml[extra]
```


See also [Troubleshooting](TROUBLESHOOTING.md) for common problems.

## Examples

See the [notebooks](notebooks) section for examples that demonstrate use and speedups.

In general, Hummingbird syntax is very intuitive and minimal. To run your traditional ML model on DNN frameworks, you only need to `import hummingbird.ml` and add `to('dnn_framework')` to your code. Below is an example using a [scikit-learn random forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) model and [PyTorch](https://pytorch.org/) as target framework.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert

# Create some random data for binary classification
num_classes = 2
X = np.array(np.random.rand(100000, 28), dtype=np.float32)
y = np.random.randint(num_classes, size=100000)

# Create and train a model (scikit-learn RandomForestClassifier in this case)
skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
skl_model.fit(X, y)

# Use Hummingbird to convert the model to PyTorch
model = convert(skl_model, 'pytorch')

# Run predictions on CPU
model.predict(X)

# Run predictions on GPU
model.to('cuda')
model.predict(X)
```

# Documentation

The API documentation is [here](https://microsoft.github.io/hummingbird/).

You can also read about Hummingbird in our blog post [here](https://azuredata.microsoft.com/articles/ebd95ec0-1eae-44a3-90f5-c11f5c916d15).

For more details on the vision and on the technical details related to Hummingbird, please check our papers:

* [Taming Model Serving Complexity, Performance and Cost: A Compilation to Tensor Computations Approach](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf). Supun Nakandalam, Karla Saur, Gyeong-In Yu, Konstantinos Karanasos, Carlo Curino, Markus Weimer, Matteo Interlandi. Technical Report
* [Compiling Classical ML Pipelines into Tensor Computations for One-size-fits-all Prediction Serving](http://learningsys.org/neurips19/assets/papers/27_CameraReadySubmission_Hummingbird%20(5).pdf). Supun Nakandala, Gyeong-In Yu, Markus Weimer, Matteo Interlandi. System for ML Workshop. NeurIPS 2019

# Contributing

We welcome contributions! Please see the guide on [Contributing](CONTRIBUTING.md).

Also, see our [roadmap](https://github.com/microsoft/hummingbird/wiki/Roadmap-for-Upcoming-Features-and-Support) of planned features.

# Community

Join our community! [![Gitter](https://badges.gitter.im/hummingbird-ml/community.svg)](https://gitter.im/hummingbird-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

 For more formal enquiries, you can [contact us](mailto:hummingbird-dev@microsoft.com).

# Authors

* Supun Nakandala
* Matteo Interlandi
* Karla Saur

# License
[MIT License](LICENSE)
