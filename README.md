[![](https://i.imgur.com/0pp9lMS.png?1)](https://github.com/microsoft/hummingbird/)

# Hummingbird

![](https://github.com/microsoft/hummingbird/workflows/Python%20application/badge.svg?branch=develop)
![coverage](https://codecov.io/gh/microsoft/hummingbird/branch/master/graph/badge.svg)
[![Gitter](https://badges.gitter.im/hummingbird-ml/community.svg)](https://gitter.im/hummingbird-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## Introduction
*Hummingbird* converts trained traditional Machine Learning models into [PyTorch](https://pytorch.org/). Once in the PyTorch format, <!--you can further convert to [ONNX](https://github.com/onnx/onnx) or [TorchScript](https://pytorch.org/docs/stable/jit.html), and --> you can run the models on GPU for high performance native scoring. For full details, see [our papers](#documentation).

Currently we support [these](https://github.com/microsoft/hummingbird/blob/develop/hummingbird/_supported_operators.py#L26) tree-based classifiers and regressors.  These models include
[scikit-learn](https://scikit-learn.org/stable/) models such as  Decision Trees and Random Forest, and also [LightGBM](https://github.com/Microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost) Classifiers/Regressors.

## Installation

This was tested on Python 3.7 on a Linux machine.  It is recommended to use a virtual environment (See: [python3 venv doc](https://docs.python.org/3/tutorial/venv.html) or [Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).)
```
mkdir hummingbird
cd hummingbird
git clone https://github.com/microsoft/hummingbird.git .
python setup.py install
```



See also [Troubleshooting](TROUBLESHOOTING.md) for common problems.

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
model = lgb.LGBMClassifier()
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

# Documentation

The API documentation is [here](https://microsoft.github.io/hummingbird/).

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
