# Hummingbird

[![PyPI version](https://badge.fury.io/py/hummingbird-ml.svg)](https://badge.fury.io/py/hummingbird-ml)
[![](https://github.com/microsoft/hummingbird/workflows/Build/badge.svg?branch=main)](https://github.com/microsoft/hummingbird/actions)
![](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7%20%7C%203.8-blue)
[![coverage](https://codecov.io/gh/microsoft/hummingbird/branch/main/graph/badge.svg)](https://codecov.io/github/microsoft/hummingbird?branch=main)
[![Gitter](https://badges.gitter.im/hummingbird-ml/community.svg)](https://gitter.im/hummingbird-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Downloads](https://pepy.tech/badge/hummingbird-ml)](https://pepy.tech/project/hummingbird-ml)

<p>
    <img src="https://github.com/microsoft/hummingbird/raw/main/website/images/hb-logo-notext.png"  width=200  >
    <br>

</p>

## Introduction
*Hummingbird* is a library for compiling trained traditional ML models into tensor computations. *Hummingbird* allows users to seamlessly leverage neural network frameworks (such as [PyTorch](https://pytorch.org/)) to accelerate traditional ML models. Thanks to *Hummingbird*, users can benefit from: (1) all the current and future optimizations implemented in neural network frameworks; (2) native hardware acceleration; (3) having a unique platform to support for both traditional and neural network models; and have all of this (4) without having to re-engineer their models.

Currently, you can use *Hummingbird* to convert your trained traditional ML models into [PyTorch](https://pytorch.org/), [TorchScript](https://pytorch.org/docs/stable/jit.html), [ONNX](https://onnx.ai/), and [TVM](https://docs.tvm.ai/)). *Hummingbird* [supports](https://github.com/microsoft/hummingbird/wiki/Supported-Operators) a variety of ML models and featurizers.  These models include
[scikit-learn](https://scikit-learn.org/stable/) Decision Trees and Random Forest, and also [LightGBM](https://github.com/Microsoft/LightGBM) and [XGBoost](https://github.com/dmlc/xgboost) Classifiers/Regressors. Support for other neural network backends and models is on our [roadmap](https://github.com/microsoft/hummingbird/wiki/Roadmap-for-Upcoming-Features-and-Support).

Hummingbird also provides a convenient uniform "inference" API following the Sklearn API. This allows swapping Sklearn models with Hummingbird-generated ones without having to change the inference code.

## How Hummingbird Works

Hummingbird works by reconfiguring algorithmic operators such that we can perform more regular computations which are amenable to vectorized and GPU execution. Each operator is slightly different, and we incorporate multiple strategies. This example explains one of Hummingbird's strategies for translating a decision tree into tensors involving GEMM  (GEneric Matrix Multiplication), where we implement the traversal of the tree using matrix multiplications.  (GEMM is one of the three tree conversion strategies we currently support.)


<p align="center">
    <img src="https://github.com/microsoft/hummingbird/raw/main/website/images/1-simple-reg-tree.png" width=600 >
    <br>
    <em>Simple decision tree</em>
</p>


In this example, the decision tree has four decision nodes (orange), and five leaf nodes (blue). The tree takes a feature vector with five elements as input. For example, assume that we want to calculate the output of this observation:


<p align="center">
    <img src="https://github.com/microsoft/hummingbird/raw/main/website/images/2-calc-output.png" width=400 >
</p>

**Step 1:** Multiply the `input tensor` with tensor `A` (computed from the decision tree model above) that captures the relationship between input features and internal nodes. Then compare it with tensor `B` which is set to the value of each internal node (orange) to create the tensor `input path` that represents the path from input to node. In this case, the tree model has 4 conditions and the input vector is 5, therefore, the shape of tensor `A` is 5x4 and tensor B is 1x4.

<p align="center">
<img src="https://github.com/microsoft/hummingbird/raw/main/website/images/3-matrix.png" width=450 >
</p>

**Step 2:** The `input path` tensor will be multiplied with tensor `C` that captures whether the internal node is a parent of that internal node, and if so, whether it is in the left or right sub-tree (left = 1, right =-1, otherwise =0) and then check the equals with tensor `D` that captures the count of the left child of its parent in the path from a leaf node to the tree root to create the tenor output path that represents the path from node to output. In this case, this tree model has 5 outputs with 4 conditions, therefore, the shape of tensor `C` is 4x5 and tensor `D` is 1x5.

<p align="center">
<img src="https://github.com/microsoft/hummingbird/raw/main/website/images/4-matrixnext.png" width=450 >
</p>

**Step 3:** The `output path` will be multiplied with tensor `E` that captures the mapping between leaf nodes to infer the final prediction. In this case, tree model has 5 outputs, therefore, shape of tensor `E` is 5x1.

<p align="center">
<img src="https://github.com/microsoft/hummingbird/raw/main/website/images/5-singletensor.png" width=450>
</p>

And now Hummingbird has compiled a tree-based model using the GEMM strategy!  For more details, please see [Figure 3](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf) of our paper.


_Thank you to [Chien Vu](https://www.linkedin.com/in/vumichien/) for contributing the graphics and descriptions in his [blog](https://towardsdatascience.com/standardizing-traditional-machine-learning-pipelines-to-tensor-computation-using-hummingbird-7a0b3168670) for this example!_

## Installation

Hummingbird was tested on Python >= 3.5 on Linux, Windows and MacOS machines.  It is recommended to use a virtual environment (See: [python3 venv doc](https://docs.python.org/3/tutorial/venv.html) or [Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).)

Hummingbird requires PyTorch >= 1.4.0. Please go [here](https://pytorch.org/) for instructions on how to install PyTorch based on your platform and hardware.

Once PyTorch is installed, you can get Hummingbird from pip with:
```
pip install hummingbird-ml
```

If you require the optional dependencies lightgbm and xgboost, you can use:
```
pip install hummingbird-ml[extra]
```


See also [Troubleshooting](TROUBLESHOOTING.md) for common problems.

## Examples

See the [notebooks](notebooks) section for examples that demonstrate use and speedups.

In general, Hummingbird syntax is very intuitive and minimal. To run your traditional ML model on DNN frameworks, you only need to `import hummingbird.ml` and add `convert(model, 'dnn_framework')` to your code. Below is an example using a [scikit-learn random forest](https://scikit-learn.org/stable/modules/ensemble.html#forest) model and [PyTorch](https://pytorch.org/) as target framework.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from hummingbird.ml import convert

# Create some random data for binary classification
num_classes = 2
X = np.random.rand(100000, 28)
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

# Save the model
model.save('hb_model')

# Load the model back
model = hummingbird.ml.load('hb_model')
```

# Documentation

The API documentation is [here](https://microsoft.github.io/hummingbird/).

You can also read about Hummingbird in our blog post [here](https://azuredata.microsoft.com/articles/ebd95ec0-1eae-44a3-90f5-c11f5c916d15).

For more details on the vision and on the technical details related to Hummingbird, please check our papers:

* [A Tensor Compiler for Unified Machine Learning Prediction Serving](https://arxiv.org/abs/2010.04804). Supun Nakandala, Karla Saur, Gyeong-In Yu, Konstantinos Karanasos, Carlo Curino, Markus Weimer, Matteo Interlandi. To appear at OSDI 2020.
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
