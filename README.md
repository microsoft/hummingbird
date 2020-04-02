# Hummingbird

![](https://github.com/microsoft/hummingbird/workflows/Python%20application/badge.svg?branch=develop)

## Introduction
*hummingbird* converts [scikit-learn](https://scikit-learn.org/stable/) models to [PyTorch](https://pytorch.org/). Once in the PyTorch format, <!--you can further convert to [ONNX](https://github.com/onnx/onnx) or [TorchScript](https://pytorch.org/docs/stable/jit.html), and --> you can run the models on GPU for high performance native scoring. For full details, see [our paper](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf). 

Currently we support [these](https://github.com/microsoft/hummingbird/blob/develop/hummingbird/_supported_operators.py#L26) tree-based classifiers and regressors.

## Installation

This was tested on Python 3.7.
```
python setup.py install
```

## Examples

See the [notebooks](notebooks) section for examples.  In general, the syntax is very similar to [skl2onnx](https://github.com/onnx/sklearn-onnx), as hummingbird started as a fork of that project.
```
from hummingbird import convert_sklearn

model = pickle.load(open("my-skl-model.pkl", "rb"))
pytorch_model = convert_sklearn(model,[("input", FloatTensorType([200000, 28]))])
```

## Developing

This project uses [pre-commit](https://pre-commit.com/) hooks. (Run  `pip install pre-commit` if you don't already have this.)

To begin, run `pre-commit install` to install pre-commit into your git hooks.

And before you commit, you can run it like this `pre-commit run --all-files` and should see output such as:

```
black....................................................................Passed
Flake8...................................................................Passed
...
Don't commit to branch...................................................Passed
```

If you have installed your pre-commit hooks successfully, you should see something like this if you
try to commit something non-conformant:
```
# git commit -m "testing"
black....................................................................Failed
- hook id: black
- files were modified by this hook

reformatted hummingbird/convert.py
All done!
1 file reformatted.
```

## License

[MIT License](https://github.com/microsoft/hummingbird/blob/master/LICENSE)


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
