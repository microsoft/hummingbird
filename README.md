# Hummingbird

![](https://github.com/microsoft/hummingbird/workflows/Python%20application/badge.svg?branch=develop)

## Introduction
*sklearn-pytorch* converts [scikit-learn](https://scikit-learn.org/stable/) models to [PyTorch](https://pytorch.org/). Once in the PyTorch format, you can further convert to [ONNX](https://github.com/onnx/onnx) or [TorchScript](https://pytorch.org/docs/stable/jit.html) for high performance native scoring.

## Installation

This was tested on Python 3.7.   (Newer versions such as 3.8 do not yet work with torch==1.3.1).

Note that the xgboost package requires `cmake` on your system.
**TODO other xbg install complications?**

```
python setup.py develop
```
Or you can install from the source with the latest changes.
**#TODO update URL**
```
pip install git+https://github.com/ksaur/hb-dryrun.git
```

## TODO
* explain commit hooks
* explain which pep8 we use
* write commiter guide
* write more documentation, especially explaining overall design

## Developing

This project uses [pre-commit](https://pre-commit.com/) hooks. (Run  `pip install pre-commit` if you don't already have this.)

To begin, run `pre-commit install` to install pre-commit into your git hooks.

And before you commit, you can run it like this `pre-commit run --all-files` and should see output such as:

```
black....................................................................Passed
Flake8...................................................................Passed
Check for added large files..............................................Passed
Check python ast.........................................................Passed
Check for byte-order marker..............................................Passed
Check for merge conflicts................................................Passed
Detect Private Key.......................................................Passed
Trim Trailing Whitespace.................................................Passed
Don't commit to branch...................................................Passed
```

If you have installed your pre-commit hooks successfull, you should see something like this if you
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
**#TODO what license do we want?**
[MIT License](LICENSE)

## Troubleshooting:

* On installing xgboost:  (Ex:  `./xgboost/build-python.sh: line 21: cmake: command not found`)
  * install cmake (Ex: `brew install cmake` or `apt install cmake`)

* `OSError: dlopen(lib_lightgbm.so, 6): Library not loaded: ...libomp.dylib`
 * TODO:


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
