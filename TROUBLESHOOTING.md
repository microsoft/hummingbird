# Troubleshooting Common Problems:


### Install:
* During `pip install`, xgboost error:  (Ex:  `./xgboost/build-python.sh: line 21: cmake: command not found`).
    * Install `cmake` (Ex: `brew install cmake` or `apt install cmake`).


*  During `pip install`, an error with lightgbm: (Ex: `OSError: dlopen(lib_lightgbm.so, 6): Library not loaded: ...libomp.dylib`).
    * There is a fixed issue with lgbm and MacOS.  See [LightGBM#1369](https://github.com/Microsoft/LightGBM/issues/1369).

* During `pip install` on Windows, an error with PyTorch (Ex: ` ERROR: Could not find a version that satisfies the requirement torch>=1.4.0`).
    * Install PyTorch manually by following the instructions on PyTorch [website](https://pytorch.org/).