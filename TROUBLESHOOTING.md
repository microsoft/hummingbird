# Troubleshooting Common Problems:

### Running PyTorch on GPU:

In order to run Hummingbird on PyTorch with GPU, you will need to `python -m pip uninstall torch` and re-install pytorch using the CUDA version of your machine.

### Installation Issues with External Libraries:


#### MacOS
* *xgboost installation:*  Ex:  `./xgboost/build-python.sh: line 21: cmake: command not found`
  * Install `cmake` (Ex: `brew install cmake`).


* *lightgbm installation:* `OSError: dlopen(lib_lightgbm.so, 6): Library not loaded: ...libomp.dylib`
    * There is a fixed issue with lgbm and MacOS.  See [LightGBM#1369](https://github.com/Microsoft/LightGBM/issues/1369).
    * see also our [build file](https://github.com/microsoft/hummingbird/blob/main/.github/workflows/pythonapp.yml) with `brew install libomp`

#### Linux
 * With Yum-based Linux systems, there is an issue installing LightGBM: `OSError: libgomp.so.1: cannot open shared object file: No such file or directory`
   * Install `libgomp` with `yum install libgomp`

#### Windows
* *Pytorch installation:* ` ERROR: Could not find a version that satisfies the requirement torch>=1.4.0`).
    * Install PyTorch manually by following the instructions on PyTorch [website](https://pytorch.org/).