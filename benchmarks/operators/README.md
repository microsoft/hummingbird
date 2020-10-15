# Operators Experiments

This directory contains the script to reproduce the experiments of Section 6.1.2 of the paper **A Tensor Compiler for Unified Machine Learning Prediction Serving**. This script is configured to run _sklearn_ and compare it against _onnx-ml_, _torchscript_ and _onnx_ (the last 2 using Hummingbird), for the _iris_ dataset over 1 core, and with batch of 1M.

- `python run.py` will run the benchmarks for CPU
- `python run.py -gpu` will run the benchmarks for GPU