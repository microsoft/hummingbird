# Pipelines Experiments

This directory contains the script to reproduce the experiments of Section 6.3 of the paper **A Tensor Compiler for Unified Machine Learning Prediction Serving**.
There are two main scripts to run for this experiment:
- `openml_pipelines.py` is used to download and train all the scikit-learn pipelines of the [openML-CC18](https://docs.openml.org/benchmark/#openml-cc18) benchmark.
- `run.py` is used to run evaluate the performance of scikit-learn and Hummingbird over the trained pipelines.

This experiment is composed of two steps.

## Step1: Generating the Pipelines
The first step is to generate the prediction pipelines. This can achieved by running:

```
python openml_pipelines.py | tee openml-cc18.log
```
This script will take some time to run (several hours).

While executing, this script will log the number of successfully trained pipelines, as well as additional statistics. Once completed, you can check the `openml-cc18.log` file to see the statistics. Per task statistics are logged into the relative folder.

## Running the experiment
Once the first step is completed, in the second step we evaluate the scoring time of the generated pipelines, and compared the speed ups introduced by Hummingbird compared to scikit-learn. This experiment can be executed both on CPU and GPU, and in both cases it takes about an hour.

 The following script runs the experiments on CPU:

 ```
 python run.py
 ```

 While the following script is for GPU execution:

 ```
 python run.py -gpu
 ```

 The `run.py` script contains few other options that can be explored: e.g., by default the scikit-learn pipelines are compared against torch, but all the other supported backends can be tested as well.
