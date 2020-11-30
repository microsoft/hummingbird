# Trees Benchmarks

This directory contains the scripts for benchamrking tree implementations. The scripts are configured to run _sklearn_ and compare it against _onnx-ml_, _pytorch_, _torchscript_, and _onnx_ (the last 3 using Hummingbird), for datasets [fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud), [year](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd), [covtype](https://archive.ics.uci.edu/ml/datasets/covertype), [epsilon](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [higgs](https://archive.ics.uci.edu/ml/datasets/HIGGS), and [airline](http://kt.ijs.si/elena_ikonomovska/data.html), and with batch of 10k.

The scripts will automatically download datasets as needed using wget or the [Kaggle API](https://github.com/Kaggle/kaggle-api). To use the kaggle datasets you will need a valid kaggle account and API token. Please follow the instructions in the link to setup the Kaggle API.

The scipts can be easily adapted to reproduce the experiments of Section 6.1.1 of the paper **A Tensor Compiler for Unified Machine Learning Prediction Serving**.

 The following script runs the experiments on CPU for the _fraud_, _year_, _covtype_, _epsilon_:

 ```
 python run.py -dataset fraud,year,covtype,epsilon
 ```

 While the following script is for GPU execution:

 ```
 python run.py -dataset fraud,year,covtype,epsilon -gpu
 ```

  As starting point we suggest to use the above 4 datasets (skipping 'higgs'/'airline') because the complete script (which can be run with just `python run.py`) over all backends and datasets takes more than one day to complete. After the script is run for the first time, the datasets and trained models are cached (in `datasets` and `models`, respectively), so that following executions will be faster. Serveral other arguments can be changed in the script (e.g., batch size, number of trees, etc.).

  The output of the above commands is a `json` file with entries as follows:

  ```
  "higgs": {
    "lgbm": {
      "hb-pytorch": {
        "conversion_time": "7.474999000000025",
        "is_same_output": false,
        "peak_mem": "3396.64453125",
        "prediction_time": "158.97681199999988"
      },
      "hb-torchscript": {
        "conversion_time": "994.1901349999998",
        "is_same_output": true,
        "peak_mem": "3402.48046875",
        "prediction_time": "148.99096800000007"
      },
      "peak_mem": 3329.22265625,
      "prediction_time": 195.27247699999998,
      "train_accuracy": "{'Accuracy': 0.7341090909090909, 'Log_Loss': 3.0611895593053973, 'AUC': 0.7233984192431102, 'Precision': 0.691083476772121, 'Recall': 0.9013004340496611}",
      "train_time": "1858.368385"
    }
  }
  ```
This entry if for the `higgs` dataset, `lightgbm` algorithm. It reports the training time and accuracy (if the mode is not cached), and prediction time (195.25) in seconds, as well the peak memory used. The baseline is then compared against Hummingbird with pytorch (`hb-pytorch`) and torchscript (`hb-torchscript`) backends. The entry `is_same_output` specifies whether the results of the translated models match those of the baseline (up to a tolerange of 10^-6). In the result is false, you can re-run the script with `-validate` flag on to check the percentage of wrong results.
