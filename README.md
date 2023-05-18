# tango-mlflow

[![Actions Status](https://github.com/altescy/tango-mlflow/workflows/CI/badge.svg)](https://github.com/altescy/tango-mlflow/actions/workflows/ci.yml)
[![Python version](https://img.shields.io/pypi/pyversions/tango-mlflow)](https://github.com/altescy/tango-mlflow)
[![pypi version](https://img.shields.io/pypi/v/tango-mlflow)](https://pypi.org/project/tango-mlflow/)
[![License](https://img.shields.io/github/license/altescy/tango-mlflow)](https://github.com/altescy/tango-mlflow/blob/main/LICENSE)

MLflow integration for [ai2-tango](https://github.com/allenai/tango)

## Introduction

`tango-mlflow` is a Python library that connects the [Tango](https://github.com/allenai/tango) to [MLflow](https://mlflow.org/).
Tango, developed by AllenAI, is a flexible pipeline library for managing experiments and caching outputs of each step.
MLflow, on the other hand, is a platform that helps manage the Machine Learning lifecycle, including experimentation, reproducibility, and deployment.
This integration enables you to store and manage complete experimental settings and artifacts of your Tango run in MLflow, thus enhancing your experiment tracking capabilities.

Here's a screenshot of the MLflow interface when executing with `tango-mlflow`:

![239279085-79ca2c6b-f9a1-49aa-a301-bb502a2340d2](https://github.com/altescy/tango-mlflow/assets/16734471/d41c6d08-faec-4ea0-8b23-fbc8c6147626)

- **Tango run**: The top-level run in MLflow corresponds to a single execution in Tango, and its name matches the execution name in Tango.
- **Tango steps**: Results of each step in Tango are recorded as nested runs in MLflow. The names of these nested runs correspond to the names of the steps in Tango.
- **Parameters and Metrics**: The entire settings for the execution, as well as the parameters for each step, are automatically logged in MLflow.
- **Artifacts and Caching**: The cached outputs of each step's execution are saved as artifacts under the corresponding MLflow run. These can be reused when executing Tango, enhancing efficiency and reproducibility.

## Installation

Install `tango-mlflow` by running the following command in your terminal:

```shell
pip install tango-mlflow[all]
```

## Usage

You can use the `MLFlowWorkspace` with command line arguments as follows:

```shell
tango run --workspace mlflow://your_experiment_name --include-package tango_mlflow
```

Alternatively, you can define your configuration in a `tango.yml` file:

```tango.yml
workspace:
  type: mlflow
  experiment_name: your_experiment_name

include_package:
  - tango_mlflow
```

In the above `tango.yml` configuration, `type` specifies the workspace type as MLflow, and `experiment_name` sets the name of your experiment in MLflow.
The `include_package` field needs to include the `tango_mlflow` package to use `tango-mlflow` functionality.

Remember to replace `your_experiment_name` with the name of your specific experiment.

To log runs remotely, set the `MLFLOW_TRACKING_URI` environment variable to a tracking server’s URI like below:

```shell
export MLFLOW_TRACKING_URI=https://mlflow.example.com
```

## Functionalities

### Logging metrics into MLflow

The `tango-mlflow` package provides the `MLflowStep` class, which allows you to easily log the results of each step execution to MLflow.

```python
from tango_mlflow.step import MLflowStep

class TrainModel(MLflowStep):
    def run(self, **params):

        # pre-process...

        for epoch in range(max_epochs):
            loss = train_epoch(...)
            metrics = evaluate_model(...)
            # log metrics with mlflow_logger
            for name, value in metrics.items():
                self.mlflow_logger.log_metric(name, value, step=epoch)

        # post-process...
```

In the example above, the `TrainModel` step inherits from `MLflowStep`.
Inside the step, you can directly record metrics to the corresponding MLflow run by invoking `self.mlflow_logger.log_metric(...)`.

Please note, this functionality must be used in conjunction with `MLFlowWorkspace`.

### Summarizing Tango run metrics

You can specify a step to record its returned metrics as representative values of the Tango run by setting the class variable `MLFLOW_SUMMARY = True`.
This feature enables you to conveniently view metrics for each Tango run directly in the MLflow interface

```python
class EvaluateModel(Step):
    MLFLOW_SUMMARY = True  # Enables MLflow summary!

    def run(self, ...) -> dict[str, float]:
        # compute metrics ...
        return metrics
```

In the example above, the `EvaluateModel` step returns metrics that are logged as the representative values for that Tango run. These metrics are then recorded in the corresponding (top-level) MLflow run.

Please note the following requirements:
- The return value of a step where `MLFLOW_SUMMARY = True` is set must always be `dict[str, float]`.
- You don't necessarily need to inherit from `MLflowStep` to use `MLFLOW_SUMMARY`.

### Tuning hyperparameters with Optuna

`tango-mlflow` also provides the `tango-mlflow tune` command for tuning hyperparameters with [Optuna](https://optuna.org/).
For more details, please refer to the [examples/breast_cancer](https://github.com/altescy/tango-mlflow/tree/main/examples/breast_cancer) directory.

## Examples

- Basic example: [examples/euler](https://github.com/altescy/tango-mlflow/tree/main/examples/euler)
- Hyper parameter tuning with [Optuna](https://optuna.org/): [examples/breast_cancer](https://github.com/altescy/tango-mlflow/tree/main/examples/breast_cancer)
