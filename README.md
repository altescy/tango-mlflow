# tango-mlflow

[![Actions Status](https://github.com/altescy/tango-mlflow/workflows/CI/badge.svg)](https://github.com/altescy/tango-mlflow/actions/workflows/ci.yml)
[![Python version](https://img.shields.io/pypi/pyversions/tango-mlflow)](https://github.com/altescy/tango-mlflow)
[![pypi version](https://img.shields.io/pypi/v/tango-mlflow)](https://pypi.org/project/tango-mlflow/)
[![License](https://img.shields.io/github/license/altescy/tango-mlflow)](https://github.com/altescy/tango-mlflow/blob/main/LICENSE)

MLflow integration for [ai2-tango](https://github.com/allenai/tango)

## Introduction

`tango-mlflow` is a Python library that connects the [tango](https://github.com/allenai/tango) to [MLflow](https://mlflow.org/).
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

```bash
pip install tango-mlflow[all]
```

## Usage

You can use the `MLFlowWorkspace` with command line arguments as follows:

```bash
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

## Examples

- Basic example: [examples/euler](https://github.com/altescy/tango-mlflow/tree/main/examples/euler)
- Hyper parameter tuning with [Optuna](https://optuna.org/): [examples/breast_cancer](https://github.com/altescy/tango-mlflow/tree/main/examples/breast_cancer)
