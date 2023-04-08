# tango-mlflow

[![Actions Status](https://github.com/altescy/tango-mlflow/workflows/CI/badge.svg)](https://github.com/altescy/tango-mlflow/actions/workflows/ci.yml)
[![Python version](https://img.shields.io/pypi/pyversions/tango-mlflow)](https://github.com/altescy/tango-mlflow)
[![pypi version](https://img.shields.io/pypi/v/tango-mlflow)](https://pypi.org/project/tango-mlflow/)
[![License](https://img.shields.io/github/license/altescy/tango-mlflow)](https://github.com/altescy/tango-mlflow/blob/main/LICENSE)

`tango-mlflow` integrates your [tango](https://github.com/allenai/tango) project with [MLflow](https://mlflow.org/), which enables you to log entire configs and artifacts of your tango run.

## Installation

```bash
pip install tango-mlflow[all]
```

## Usage

You can use `MLFlowWorkspace` by using command line arguments like below:

```bash
tango run --workspace mlflow://your_experiment_name --include-package tango-mlflow
```

or write your configuration in `tango.yml`:

```tango.yml
workspace:
  type: mlflow
  experiment_name: your_experiment_name

include_package:
  - tango_mlflow
```
