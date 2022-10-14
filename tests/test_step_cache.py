from pathlib import Path
from unittest import mock

import mlflow
from tango.step import Step

from tango_mlflow.step_cache import MLFlowStepCache

MLFLOW_EXPERIMENT_NAME = "tango-mlflow-testing"


class SomeFaceStep(Step):
    DETERMINISTIC = True
    CACHABLE = True

    def run(self) -> int:  # type: ignore[override]
        return 1


def test_step_cache_artifact_not_found(tmp_path: Path) -> None:
    tango_cache_dir = tmp_path / "tango_cache"
    mlflow_tracking_uri = f"file://{tmp_path.absolute() / 'mlruns'}"
    with mock.patch("tango_mlflow.step_cache.tango_cache_dir", return_value=tango_cache_dir):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        step = SomeFaceStep(step_name="hi there")
        step_cache = MLFlowStepCache(experiment_name=MLFLOW_EXPERIMENT_NAME)
        print(list(tmp_path.glob("**/*")))
        assert step not in step_cache


def test_step_cache_can_store_result(tmp_path: Path) -> None:
    tango_cache_dir = tmp_path / "tango_cache"
    mlflow_tracking_uri = f"file://{tmp_path.absolute() / 'mlruns'}"
    with mock.patch("tango_mlflow.step_cache.tango_cache_dir", return_value=tango_cache_dir):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run():
            step = SomeFaceStep(step_name="hi there")
            step_cache = MLFlowStepCache(experiment_name=MLFLOW_EXPERIMENT_NAME)
            step_cache[step] = step.result()
        assert step in step_cache
        assert step_cache[step] == 1
