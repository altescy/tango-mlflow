from pathlib import Path
from unittest import mock

import mlflow
from tango.step import Step

from tango_mlflow.step_cache import MlflowStepCache
from tango_mlflow.util import RunKind

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
        step_cache = MlflowStepCache(experiment_name=MLFLOW_EXPERIMENT_NAME)
        assert step not in step_cache


def test_step_cache_can_store_result(tmp_path: Path) -> None:
    tango_cache_dir = tmp_path / "tango_cache"
    mlflow_tracking_uri = f"file://{tmp_path.absolute() / 'mlruns'}"
    with mock.patch("tango_mlflow.step_cache.tango_cache_dir", return_value=tango_cache_dir):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        step_name = "hi there"
        step_id = "test_step"
        with mlflow.start_run(
            tags={
                "job_type": RunKind.STEP.value,
                "step_name": step_name,
                "step_id": step_id,
            },
        ):
            step = SomeFaceStep(step_name=step_name, step_unique_id_override=step_id)
            step_cache = MlflowStepCache(experiment_name=MLFLOW_EXPERIMENT_NAME)
            step_cache[step] = step.result()
        assert step in step_cache
        assert step_cache[step] == 1
