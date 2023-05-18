import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import mlflow
from mlflow.entities import Run as MLFlowRun
from mlflow.tracking.client import MlflowClient
from tango.common.aliases import PathOrStr
from tango.common.file_lock import AcquireReturnProxy, FileLock
from tango.common.params import Params
from tango.common.util import tango_cache_dir
from tango.step import Step
from tango.step_cache import CacheMetadata, StepCache
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_info import StepInfo

from tango_mlflow.util import (
    RunKind,
    get_mlflow_local_artifact_storage_path,
    get_mlflow_run_by_tango_step,
    get_mlflow_runs,
)

logger = logging.getLogger(__name__)


@StepCache.register("mlflow")
class MLFlowStepCache(LocalStepCache):
    def __init__(self, experiment_name: str) -> None:
        super().__init__(tango_cache_dir() / "mlflow_cache")
        self.experiment_name = experiment_name

    @property
    def mlflow_client(self) -> MlflowClient:
        return MlflowClient()

    def step_dir(self, step: Union[Step, StepInfo, str]) -> Path:
        mlflow_run = get_mlflow_run_by_tango_step(
            self.mlflow_client,
            self.experiment_name,
            tango_step=step,
        )
        if mlflow_run is None:
            return super().step_dir(step)

        mlflow_local_artifact_storage_path = get_mlflow_local_artifact_storage_path(mlflow_run)

        if mlflow_local_artifact_storage_path is None:
            return super().step_dir(step)

        return mlflow_local_artifact_storage_path

    def get_step_result_mlflow_run(self, step: Union[Step, StepInfo]) -> Optional[MLFlowRun]:
        return get_mlflow_run_by_tango_step(
            self.mlflow_client,
            self.experiment_name,
            tango_step=step,
            additional_filter_string="attributes.status = 'FINISHED'",
        )

    def create_step_result_artifact(
        self,
        step: Union[Step, StepInfo],
        objects_dir: Optional[PathOrStr] = None,
    ) -> None:
        mlflow_run = get_mlflow_run_by_tango_step(
            self.mlflow_client,
            self.experiment_name,
            tango_step=step,
        )
        if mlflow_run is None:
            raise RuntimeError(f"No MLflow run found for step: {step.unique_id}")

        if objects_dir is not None:
            self.mlflow_client.log_artifacts(mlflow_run.info.run_id, objects_dir)

    def _acquire_step_lock_file(
        self,
        step: Union[Step, StepInfo],
        read_only_ok: bool = False,
    ) -> AcquireReturnProxy:
        return FileLock(self.step_dir(step).with_suffix(".lock"), read_only_ok=read_only_ok).acquire_with_updates(
            desc=f"acquiring step cache lock for '{step.unique_id}'"
        )

    def __contains__(self, step: Any) -> bool:
        if not isinstance(step, (Step, StepInfo)):
            return False

        cacheable = step.cache_results if isinstance(step, Step) else step.cacheable
        if not cacheable:
            return False

        key = step.unique_id
        if key in self.strong_cache:
            return True
        if key in self.weak_cache:
            return True

        mlflow_run = self.get_step_result_mlflow_run(step)
        return mlflow_run is not None

    def __getitem__(self, step: Union[Step, StepInfo]) -> Any:
        key = step.unique_id

        # Try getting the result from our in-memory caches first.
        result = self._get_from_cache(key)
        if result is not None:
            return result

        def load_and_return() -> Any:
            metadata = CacheMetadata.from_params(Params.from_file(self._metadata_path(step)))
            result = metadata.format.read(self.step_dir(step))
            self._add_to_cache(key, result)
            return result

        # Next check our local on-disk cache
        with self._acquire_step_lock_file(step, read_only_ok=True):
            if self.step_dir(step).is_dir():
                try:
                    return load_and_return()
                except FileNotFoundError:
                    logger.info("Spep cache is incomplete, trying to re-download it from MLflow...")
                    shutil.rmtree(self.step_dir(step))

        # Finally, check MLflow for the corresponding artifact.
        with self._acquire_step_lock_file(step):
            # Make sure the step wasn't cached since the last time we checked (above).
            if self.step_dir(step).is_dir():
                try:
                    return load_and_return()
                except FileNotFoundError:
                    logger.info("Spep cache is incomplete, trying to re-download it from MLflow...")
                    shutil.rmtree(self.step_dir(step))

            mlflow_run = self.get_step_result_mlflow_run(step)
            if mlflow_run is None:
                raise KeyError(step)

            with tempfile.TemporaryDirectory(dir=self.dir, prefix=key) as temp_dir:
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run.info.run_id,
                    artifact_path="",
                    dst_path=temp_dir,
                )
                os.replace(temp_dir, self.step_dir(step))

            return load_and_return()

    def __setitem__(self, step: Step, value: Any) -> None:
        if not step.cache_results:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)
            return

        with self._acquire_step_lock_file(step):
            with tempfile.TemporaryDirectory(dir=self.dir, prefix=step.unique_id) as _temp_dir:
                temp_dir = Path(_temp_dir)
                step.format.write(value, temp_dir)
                metadata = CacheMetadata(step=step.unique_id, format=step.format)
                metadata.to_params().to_file(temp_dir / self.METADATA_FILE_NAME)
                self.create_step_result_artifact(step, temp_dir)
                if self.step_dir(step).is_dir():
                    shutil.rmtree(self.step_dir(step), ignore_errors=True)
                os.replace(temp_dir, self.step_dir(step))

        self._add_to_cache(step.unique_id, value)

    def __len__(self) -> int:
        mlflow_runs = get_mlflow_runs(
            self.mlflow_client,
            self.experiment_name,
            run_kind=RunKind.STEP,
            additional_filter_string="attributes.status = 'FINISHED'",
        )
        return len(mlflow_runs)
