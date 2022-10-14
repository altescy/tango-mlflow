import json
import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterable, Iterator, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult

import mlflow
import petname
import pytz  # type: ignore
from mlflow.entities import Run as MLFlowRun
from mlflow.tracking.client import MlflowClient
from tango.common.exceptions import StepStateError
from tango.common.file_lock import FileLock
from tango.common.logging import file_handler
from tango.common.util import exception_to_string, tango_cache_dir, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace

from tango_mlflow.step_cache import MLFlowStepCache
from tango_mlflow.util import RunKind, flatten_dict

logger = logging.getLogger(__name__)

T = TypeVar("T")


@Workspace.register("mlflow")
class MLFlowWorkspace(Workspace):
    def __init__(self, experiment_name: str) -> None:
        mlflow.set_experiment(experiment_name)
        super().__init__()  # type: ignore[no-untyped-call]
        self.experiment_name = experiment_name
        self.cache = MLFlowStepCache(experiment_name=self.experiment_name)
        self.steps_dir = tango_cache_dir() / "mlflow_workspace"
        self.locks: Dict[Step, FileLock] = {}
        self._running_step_info: Dict[str, StepInfo] = {}

    def __getstate__(self) -> Dict[str, Any]:
        out = super().__getstate__()  # type: ignore[no-untyped-call]
        out["locks"] = {}
        return cast(Dict[str, Any], out)

    @property
    def url(self) -> str:
        return f"mlflow://{self.experiment_name}"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> "MLFlowWorkspace":
        experiment_name = parsed_url.netloc
        return cls(experiment_name=experiment_name)

    @property
    def step_cache(self) -> StepCache:
        return self.cache

    @property
    def mlflow_client(self) -> MlflowClient:
        return MlflowClient()

    @property
    def mlflow_tracking_uri(self) -> str:
        return str(mlflow.get_tracking_uri()).rstrip("/")

    def _get_unique_id(self, step_or_unique_id: Union[Step, str]) -> str:
        if isinstance(step_or_unique_id, Step):
            return step_or_unique_id.unique_id
        return step_or_unique_id

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        unique_id = self._get_unique_id(step_or_unique_id)
        path = self.steps_dir / unique_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, step: Step) -> Path:
        path = self.step_dir(step) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        unique_id = self._get_unique_id(step_or_unique_id)
        if unique_id in self._running_step_info:
            return self._running_step_info[unique_id]
        step_info = self._get_updated_step_info(
            unique_id,
            step_name=step_or_unique_id.name if isinstance(step_or_unique_id, Step) else None,
        )
        if step_info is None:
            raise KeyError(step_or_unique_id)
        return step_info

    def step_starting(self, step: Step) -> None:
        lock_path = self.step_dir(step) / "lock"
        lock = FileLock(lock_path, read_only_ok=True)
        lock.acquire_with_updates(desc=f"acquiring lock for '{step.name}'")
        self.locks[step] = lock

        step_info = self._get_updated_step_info(step.unique_id) or StepInfo.new_from_step(step)
        if step_info.state not in {StepState.INCOMPLETE, StepState.FAILED, StepState.UNCACHEABLE}:
            raise StepStateError(
                step,
                step_info.state,
                context="If you are certain the step is not running somewhere else, delete the lock "
                f"file at {lock_path}.",
            )

        try:
            mlflow_run = mlflow.start_run(
                run_name=step_info.step_name,
                nested=True,
                tags={
                    "job_type": RunKind.STEP.value,
                    "step_name": step.name,
                    "step_id": step.unique_id,
                },
                description="\n".join(
                    (
                        f"### Tango step ***{step.name}***",
                        f"- type: {step_info.step_class_name}",
                        f"- ID: {step.unique_id}",
                    )
                ),
            )

            logger.info(
                "Tracking '%s' step on MLflow: %s/#/experiments/%s/runs/%s",
                step.name,
                self.mlflow_tracking_uri,
                mlflow_run.info.experiment_id,
                mlflow_run.info.run_id,
            )

            step_info.start_time = utc_now_datetime()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None
            mlflow.log_params(flatten_dict(step.config))
            mlflow.log_dict(step_info.to_json_dict(), "step_info.json")
            self._running_step_info[step.unique_id] = step_info
        except:  # noqa: E722
            lock.release()
            del self.locks[step]
            raise

    def step_finished(self, step: Step, result: T) -> T:
        mlflow_run = mlflow.active_run()
        if mlflow_run is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.step_finished() called outside of a W&B run. "
                f"Did you forget to call {self.__class__.__name__}.step_starting() first?"
            )

        step_info = self._running_step_info.get(step.unique_id) or self._get_updated_step_info(step.unique_id)
        if step_info is None:
            raise KeyError(step.unique_id)

        try:
            if step.cache_results:
                self.step_cache[step] = result
                if hasattr(result, "__next__"):
                    assert isinstance(result, Iterator)
                    # Caching the iterator will consume it, so we write it to the
                    # cache and then read from the cache for the return value.
                    result = self.step_cache[step]
                step_info.result_location = mlflow_run.info.artifact_uri
            else:
                self.cache.create_step_result_artifact(step)

            step_info.end_time = utc_now_datetime()
            mlflow.log_dict(step_info.to_json_dict(), "step_info.json")

            self.mlflow_client.set_terminated(mlflow_run.info.run_id)
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        mlflow_run = mlflow.active_run()
        if mlflow_run is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.step_finished() called outside of a MLflow run. "
                f"Did you forget to call {self.__class__.__name__}.step_starting() first?"
            )

        step_info = self._running_step_info.get(step.unique_id) or self._get_updated_step_info(step.unique_id)
        if step_info is None:
            raise KeyError(step.unique_id)

        try:
            if step_info.state != StepState.RUNNING:
                raise StepStateError(step, step_info.state)
            step_info.end_time = utc_now_datetime()
            step_info.error = exception_to_string(e)
            mlflow.log_dict(step_info.to_json_dict(), "step_info.json")
            mlflow.end_run(status="FAILED")
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        step_ids: Dict[str, bool] = {}
        step_name_to_info: Dict[str, Dict[str, Any]] = {}
        for step in all_steps:
            step_info = StepInfo.new_from_step(step)
            step_name_to_info[step.name] = {k: v for k, v in step_info.to_json_dict().items() if v is not None}
            step_ids[step.unique_id] = True

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        description = "# Tango run"
        cacheable_steps = {step for step in all_steps if step.cache_results}
        if cacheable_steps:
            description += "\nCacheable steps:\n"
            for step in sorted(cacheable_steps, key=lambda step: step.name):
                description += f"- {step.name}"
                dependencies = step.dependencies
                if dependencies:
                    description += ", depends on: " + ", ".join(
                        sorted(
                            [f"'{dep.name}'" for dep in dependencies],
                        )
                    )
                description += f"\n  - {step.unique_id}\n"

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        while name is None or self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.job_type = '{RunKind.TANGO_RUN.value}' AND tags.mlflow.runName = '{name}'",
        ):
            name = petname.generate()

        mlflow_run = mlflow.start_run(
            run_name=name,
            tags={"job_type": RunKind.TANGO_RUN.value},
            description=description,
        )

        logger.info("Registring run %s with MLflow", name)
        logger.info(
            "View run at: %s/#/experiments/%s/runs/%s",
            self.mlflow_tracking_uri,
            mlflow_run.info.experiment_id,
            mlflow_run.info.run_id,
        )

        mlflow.log_params(flatten_dict({step.name: step.config for step in all_steps}))
        mlflow.log_params(flatten_dict({"_step_ids": step_ids}))
        mlflow.log_dict({"steps": step_name_to_info, "_step_ids": step_ids}, "steps.json")

        run = self.registered_run(mlflow_run.data.tags["mlflow.runName"])

        def on_run_end(_: Run) -> None:
            mlflow_run = mlflow.active_run()
            if mlflow_run is not None:
                run = self._get_run_from_mlflow_run(mlflow_run)
                is_finished = all(step_info.error is None for step_info in run.steps.values())
                mlflow.end_run(status="FINISHED" if is_finished else "FAILED")

        setattr(run, "__del__", on_run_end)

        return run

    def registered_runs(self) -> Dict[str, Run]:
        runs: Dict[str, Run] = {}
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        page_token: Optional[str] = None
        while True:
            matching_runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.job_type = '{RunKind.TANGO_RUN.value}'",
                page_token=page_token,
            )
            for mlflow_run in matching_runs:
                run_name = mlflow_run.data.tags["mlflow.runName"]
                runs[run_name] = self._get_run_from_mlflow_run(mlflow_run)
            if matching_runs.token is None:
                break
            page_token = matching_runs.token
        return runs

    def registered_run(self, name: str) -> Run:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        matching_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.job_type = '{RunKind.TANGO_RUN.value}' AND tags.mlflow.runName = '{name}'",
        )
        if not matching_runs:
            raise KeyError(f"Run '{name}' not found in workspace")
        elif len(matching_runs) > 1:
            raise ValueError(f"Found more than one run name '{name}' in MLflow experiment")
        return self._get_run_from_mlflow_run(matching_runs[0])

    def _get_run_from_mlflow_run(self, mlflow_run: MLFlowRun) -> Run:
        step_name_to_info: Dict[str, StepInfo] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(self.mlflow_client.download_artifacts(mlflow_run.info.run_id, "", temp_dir))
            steps_json_path = artifact_path / "steps.json"
            with open(steps_json_path, "r") as jsonfile:
                for step_name, step_info_dict in json.load(jsonfile)["steps"].items():
                    step_info = StepInfo.from_json_dict(step_info_dict)
                    if step_info.cacheable:
                        updated_step_info = self._get_updated_step_info(step_info.unique_id, step_name=step_name)
                        if updated_step_info is not None:
                            step_info = updated_step_info
                    step_name_to_info[step_name] = step_info

        run_name = mlflow_run.data.tags["mlflow.runName"]
        return Run(
            name=run_name,
            steps=step_name_to_info,
            start_date=datetime.fromtimestamp(mlflow_run.info.start_time / 1000, tz=pytz.utc),
        )

    def _get_updated_step_info(self, step_id: str, step_name: Optional[str] = None) -> Optional[StepInfo]:
        experiment = self.mlflow_client.get_experiment_by_name(self.experiment_name)
        filter_string = f"tags.job_type = '{RunKind.STEP.value}' AND tags.step_id = '{step_id}'"
        if step_name is not None:
            filter_string += f" AND tags.step_name = '{step_name}'"
        matching_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
        )

        def load_step_info(mlflow_run: MLFlowRun) -> StepInfo:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = self.mlflow_client.download_artifacts(mlflow_run.info.run_id, "step_info.json", temp_dir)
                with open(path, "r") as jsonfile:
                    return StepInfo.from_json_dict(json.load(jsonfile))

        for mlflow_run in matching_runs:
            step_info = load_step_info(mlflow_run)
            if step_info.start_time is None:
                step_info.start_time = datetime.fromtimestamp(mlflow_run.info.start_time / 1000, tz=pytz.utc)
            if mlflow_run.info.status in ("FINISHED", "FAILED"):
                if step_info.end_time is None:
                    step_info.end_time = datetime.fromtimestamp(mlflow_run.info.end_time / 1000, tz=pytz.utc)
                if mlflow_run.info.status == "FAILED" and step_info.error is None:
                    step_info.error = "Exception"
            return step_info

        def load_step_info_dict(mlflow_run: MLFlowRun) -> Dict[str, Any]:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = self.mlflow_client.download_artifacts(mlflow_run.info.run_id, "steps.json", temp_dir)
                with open(path, "r") as jsonfile:
                    step_info_dict = json.load(jsonfile)
                    assert isinstance(step_info_dict, dict)
                    return step_info_dict

        filter_string = f"tags.job_type = '{RunKind.TANGO_RUN.value}' AND params._step_ids.\"{step_id}\" = 'True'"
        matching_runs = self.mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_string,
        )
        for mlflow_run in matching_runs:
            step_info_dict = load_step_info_dict(mlflow_run)
            if step_name is not None:
                step_info_data = step_info_dict["steps"][step_name]
            else:
                step_info_data = next(d for d in step_info_dict["steps"].values() if d["unique_id"] == step_id)
            step_info = StepInfo.from_json_dict(step_info_data)
            return step_info

        return None

    def capture_logs_for_run(self, name: str) -> ContextManager[None]:
        @contextmanager
        def capture_logs() -> Iterator[None]:
            with tempfile.TemporaryDirectory() as temp_dir:
                log_path = Path(temp_dir) / "out.log"
                try:
                    with file_handler(log_path) as handler:
                        yield handler
                finally:
                    mlflow_run = mlflow.active_run()
                    parent_run_id = mlflow_run.data.tags.get("mlflow.parentRunId")
                    if parent_run_id:
                        mlflow_run = self.mlflow_client.get_run(parent_run_id)
                    self.mlflow_client.log_artifact(mlflow_run.info.run_id, str(log_path))

        return capture_logs()
