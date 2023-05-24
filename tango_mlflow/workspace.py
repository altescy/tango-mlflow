import atexit
import json
import logging
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, ContextManager, Dict, Iterable, Iterator, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult, parse_qs, quote

import mlflow
import pytz  # type: ignore
from mlflow.entities import Run as MLFlowRun
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from tango.common.exceptions import StepStateError
from tango.common.file_lock import FileLock
from tango.common.logging import file_handler
from tango.common.util import exception_to_string, tango_cache_dir, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace

from tango_mlflow.step import MLflowStep, MLflowSummaryStep
from tango_mlflow.step_cache import MLFlowStepCache
from tango_mlflow.util import (
    RunKind,
    add_mlflow_run_of_tango_run,
    add_mlflow_run_of_tango_step,
    flatten_dict,
    get_mlflow_run_by_tango_run,
    get_mlflow_run_by_tango_step,
    get_mlflow_runs,
    terminate_mlflow_run_of_tango_step,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@Workspace.register("mlflow")
class MLFlowWorkspace(Workspace):
    def __init__(
        self,
        experiment_name: str,
        tags: Optional[Dict[str, str]] = None,
        tracking_uri: Optional[str] = None,
    ) -> None:
        mlflow.set_experiment(experiment_name)
        super().__init__()  # type: ignore[no-untyped-call]
        self.experiment_name = experiment_name
        self.cache = MLFlowStepCache(experiment_name=self.experiment_name)
        self.steps_dir = tango_cache_dir() / "mlflow_workspace"
        self.locks: Dict[Step, FileLock] = {}
        self._running_step_info: Dict[str, StepInfo] = {}
        self._step_id_to_run_name: Dict[str, str] = {}
        self._mlflow_tags = dict(sorted(tags.items())) if tags else {}
        self._mlflow_tracking_uri = tracking_uri
        self._run_status: Dict[str, str] = {}

        if self._mlflow_tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

    def __getstate__(self) -> Dict[str, Any]:
        out = super().__getstate__()  # type: ignore[no-untyped-call]
        out["locks"] = {}
        return cast(Dict[str, Any], out)

    @property
    def url(self) -> str:
        url = f"mlflow://{self.experiment_name}"
        if self._mlflow_tags:
            url += f"?tags={quote(json.dumps(self._mlflow_tags))}"
        if self._mlflow_tracking_uri is not None:
            url += f"&tracking_uri={quote(self._mlflow_tracking_uri)}"
        return url

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> "MLFlowWorkspace":
        queries = parse_qs(parsed_url.query)
        experiment_name = parsed_url.netloc
        tags: Dict[str, Any] = {}
        for json_strng in queries.get("tags", []):
            subtags = json.loads(json_strng)
            if not isinstance(subtags, dict):
                raise ValueError(f"Invalid tags: {subtags}")
            tags.update(subtags)
        tracking_uri: Optional[str] = None
        for uri in queries.get("tracking_uri", []):
            tracking_uri = uri
        return cls(
            experiment_name=experiment_name,
            tags=tags,
            tracking_uri=tracking_uri,
        )

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
            mlflow_run = get_mlflow_run_by_tango_step(self.mlflow_client, self.experiment_name, step_info)
            if not mlflow_run:
                raise RuntimeError(
                    f"Could not find mlflow run for step {step.unique_id}. There is a possibility that"
                    " the run was deleted or modified during the execution. Please try again."
                )
            if step_info.state == StepState.RUNNING:
                raise StepStateError(
                    step,
                    step_info.state,
                    context=(
                        "This step is already running. If you are certain the step is not running somewhere else,"
                        " it seems that this step was accidentally killed in a previous run. In this case, please"
                        f" delete the mlflow run ({mlflow_run.info.run_id}) for this step and try again."
                    ),
                )
            if step_info.state == StepState.COMPLETED:
                raise StepStateError(
                    step,
                    step_info.state,
                    context=(
                        "This step has already been completed in a previous run. Please rerun this step, and the"
                        " cached result will be used. If this error persists, please open an issue from here:"
                        " https://github.com/altescy/tango-mlflow/issues/new"
                    ),
                )
            raise StepStateError(
                step,
                step_info.state,
                context=(
                    "Unknown step state is detected. Please open an issue from here:"
                    " https://github.com/altescy/tango-mlflow/issues/new"
                ),
            )

        try:
            step_info.start_time = utc_now_datetime()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None

            mlflow_run = add_mlflow_run_of_tango_step(
                self.mlflow_client,
                self.experiment_name,
                tango_run=self._step_id_to_run_name[step.unique_id],
                step_info=step_info,
            )

            if isinstance(step, MLflowStep):
                step.setup_mlflow(mlflow_run)

            logger.info(
                "Tracking '%s' step on MLflow: %s/#/experiments/%s/runs/%s",
                step.name,
                self.mlflow_tracking_uri,
                mlflow_run.info.experiment_id,
                mlflow_run.info.run_id,
            )

            self._running_step_info[step.unique_id] = step_info
        except:  # noqa: E722
            lock.release()
            del self.locks[step]
            raise

    def step_finished(self, step: Step, result: T) -> T:
        mlflow_run = get_mlflow_run_by_tango_step(self.mlflow_client, self.experiment_name, step)
        if mlflow_run is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.step_finished() called outside of a MLflow run. "
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
            terminate_mlflow_run_of_tango_step(
                self.mlflow_client,
                self.experiment_name,
                status="FINISHED",
                step_info=step_info,
            )

            # Log the result of summary step to a parent mlflow run
            if isinstance(step, MLflowSummaryStep) and step.MLFLOW_SUMMARY:
                if not isinstance(result, dict):
                    raise ValueError(
                        f"Result value of Step {step.name} with MLFLOW_SUMMARY=True"
                        f"must be a dict, but got {type(result)}"
                    )

                mlflow_run_of_tang_run = get_mlflow_run_by_tango_run(
                    self.mlflow_client,
                    self.experiment_name,
                    tango_run=self._step_id_to_run_name[step.unique_id],
                )
                if mlflow_run_of_tang_run is None:
                    raise RuntimeError(
                        f"Could not find MLflow run for Tango run {self._step_id_to_run_name[step.unique_id]}"
                    )

                for key, value in flatten_dict({step.name: result}).items():
                    self.mlflow_client.log_metric(mlflow_run_of_tang_run.info.run_id, key, value)
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        self._run_status[self._step_id_to_run_name[step.unique_id]] = "FAILED"
        mlflow_run = get_mlflow_run_by_tango_step(
            self.mlflow_client,
            self.experiment_name,
            tango_step=step,
        )
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
            terminate_mlflow_run_of_tango_step(
                self.mlflow_client,
                self.experiment_name,
                status="FAILED",
                step_info=step_info,
            )
        finally:
            self.locks[step].release()
            del self.locks[step]
            if step.unique_id in self._running_step_info:
                del self._running_step_info[step.unique_id]

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        mlflow_run = add_mlflow_run_of_tango_run(
            self.mlflow_client,
            self.experiment_name,
            steps=all_steps,
            run_name=name,
            mlflow_tags=self._mlflow_tags,
        )

        logger.info("Registring run %s with MLflow", name)
        logger.info(
            "View run at: %s/#/experiments/%s/runs/%s",
            self.mlflow_tracking_uri,
            mlflow_run.info.experiment_id,
            mlflow_run.info.run_id,
        )

        run = self.registered_run(mlflow_run.data.tags[MLFLOW_RUN_NAME])

        for step in all_steps:
            self._step_id_to_run_name[step.unique_id] = run.name

        class RunTerminator:
            workspace = self

            def __del__(self) -> None:
                self.workspace.terminate_run(run)

        setattr(run, "_tango_mlflow_run_terminator", RunTerminator())
        atexit.register(self.terminate_run, run)

        return run

    def terminate_run(self, run: Union[str, Run]) -> None:
        run_name = run if isinstance(run, str) else run.name

        mlflow_run = get_mlflow_run_by_tango_run(
            self.mlflow_client,
            self.experiment_name,
            tango_run=run,
            additional_filter_string=" AND ".join(
                (
                    "attributes.status != 'FINISHED'",
                    "attributes.status != 'FAILED'",
                    "attributes.status != 'KILLED'",
                )
            ),
        )
        if mlflow_run is not None:
            status = self._run_status.get(run_name, "FINISHED")
            logger.info("Terminating run %s with status %s", run.name if isinstance(run, Run) else run, status)
            self.mlflow_client.set_terminated(mlflow_run.info.run_id, status=status)

        if run_name in self._run_status:
            self._run_status.pop(run_name)

    def registered_runs(self) -> Dict[str, Run]:
        return {
            mlflow_run.data.tags[MLFLOW_RUN_NAME]: self._get_tango_run_by_mlflow_run(mlflow_run)
            for mlflow_run in get_mlflow_runs(
                self.mlflow_client,
                self.experiment_name,
                run_kind=RunKind.TANGO_RUN,
            )
        }

    def registered_run(self, name: str) -> Run:
        mlflow_run = get_mlflow_run_by_tango_run(self.mlflow_client, self.experiment_name, name)
        if mlflow_run is None:
            raise KeyError(f"Run '{name}' not found in workspace")
        return self._get_tango_run_by_mlflow_run(mlflow_run)

    def _get_tango_run_by_mlflow_run(self, mlflow_run: MLFlowRun) -> Run:
        step_name_to_info: Dict[str, StepInfo] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run.info.run_id,
                    artifact_path="",
                    dst_path=temp_dir,
                )
            )
            steps_json_path = artifact_path / "steps.json"
            with steps_json_path.open("r") as jsonfile:
                for step_name, step_info_dict in json.load(jsonfile)["steps"].items():
                    step_info = StepInfo.from_json_dict(step_info_dict)
                    if step_info.cacheable:
                        updated_step_info = self._get_updated_step_info(step_info.unique_id, step_name=step_name)
                        if updated_step_info is not None:
                            step_info = updated_step_info
                    step_name_to_info[step_name] = step_info

        run_name = mlflow_run.data.tags[MLFLOW_RUN_NAME]
        return Run(
            name=run_name,
            steps=step_name_to_info,
            start_date=datetime.fromtimestamp(mlflow_run.info.start_time / 1000, tz=pytz.utc),
        )

    def _get_updated_step_info(self, step_id: str, step_name: Optional[str] = None) -> Optional[StepInfo]:
        def load_step_info(mlflow_run: MLFlowRun) -> StepInfo:
            with tempfile.TemporaryDirectory() as temp_dir:
                path = mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run.info.run_id,
                    artifact_path="step_info.json",
                    dst_path=temp_dir,
                )
                with open(path, "r") as jsonfile:
                    return StepInfo.from_json_dict(json.load(jsonfile))

        for mlflow_run in get_mlflow_runs(
            self.mlflow_client,
            self.experiment_name,
            run_kind=RunKind.STEP,
            tango_step=step_id,
            additional_filter_string=None if step_name is None else f"tags.step_name = '{step_name}'",
        ):
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
                path = mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run.info.run_id,
                    artifact_path="steps.json",
                    dst_path=temp_dir,
                )
                with open(path, "r") as jsonfile:
                    step_info_dict = json.load(jsonfile)
                    assert isinstance(step_info_dict, dict)
                    return step_info_dict

        for mlflow_run in get_mlflow_runs(
            self.mlflow_client,
            self.experiment_name,
            run_kind=RunKind.TANGO_RUN,
            tango_step=step_id,
        ):
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
                    mlflow_run = get_mlflow_run_by_tango_run(self.mlflow_client, self.experiment_name, name)
                    if mlflow_run is None:
                        raise RuntimeError(f"Run '{name}' not found in workspace")
                    self.mlflow_client.log_artifact(mlflow_run.info.run_id, str(log_path))

        return capture_logs()

    def _to_params(self) -> Dict[str, Any]:
        return {
            "type": "mlflow",
            "experiment_name": self.experiment_name,
            "tags": self._mlflow_tags,
            "tracking_uri": self.mlflow_client.tracking_uri,
        }
