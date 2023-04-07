import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlencode, urlparse

import mlflow
import petname
from mlflow.entities import Experiment as MLFlowExperiment
from mlflow.entities import Run as MLFlowRun
from mlflow.tracking import artifact_utils
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME, MLFLOW_RUN_NOTE
from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Run as TangoRun


class RunKind(Enum):
    STEP = "step"
    TANGO_RUN = "tango_run"
    OPTUNA_STUDY = "optuna_study"
    TRAIN_METRICS = "train_metrics"


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single level dictionary.
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            value = {str(i): v for i, v in enumerate(value)}
        if isinstance(value, dict):
            result.update({f"{key}.{k}": v for k, v in flatten_dict(value).items()})
        else:
            result[key] = value
    return result


def get_timestamp() -> int:
    """timestamp as an integer (milliseconds since the Unix epoch)."""
    return int(datetime.datetime.now().timestamp() * 1000)


def is_mlflow_using_local_artifact_storage(
    mlflow_run: Union[str, MLFlowRun],
) -> bool:
    mlflow_run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MLFlowRun) else mlflow_run
    return str(artifact_utils.get_artifact_uri(run_id=mlflow_run_id)).startswith("file")


def get_mlflow_local_artifact_storage_path(
    mlflow_run: Union[str, MLFlowRun],
) -> Optional[Path]:
    mlflow_run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MLFlowRun) else mlflow_run
    mlflow_artifact_uri = urlparse(artifact_utils.get_artifact_uri(run_id=mlflow_run_id))
    if mlflow_artifact_uri.scheme == "file":
        return Path(mlflow_artifact_uri.path)
    return None


def build_filter_string(
    run_kind: Optional[RunKind] = None,
    tango_step: Optional[Union[str, Step, StepInfo]] = None,
    tango_run: Optional[Union[str, TangoRun]] = None,
    additional_filter_string: Optional[str] = None,
) -> str:
    conditions: List[str] = []
    if run_kind is not None:
        conditions.append(f"tags.job_type = '{run_kind.value}'")
    if tango_step is not None:
        tango_step_id = tango_step if isinstance(tango_step, str) else tango_step.unique_id
        if run_kind == RunKind.TANGO_RUN:
            conditions.append(f"params._step_ids.\"{tango_step_id}\" = 'True'")
        else:
            conditions.append(f"tags.step_id = '{tango_step_id}'")
    if tango_run is not None:
        tango_run_name = tango_run.name if isinstance(tango_run, TangoRun) else tango_run
        conditions.append(f"tags.{MLFLOW_RUN_NAME} = '{tango_run_name}'")
    if additional_filter_string is not None:
        conditions.append(additional_filter_string)
    return " AND ".join(conditions)


def get_mlflow_run_by_tango_step(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    tango_step: Union[str, Step, StepInfo],
    additional_filter_string: Optional[str] = None,
) -> Optional[MLFlowRun]:
    if isinstance(experiment, str):
        experiment = mlflow_client.get_experiment_by_name(experiment)

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=build_filter_string(
            run_kind=RunKind.STEP,
            tango_step=tango_step,
            additional_filter_string=additional_filter_string,
        ),
    )

    if not runs:
        return None

    return runs[0]


def get_mlflow_run_by_tango_run(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    tango_run: Union[str, TangoRun],
    additional_filter_string: Optional[str] = None,
) -> Optional[MLFlowRun]:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)

    tango_run_name = tango_run.name if isinstance(tango_run, TangoRun) else tango_run

    mlflow_runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=build_filter_string(
            run_kind=RunKind.TANGO_RUN,
            tango_run=tango_run_name,
            additional_filter_string=additional_filter_string,
        ),
    )

    if not mlflow_runs:
        return None

    if len(mlflow_runs) > 1:
        raise ValueError(f"Found more than one run name '{tango_run_name}' in MLflow experiment")

    return mlflow_runs[0]


def get_mlflow_runs(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    run_kind: Optional[RunKind] = None,
    tang_run: Optional[Union[str, TangoRun]] = None,
    tango_step: Optional[Union[str, Step, StepInfo]] = None,
    additional_filter_string: Optional[str] = None,
) -> List[MLFlowRun]:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)

    mlflow_runs: List[MLFlowRun] = []
    page_token: Optional[str] = None
    while True:
        matching_runs = mlflow_client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=build_filter_string(
                run_kind=run_kind,
                tango_step=tango_step,
                tango_run=tang_run,
                additional_filter_string=additional_filter_string,
            ),
            page_token=page_token,
        )
        mlflow_runs.extend(matching_runs)
        if matching_runs.token is None:
            break
        page_token = matching_runs.token

    return mlflow_runs


def generate_unique_run_name(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
) -> str:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)
    name: Optional[str] = None
    while name is None or get_mlflow_run_by_tango_run(mlflow_client, experiment, name) is not None:
        name = petname.generate()
    return name


def add_mlflow_run_of_tango_run(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    steps: Set[Step],
    run_name: Optional[str] = None,
    mlflow_tags: Optional[Dict[str, str]] = None,
) -> MLFlowRun:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)

    # generate a unique run name
    if run_name is None:
        run_name = generate_unique_run_name(mlflow_client, experiment)

    # build description of the run
    description = "# Tango run"
    cacheable_steps = {step for step in steps if step.cache_results}
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
            search_query_key = "searchInput" if mlflow.__version__ < "2.0.0" else "searchFilter"
            query = {search_query_key: f"tags.step_id = '{step.unique_id}'"}
            url = f"/#/experiments/{experiment.experiment_id}/s?{urlencode(query)}"
            description += f"\n  - [{step.unique_id}]({url})\n"

    mlflow_run = mlflow_client.create_run(
        experiment_id=experiment.experiment_id,
        tags=context_registry.resolve_tags(
            {
                "job_type": RunKind.TANGO_RUN.value,
                MLFLOW_RUN_NAME: run_name,
                MLFLOW_RUN_NOTE: description,
                **(mlflow_tags or {}),
            }
        ),
        run_name=run_name,
    )

    step_ids: Dict[str, bool] = {}
    step_name_to_info: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        step_info = StepInfo.new_from_step(step)
        step_name_to_info[step.name] = {k: v for k, v in step_info.to_json_dict().items() if v is not None}

        # log step config
        step_config = flatten_dict({step.name: step.config})
        for key, value in step_config.items():
            mlflow_client.log_param(mlflow_run.info.run_id, key, value)

        # log dependent steps
        step_ids[step.unique_id] = True
        mlflow_client.log_param(mlflow_run.info.run_id, f"_step_ids.{step.unique_id}", True)

    # log entire step info
    mlflow_client.log_dict(
        run_id=mlflow_run.info.run_id,
        dictionary={"steps": step_name_to_info, "_step_ids": step_ids},
        artifact_file="steps.json",
    )

    return mlflow_run


def add_mlflow_run_of_tango_step(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    tango_run: Union[str, TangoRun],
    step_info: StepInfo,
) -> MLFlowRun:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)

    parent_mlflow_run = get_mlflow_run_by_tango_run(mlflow_client, experiment, tango_run)
    if parent_mlflow_run is None:
        raise ValueError(f"Could not find MLflow run for Tango run '{tango_run}'")

    description = "\n".join(
        (
            f"### Tango step ***{step_info.step_name}***",
            f"- type: {step_info.step_class_name}",
            f"- ID: {step_info.unique_id}",
        )
    )

    mlflow_run = mlflow_client.create_run(
        experiment_id=experiment.experiment_id,
        start_time=None if step_info.start_time is None else round(1000 * step_info.start_time.timestamp()),
        tags=context_registry.resolve_tags(
            {
                "job_type": RunKind.STEP.value,
                "step_name": step_info.step_name,
                "step_id": step_info.unique_id,
                MLFLOW_RUN_NAME: step_info.step_name,
                MLFLOW_PARENT_RUN_ID: parent_mlflow_run.info.run_id,
                MLFLOW_RUN_NOTE: description,
            }
        ),
        run_name=step_info.step_name,
    )

    for key, value in flatten_dict(step_info.config or {}).items():
        mlflow_client.log_param(mlflow_run.info.run_id, key, value)
    mlflow_client.log_dict(
        run_id=mlflow_run.info.run_id,
        dictionary=step_info.to_json_dict(),
        artifact_file="step_info.json",
    )

    return mlflow_run


def terminate_mlflow_run_of_tango_step(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    status: str,
    step_info: StepInfo,
) -> None:
    if isinstance(experiment, str):
        experiment = mlflow.get_experiment_by_name(experiment)

    mlflow_run = get_mlflow_run_by_tango_step(mlflow_client, experiment, step_info)
    if mlflow_run is None:
        raise ValueError(f"Could not find MLFlow run for step {step_info.unique_id}")

    mlflow_client.log_dict(
        run_id=mlflow_run.info.run_id,
        dictionary=step_info.to_json_dict(),
        artifact_file="step_info.json",
    )
    mlflow_client.set_terminated(
        run_id=mlflow_run.info.run_id,
        status=status,
        end_time=None if step_info.end_time is None else round(1000 * step_info.end_time.timestamp()),
    )


def is_all_child_run_finished(
    mlflow_client: MlflowClient,
    experiment: Union[str, MLFlowExperiment],
    mlflow_run: Union[str, MLFlowRun],
) -> bool:
    run_id = mlflow_run.info.run_id if isinstance(mlflow_run, MLFlowRun) else mlflow_run

    for child_run in get_mlflow_runs(
        mlflow_client,
        experiment,
        additional_filter_string=f"tags.{MLFLOW_PARENT_RUN_ID} = '{run_id}'",
    ):
        if child_run.info.status != "FINISHED":
            return False
    return True
