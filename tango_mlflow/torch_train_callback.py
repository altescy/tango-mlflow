from contextlib import suppress
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import Metric
from mlflow.entities import Run as MlflowRun
from mlflow.tracking.context import registry as context_registry
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from tango.common.exceptions import IntegrationMissingError

from tango_mlflow.util import RunKind, flatten_dict, get_mlflow_run_by_tango_step, get_timestamp
from tango_mlflow.workspace import MLFlowWorkspace

with suppress(ModuleNotFoundError, IntegrationMissingError):
    import torch
    from tango.integrations.torch.train_callback import TrainCallback
    from tango.integrations.torch.util import peak_gpu_memory

    @TrainCallback.register("mlflow::log")
    class MLFlowTrainCallback(TrainCallback):
        def __init__(
            self,
            *args: Any,
            experiment_name: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None,
            tracking_uri: Optional[str] = None,
            mlflow_config: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            if torch is None:
                raise IntegrationMissingError("torch")

            super().__init__(*args, **kwargs)

            if isinstance(self.workspace, MLFlowWorkspace):
                experiment_name = experiment_name or self.workspace.experiment_name
                tracking_uri = tracking_uri or self.workspace.mlflow_tracking_uri

            self.experiment_name = experiment_name
            self.tags = tags or {}
            self.tracking_uri = tracking_uri

            self.mlflow_config = self.train_config.as_dict()
            self.mlflow_config.pop("worker_id")
            if mlflow_config is not None:
                self.mlflow_config.update(mlflow_config)

            self.mlflow_run: Optional[MlflowRun] = None

        @property
        def mlflow_client(self) -> mlflow.tracking.MlflowClient:
            if isinstance(self.workspace, MLFlowWorkspace):
                return self.workspace.mlflow_client
            return mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

        def ensure_mlflow_run(self) -> MlflowRun:
            if self.mlflow_run is None:
                raise RuntimeError("MLFlow run not initialized")
            return self.mlflow_run

        def state_dict(self) -> Dict[str, Any]:
            return {}

        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            pass

        def pre_train_loop(self) -> None:
            if isinstance(self.workspace, MLFlowWorkspace):
                # Use existing MLFlow run created by the MLFlowWorkspace
                self.mlflow_run = get_mlflow_run_by_tango_step(
                    self.mlflow_client,
                    experiment=self.experiment_name,
                    tango_step=self.step_id,
                    additional_filter_string="attributes.status = 'RUNNING'",
                )
                if self.mlflow_run is None:
                    raise RuntimeError(f"Could not find a running MLFlow run for step {self.step_id}")
            else:
                # Create a new MLFlow run and log the config
                self.mlflow_run = self.mlflow_client.create_run(
                    experiment_id=mlflow.get_experiment_by_name(self.experiment_name).experiment_id,
                    tags=context_registry.resolve_tags(
                        {
                            "job_type": RunKind.TRAIN_METRICS,
                            "step_name": self.step_name,
                            "step_id": self.step_id,
                            MLFLOW_RUN_NAME: self.step_name,
                        }.update(self.tags)
                    ),
                )
                for key, value in flatten_dict(self.mlflow_config).items():
                    self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            peak_gpu_mbs = peak_gpu_memory()
            if self.is_local_main_process:
                timestamp = self.mlflow_run.info.start_time
                metrics = [
                    Metric(key=f"sys.worker{rank}_peak_gpu_mem", value=mbs, timestamp=timestamp, step=0)
                    for rank, mbs in peak_gpu_mbs.items()
                ]
                metrics.append(Metric(key="epoch", value=0, timestamp=timestamp, step=0))
                self.mlflow_client.log_batch(self.mlflow_run.info.run_id, metrics=metrics)

        def post_train_loop(self, step: int, epoch: int) -> None:
            if isinstance(self.workspace, MLFlowWorkspace):
                # We don't need to do anything here, as the MLFlow run will be closed by the MLFlowWorkspace
                return
            if self.is_local_main_process:
                mlflow_run = self.ensure_mlflow_run()
                self.mlflow_client.set_terminated(mlflow_run.info.run_id)

        def log_batch(
            self,
            step: int,
            epoch: int,
            batch_loss: float,
            batch_outputs: List[Dict[str, Any]],
        ) -> None:
            step = step + 1
            timestamp = get_timestamp()
            peak_gpu_mbs = peak_gpu_memory()
            if self.is_local_main_process:
                mlflow_run = self.ensure_mlflow_run()
                metrics = [
                    Metric(
                        key="train.loss",
                        value=batch_loss,
                        timestamp=timestamp,
                        step=step,
                    ),
                    Metric(
                        key="train.lr",
                        value=self.training_engine.optimizer.param_groups[0]["lr"],
                        timestamp=timestamp,
                        step=step,
                    ),
                    Metric(
                        key="epoch",
                        value=epoch,
                        timestamp=0,
                        step=step,
                    ),
                ] + [
                    Metric(key=f"sys.worker{rank}_peak_gpu_mem", value=mbs, timestamp=timestamp, step=step)
                    for rank, mbs in peak_gpu_mbs.items()
                ]
                self.mlflow_client.log_batch(mlflow_run.info.run_id, metrics=metrics)

        def post_val_loop(
            self,
            step: int,
            epoch: int,
            val_metric: float,
            best_val_metric: float,
        ) -> None:
            step = step + 1
            timestamp = get_timestamp()
            if self.is_local_main_process:
                mlflow_run = self.ensure_mlflow_run()
                metrics = [
                    Metric(
                        key="val.loss",
                        value=val_metric,
                        timestamp=timestamp,
                        step=step,
                    ),
                    Metric(
                        key=f"val.best_{self.train_config.val_metric_name}",
                        value=best_val_metric,
                        timestamp=timestamp,
                        step=step,
                    ),
                    Metric(
                        key="epoch",
                        value=epoch,
                        timestamp=timestamp,
                        step=step,
                    ),
                ]
                self.mlflow_client.log_batch(mlflow_run.info.run_id, metrics=metrics)
