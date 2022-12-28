import typing
from typing import Any, Dict, Optional, Protocol, TypeVar

from mlflow.entities import Run as MLflowRun
from mlflow.tracking import MlflowClient
from tango.step import Step

T = TypeVar("T")


@typing.runtime_checkable
class MLflowSummaryStep(Protocol):
    MLFLOW_SUMMARY: bool


class MLflowLogger:
    def __init__(self, mlflow_run: MLflowRun):
        self._mlflow_run = mlflow_run

    @property
    def mlflow_run(self) -> MLflowRun:
        return self._mlflow_run

    @property
    def mlflow_client(self) -> MlflowClient:
        return MlflowClient()

    def log_metric(
        self,
        key: str,
        value: float,
        timestamp: Optional[int] = None,
        step: Optional[int] = None,
    ) -> None:
        self.mlflow_client.log_metric(
            run_id=self.mlflow_run.info.run_id,
            key=key,
            value=value,
            timestamp=timestamp,
            step=step,
        )

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value)


class MLflowStep(Step[T]):
    MLFLOW_SUMMARY: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mlflow_run: Optional[MLflowRun] = None
        self._mlflow_logger: Optional[MLflowLogger] = None

    def setup_mlflow(self, mlflow_run: MLflowRun) -> None:
        if self._mlflow_run is not None:
            raise RuntimeError("MLflow run already set")
        self._mlflow_run = mlflow_run
        self._mlflow_logger = MLflowLogger(self._mlflow_run)

    @property
    def mlflow_run(self) -> MLflowRun:
        if self._mlflow_run is None:
            raise RuntimeError("MLflow run not set")
        return self._mlflow_run

    @property
    def mlflow_logger(self) -> MLflowLogger:
        if self._mlflow_logger is None:
            raise RuntimeError("MLflow logger not set")
        return self._mlflow_logger
