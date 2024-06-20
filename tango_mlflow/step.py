import typing
from typing import Any, Dict, Optional, Protocol, TypeVar

from mlflow.entities import Run as MlflowRun
from mlflow.tracking import MlflowClient
from tango.step import Step

T = TypeVar("T")


@typing.runtime_checkable
class MlflowSummaryStep(Protocol):
    MLFLOW_SUMMARY: bool


class MlflowLogger:
    def __init__(self, mlflow_run: MlflowRun):
        self._mlflow_run = mlflow_run

    @property
    def mlflow_run(self) -> MlflowRun:
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


class MlflowStep(Step[T]):
    MLFLOW_SUMMARY: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mlflow_run: Optional[MlflowRun] = None
        self._mlflow_logger: Optional[MlflowLogger] = None

    def setup_mlflow(self, mlflow_run: MlflowRun) -> None:
        if self._mlflow_run is not None:
            raise RuntimeError("Mlflow run already set")
        self._mlflow_run = mlflow_run
        self._mlflow_logger = MlflowLogger(self._mlflow_run)

    @property
    def mlflow_run(self) -> MlflowRun:
        if self._mlflow_run is None:
            raise RuntimeError("Mlflow run not set")
        return self._mlflow_run

    @property
    def mlflow_logger(self) -> MlflowLogger:
        if self._mlflow_logger is None:
            raise RuntimeError("Mlflow logger not set")
        return self._mlflow_logger
