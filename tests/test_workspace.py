import pickle
import sys

import mlflow
import pytest
from tango.common import util
from tango.common.testing import TangoTestCase
from tango.workspace import Workspace

from tango_mlflow.workspace import MLFlowWorkspace

MLFLOW_EXPERIMENT_NAME = "tango-mlflow-testing"


class TestMlflowWorkspace(TangoTestCase):
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch) -> None:  # type: ignore
        super().setup_method()  # type: ignore[no-untyped-call]
        monkeypatch.setattr(util, "tango_cache_dir", lambda: self.TEST_DIR)
        mlflow.set_tracking_uri(f"file://{self.TEST_DIR.absolute() / 'mlruns'}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    @pytest.mark.parametrize(
        "protocol",
        [pytest.param(protocol, id=f"protocol={protocol}") for protocol in range(4)]
        + [
            pytest.param(
                5,
                id="protocol=5",
                marks=pytest.mark.skipif(sys.version_info < (3, 8), reason="Protocol 5 requires Python 3.8 or newer"),
            ),
        ],
    )
    def test_pickle_workspace(self, protocol: int) -> None:
        workspace = MLFlowWorkspace(experiment_name=MLFLOW_EXPERIMENT_NAME)
        unpickled_workspace = pickle.loads(pickle.dumps(workspace, protocol=protocol))
        assert unpickled_workspace.mlflow_client is not None
        assert unpickled_workspace.experiment_name == workspace.experiment_name
        assert unpickled_workspace.steps_dir == workspace.steps_dir

    def test_from_url(self) -> None:
        workspace = Workspace.from_url(f"mlflow://{MLFLOW_EXPERIMENT_NAME}")
        assert isinstance(workspace, MLFlowWorkspace)
        assert workspace.experiment_name == MLFLOW_EXPERIMENT_NAME
