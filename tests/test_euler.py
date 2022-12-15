import cmath
from typing import Tuple, Union

import mlflow
import pytest
from tango import Step
from tango.common import util
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.testing import TangoTestCase
from tango.settings import TangoGlobalSettings

MLFLOW_EXPERIMENT_NAME = "tango-mlflow-testing"

ComplexOrTuple = Union[complex, Tuple[float, float]]


def make_complex(x: Union[int, float, ComplexOrTuple]) -> complex:
    if isinstance(x, (int, float)):
        return complex(x)
    if isinstance(x, complex):
        return x
    return complex(*x)


@Step.register("cadd")
class AdditionStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) + make_complex(b)


@Step.register("csub")
class SubtractionStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) - make_complex(b)


@Step.register("cexp")
class ExponentiateStep(Step):
    def run(self, x: ComplexOrTuple, base: ComplexOrTuple = cmath.e) -> complex:  # type: ignore
        return make_complex(base) ** make_complex(x)


@Step.register("cmul")
class MultiplyStep(Step):
    def run(self, a: ComplexOrTuple, b: ComplexOrTuple) -> complex:  # type: ignore
        return make_complex(a) * make_complex(b)


@Step.register("csin")
class SineStep(Step):
    def run(self, x: ComplexOrTuple) -> complex:  # type: ignore
        return cmath.sin(make_complex(x))


@Step.register("ccos")
class CosineStep(Step):
    def run(self, x: ComplexOrTuple) -> complex:  # type: ignore
        return cmath.cos(make_complex(x))


class TestEuler(TangoTestCase):
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore
        super().setup_method()  # type: ignore[no-untyped-call]
        initialize_logging()
        monkeypatch.setattr(util, "tango_cache_dir", lambda: self.TEST_DIR)
        mlflow.set_tracking_uri(f"file://{self.TEST_DIR.absolute() / 'mlruns'}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        self.config = {
            "steps": {
                "i_times_pi": {"a": [0, 1], "b": [3.1415926535000001, 0], "type": "cmul"},
                "plus_one": {"a": {"ref": "pow_e", "type": "ref"}, "b": [1, 0], "type": "cadd"},
                "pow_e": {"type": "cexp", "x": {"ref": "i_times_pi", "type": "ref"}},
                "print": {"input": {"ref": "plus_one", "type": "ref"}, "type": "print"},
            }
        }

    def teardown_method(self) -> None:
        super().teardown_method()  # type: ignore[no-untyped-call]
        teardown_logging()  # type: ignore[no-untyped-call]

    def test_experiment(self) -> None:
        self.run(
            config=self.config,
            settings=TangoGlobalSettings(
                workspace={"type": "mlflow", "experiment_name": MLFLOW_EXPERIMENT_NAME},
                include_package=["tango_mlflow"],
            ),
        )
