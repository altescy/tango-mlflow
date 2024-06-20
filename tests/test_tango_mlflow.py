import tango_mlflow


def test_version() -> None:
    assert tango_mlflow.__version__ == "2.0.0"
