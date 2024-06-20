from tango.common.testing import TangoTestCase

from tango_mlflow.format import CsvFormat


class TestFormat(TangoTestCase):
    def test_list_csv_format(self) -> None:
        artifact = [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]
        print("artifact type:", type(artifact))
        format = CsvFormat()
        format.write(artifact, self.TEST_DIR)
        assert format.read(self.TEST_DIR) == artifact

    def test_iterator_csv_format(self) -> None:
        data = [
            {"a": "1", "b": "2"},
            {"a": "3", "b": "4"},
        ]
        l1 = iter(data)
        format = CsvFormat()
        format.write(l1, self.TEST_DIR)
        l2 = format.read(self.TEST_DIR)
        assert list(l2) == data
