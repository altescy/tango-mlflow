import dataclasses
from typing import Any, Dict, Iterator, List

import pandas
from tango.common.testing import TangoTestCase

from tango_mlflow.format import CsvFormat, DaciteJsonFormat, TableFormat


@dataclasses.dataclass
class Data:
    a: str
    b: int


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

    def test_mapping_table_format(self) -> None:
        data = {
            "columns": ["a", "b"],
            "data": [["x", 1], ["y", 2]],
        }
        format = TableFormat[Dict[str, Any]]()
        format.write(data, self.TEST_DIR)
        assert format.read(self.TEST_DIR) == data

    def test_dataclass_table_format(self) -> None:
        data = [
            Data("x", 1),
            Data("y", 2),
        ]
        format = TableFormat[List[Data]]()
        format.write(data, self.TEST_DIR)
        assert format.read(self.TEST_DIR) == data

    def test_pandas_table_format(self) -> None:
        data = pandas.DataFrame(
            {
                "a": ["x", "y"],
                "b": [1, 2],
            }
        )
        format = TableFormat[pandas.DataFrame]()
        format.write(data, self.TEST_DIR)
        assert format.read(self.TEST_DIR).equals(data)

    def test_dataclass_dacite_json_format(self) -> None:
        data = [
            Data("x", 1),
            Data("y", 2),
        ]
        l1 = iter(data)
        format = DaciteJsonFormat[Iterator[Data]]()
        format.write(l1, self.TEST_DIR)
        l2 = format.read(self.TEST_DIR)
        assert list(l2) == data
