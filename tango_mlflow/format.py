import csv
import dataclasses
import importlib
import itertools
import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    Tuple,
    TypeVar,
    cast,
    runtime_checkable,
)

import dacite
import pandas
from mlflow.entities import Run as MlflowRun
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_ARTIFACTS
from tango.common import PathOrStr
from tango.format import Format, JsonFormat, JsonFormatIterator


@runtime_checkable
class MlflowFormat(Protocol):
    def get_mlflow_artifact_path(self) -> str:
        ...

    def mlflow_callback(self, client: MlflowClient, run: MlflowRun) -> None:
        ...


class CsvFormat(Format[Iterable[Mapping[str, str]]]):
    class CsvIterator:
        def __init__(self, filename: PathOrStr) -> None:
            self._file: Optional[TextIO] = Path(filename).open()
            self._reader = csv.DictReader(self._file)

        def __iter__(self) -> "CsvFormat.CsvIterator":
            return self

        def __next__(self) -> Mapping[str, str]:
            if self._file is None:
                raise StopIteration
            try:
                return next(self._reader)
            except StopIteration:
                self._file.close()
                self._file = None
                raise

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _get_filename(self, dir: PathOrStr, is_iterator: bool) -> Path:
        if is_iterator:
            return Path(dir) / "data.iter.csv"
        return Path(dir) / "data.csv"

    def write(self, artifact: Iterable[Mapping[str, str]], dir: PathOrStr) -> None:
        is_iterator = hasattr(artifact, "__next__")
        artifact = iter(artifact)
        row = next(artifact)
        fieldnames = list(row.keys())
        artifact = itertools.chain([row], artifact)
        filename = self._get_filename(dir, is_iterator)
        with filename.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(artifact)

    def read(self, dir: PathOrStr) -> Iterable[Mapping[str, str]]:
        iterator_filename = self._get_filename(dir, is_iterator=True)
        iterator_exists = iterator_filename.exists()
        non_iterator_filename = self._get_filename(dir, is_iterator=False)
        non_iterator_exists = non_iterator_filename.exists()

        if iterator_exists and non_iterator_exists:
            self.logger.warning(
                "Both %s and %s exist. Ignoring %s.",
                iterator_filename,
                non_iterator_filename,
                iterator_filename,
            )
            iterator_exists = False

        if not iterator_exists and not non_iterator_exists:
            raise IOError("Attempting to read non-existing data from %s", dir)

        if iterator_exists:
            return CsvFormat.CsvIterator(iterator_filename)
        elif not iterator_exists and non_iterator_exists:
            with non_iterator_filename.open() as csv_file:
                return list(csv.DictReader(csv_file))
        else:
            raise RuntimeError("This should be impossible.")


T_TableFormattable = TypeVar(
    "T_TableFormattable",
    pandas.DataFrame,
    Sequence[Any],
    Mapping[str, Any],
)


class TableFormat(Format[T_TableFormattable], Generic[T_TableFormattable]):
    _FILENAME = "data.json"
    _METADATA_FIELD = "@@METADATA@@"

    def _get_filename(self, dir: PathOrStr) -> Path:
        return Path(dir) / self._FILENAME

    def _inspect_type(self, x: type) -> str:
        return f"{x.__module__}.{x.__name__}"

    def _get_type(self, x: str) -> Any:
        module_name, type_name = x.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, type_name)

    def _convert_sequence_to_dict(self, artifact: Sequence) -> Tuple[Dict[str, Any], Optional[str]]:
        if not artifact:
            return {"columns": [], "data": []}, None
        first = artifact[0]
        item_type = self._inspect_type(type(first))
        if isinstance(first, Mapping):
            return (
                cast(
                    Dict[str, Any],
                    pandas.DataFrame(artifact).to_dict(orient="split", index=False),
                ),
                item_type,
            )
        elif dataclasses.is_dataclass(first):
            assert all(type(row) is type(first) for row in artifact)
            return (
                cast(
                    Dict[str, Any],
                    pandas.DataFrame([dataclasses.asdict(row) for row in artifact]).to_dict(
                        orient="split", index=False
                    ),
                ),
                item_type,
            )
        else:
            raise ValueError(f"Cannot convert {type(first)} to dict")

    def write(self, artifact: T_TableFormattable, dir: PathOrStr) -> None:
        table: Dict[str, Any]
        metadata: Dict[str, Any] = {}
        if isinstance(artifact, pandas.DataFrame):
            table = artifact.to_dict(orient="split")
            metadata["type"] = "pandas"
        elif isinstance(artifact, Sequence):
            table, item_type = self._convert_sequence_to_dict(artifact)
            metadata["type"] = f"sequence:{item_type}"
        elif isinstance(artifact, Mapping):
            assert self._METADATA_FIELD not in artifact, artifact.keys()
            assert set(artifact.keys()) in ({"columns", "data"}, {"index", "columns", "data"}), artifact.keys()
            table = dict(artifact)
            metadata["type"] = "mapping"
        else:
            raise ValueError(f"Cannot convert {type(artifact)} to dict")

        assert set(table.keys()) in ({"columns", "data"}, {"index", "columns", "data"})

        table[self._METADATA_FIELD] = metadata

        filename = self._get_filename(dir)
        with filename.open("w") as f:
            json.dump(table, f, ensure_ascii=False)

    def read(self, dir: PathOrStr) -> T_TableFormattable:
        filename = self._get_filename(dir)
        with filename.open() as f:
            table = json.load(f)
        metadata = table.pop(self._METADATA_FIELD, {})
        if metadata["type"] == "pandas":
            return cast(T_TableFormattable, pandas.DataFrame(**table))
        elif metadata["type"] == "mapping":
            return cast(T_TableFormattable, table)
        elif metadata["type"].startswith("sequence:"):
            item_type = metadata["type"][9:]
            item_class = self._get_type(item_type)
            keys = table["columns"]
            return cast(
                T_TableFormattable,
                [dacite.from_dict(item_class, dict(zip(keys, row))) for row in table["data"]],
            )
        else:
            raise ValueError(f"Unknown table type {metadata['type']}")

    def get_mlflow_artifact_path(self) -> str:
        return self._FILENAME

    def mlflow_callback(self, client: MlflowClient, run: MlflowRun) -> None:
        client.set_tag(
            run.info.run_id,
            MLFLOW_LOGGED_ARTIFACTS,
            f'[{{"path": "{self.get_mlflow_artifact_path()}", "type": "table"}}]',
        )


T = TypeVar("T")


class DaciteJsonFormat(JsonFormat[T], Generic[T]):
    _TYPE_FIELD = "@@TYPE@@"

    class _Empty:
        ...

    def _inspect_type(self, x: type) -> str:
        return f"{x.__module__}.{x.__name__}"

    def _get_type(self, x: str) -> Any:
        module_name, type_name = x.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, type_name)

    def _convert_to_json(self, o: Any) -> Mapping[str, Any]:
        if isinstance(o, Mapping):
            return o
        if dataclasses.is_dataclass(o):
            data = dataclasses.asdict(o)
            assert self._TYPE_FIELD not in data
            data[self._TYPE_FIELD] = self._inspect_type(type(o))
            return data
        raise ValueError(f"Cannot convert {type(o)}")

    def _restore_from_json(self, o: Dict[str, Any]) -> Any:
        item_type = o.pop(self._TYPE_FIELD, None)
        if not item_type:
            return o
        item_class = self._get_type(item_type)
        return dacite.from_dict(item_class, o)

    def write(self, artifact: T, dir: PathOrStr) -> None:
        if hasattr(artifact, "__next__"):
            artifact = map(self._convert_to_json, artifact)  # type: ignore[call-overload]
        else:
            artifact = self._convert_to_json(artifact)  # type: ignore[assignment]
        super().write(artifact, dir)

    def read(self, dir: PathOrStr) -> T:
        artifact = super().read(dir)
        if isinstance(artifact, JsonFormatIterator):
            return cast(T, map(self._restore_from_json, artifact))
        assert isinstance(artifact, dict)
        return cast(T, self._restore_from_json(artifact))
