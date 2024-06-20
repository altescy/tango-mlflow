import csv
import itertools
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, TextIO

from tango.common import PathOrStr
from tango.format import Format


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
