# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Union
import tarfile
from pathlib import Path
import os

from .base import ArchivalRepository
from .io import PartialReader
from ...cache import CACHE
from ...commons.logger import logger


class TarRepository(ArchivalRepository):
    _warmed_up = False

    def __init__(
        self, path: str, *args, relative_path="", gzip=False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.path = path
        self.reader: PartialReader = None
        self.gzip = gzip
        self.relative_path = relative_path

    @property
    def unpacked_dir(self):
        dirname = CACHE.build_filename(self.path, ".unpacked")
        dirpath = Path(dirname)
        return dirpath

    def open(self):
        if self.reader is None:
            self.reader = PartialReader(self.path)
            self.reader.open()

            # random access on gzip does not work. More efficient to extract all
            if self.gzip:
                self.warmup()

    def close(self):
        if self.reader:
            self.reader.close()

    def ls(self):
        self.open()
        if self._warmed_up:
            assert self.unpacked_dir.is_dir()
            yield from [
                str(
                    Path(f"{dirpath}/{filename}")
                    .relative_to(self.unpacked_dir)
                    .relative_to(self.relative_path or "")
                )
                for dirpath, dirnames, filenames in os.walk(self.unpacked_dir)
                for filename in filenames
            ]
            return

        if self.gzip:
            logger.warning("tararchive is gzipped. Random access can be quite slow.")
        fh = tarfile.open(fileobj=self.reader, mode=("r:gz" if self.gzip else "r"))
        for mem in fh.getmembers():
            yield mem.name

    def warmup(self, *args, **kwargs):
        self.open()
        if self._warmed_up:
            return
        self.reader.warmup()
        fh = tarfile.open(fileobj=self.reader, mode=("r:gz" if self.gzip else "r"))
        assert (
            not self.unpacked_dir.is_file()
        ), f"{str(self.unpacked_dir)} is a file. Abort."
        if not self.unpacked_dir.is_dir():
            self.unpacked_dir.mkdir(parents=True)
        fh.extractall(self.unpacked_dir)
        self._warmed_up = True

    def get(self, filename: str) -> bytes:
        self.open()
        if self.gzip:
            self.warmup()

        filename = Path(self.relative_path) / filename

        if self._warmed_up:
            filepath = self.unpacked_dir / filename
            if filepath.is_file():
                return filepath.read_bytes()
            raise FileNotFoundError(f"{filename} not found.")

        if self.gzip:
            logger.warning("tararchive is gzipped. Random access can be quite slow.")

        fh = tarfile.open(fileobj=self.reader, mode=("r:gz" if self.gzip else "r"))
        try:
            extracted = fh.extractfile(filename)
            return extracted.read()
        except KeyError as e:
            raise FileNotFoundError(f"{filename} not found.") from e

    def search_files(self, prefix: Union[str, None] = None) -> Iterable[str]:
        self.open()
        for filename in self.ls():
            if prefix is None:
                yield filename
                continue

            if filename.startswith(prefix):
                yield filename
                continue