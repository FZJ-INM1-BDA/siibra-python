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

from pathlib import Path
from zipfile import ZipFile
from typing import Iterable
import os

from .base import ArchivalRepository
from .io import PartialReader
from ...cache import CACHE


class ZipRepository(ArchivalRepository):
    _warmed_up = False

    def __init__(self, path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.path = path
        self.reader: PartialReader = None

    @property
    def unpacked_dir(self):
        dirname = CACHE.build_filename(self.path, ".unpacked")
        dirpath = Path(dirname)
        return dirpath

    def open(self):
        if self.reader is None:
            self.reader = PartialReader(self.path)
            self.reader.open()

    def close(self):
        if self.reader:
            self.reader.close()

    def ls(self):
        self.open()
        if self._warmed_up:
            assert self.unpacked_dir.is_dir()
            yield from [
                f"{dirpath}/{filename}"
                for dirpath, dirnames, filenames in os.walk(self.unpacked_dir)
                for filename in filenames
            ]
            return

        zipfile = ZipFile(self.reader)
        for info in zipfile.filelist:
            yield info.filename

    def warmup(self, *args, **kwargs):
        self.open()
        if self._warmed_up:
            return
        self.reader.warmup()
        fh = ZipFile(self.reader, "r")
        assert (
            not self.unpacked_dir.is_file()
        ), f"{str(self.unpacked_dir)} is a file. Abort."
        if not self.unpacked_dir.is_dir():
            self.unpacked_dir.mkdir(parents=True)
        fh.extractall(self.unpacked_dir)
        self._warmed_up = True

    def get(self, filepath: str) -> bytes:
        self.open()
        if self._warmed_up:
            wanted_filepath = self.unpacked_dir / filepath
            if wanted_filepath.is_file():
                return wanted_filepath.read_bytes()
            raise FileNotFoundError
        fh = ZipFile(self.reader, "r")
        return fh.read(filepath)

    def search_files(self, prefix: str = None) -> Iterable[str]:
        self.open()
        for filename in self.ls():
            if prefix is None:
                yield filename
                continue
            if filename.startswith(prefix):
                yield filename
                continue
