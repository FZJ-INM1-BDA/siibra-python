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
        self.reader = PartialReader(path)
        self.reader.open()

    @property
    def unpacked_dir(self):
        dirname = CACHE.build_filename(self.path, ".unpacked")
        dirpath = Path(dirname)
        return dirpath

    def close(self):
        self.reader.close()

    def ls(self):
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
        if self._warmed_up:
            wanted_filepath = self.unpacked_dir / filepath
            if wanted_filepath.is_file():
                return wanted_filepath.read_bytes()
            raise FileNotFoundError
        fh = ZipFile(self.reader, "r")
        return fh.read(filepath)

    def search_files(self, prefix: str) -> Iterable[str]:
        yield from [filename for filename in self.ls() if filename.startswith(prefix)]
