from typing import Iterable
import tarfile
from pathlib import Path
import os

from .base import ArchivalRepository
from .io import PartialReader
from ...cache import CACHE


class TarRepository(ArchivalRepository):
    def __init__(self, path: str, *args, gzip=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.path = path
        self.reader = PartialReader(path)

        self.reader.open()
        self.gzip = gzip

        # TODO add on the fly parsing later
        self.warmup()

    @property
    def unpacked_dir(self):
        dirname = CACHE.build_filename(self.path, ".unpacked")
        dirpath = Path(dirname)
        return dirpath

    def close(self):
        self.reader.close()

    def ls(self):
        assert self.unpacked_dir.is_dir()
        return [
            f"{dirpath}/{filename}"
            for dirpath, dirnames, filenames in os.walk(self.unpacked_dir)
            for filename in filenames
        ]

    def warmup(self, *args, **kwargs):
        self.reader.warmup()
        if self.unpacked_dir.is_dir():
            return
        fh = tarfile.open(fileobj=self.reader, mode="r:gz")
        dirname = CACHE.build_filename(self.path, ".unpacked")
        dirpath = Path(dirname)
        assert not self.unpacked_dir.is_file(), f"{str(dirpath)} is a file. Abort."
        if not self.unpacked_dir.is_dir():
            dirpath.mkdir(parents=True)
            fh.extractall(dirpath)

    def get(self, filename: str) -> bytes:

        # TODO implement on the fly fetching
        self.warmup()

        filepath = self.unpacked_dir / filename
        try:
            assert filepath.exists()
            assert filepath.is_file()
        except AssertionError as e:
            raise FileNotFoundError from e
        return filepath.read_bytes()

    def search_files(self, prefix: str) -> Iterable[str]:
        return super().search_files(prefix)
