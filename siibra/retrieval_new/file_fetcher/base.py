from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path
import os


class Repository(ABC):

    @abstractmethod
    def search_files(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str) -> bytes:
        raise NotImplementedError

    @staticmethod
    def from_url(path: str):
        from .zip_fetcher import ZipRepository
        from .tar_fetcher import TarRepository
        from .local_directory_fetcher import LocalDirectoryRepository

        expurl = os.path.abspath(os.path.expanduser(path))

        if expurl.endswith(".zip"):
            return ZipRepository(expurl)
        if expurl.endswith(".tar") or expurl.endswith(".tar.gz"):
            return TarRepository(expurl)
        if os.path.isdir(expurl):
            return LocalDirectoryRepository(expurl)
        raise TypeError(
            "Do not know how to create a repository " f"connector from url '{path}'."
        )


class ArchivalRepository(Repository, ABC):
    """ArchivalRepository is the subclass used for Repository where file access may be bound by
    a variety of reasons (network IO, tar archive etc). As such, ArchivalRepository can implement
    `warmup` method, which is meant to remove the obstacle (e.g. download all files, extract the archive)
    """

    @property
    def is_warm(self):
        return self.unpacked_dir is not None and Path(self.unpacked_dir).is_dir()

    @property
    @abstractmethod
    def unpacked_dir(self):
        """Subclass should implement this property. Checking this directory exists and is populated
        is one of the ways where cache can be preserved."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    @abstractmethod
    def search_files(self, prefix: str = None) -> Iterable[str]:
        return super().search_files(prefix)

    @abstractmethod
    def get(self, filepath: str):
        return super().get(filepath)
