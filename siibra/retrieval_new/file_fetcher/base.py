from abc import ABC, abstractmethod
from typing import Iterable


class Repository(ABC):

    @abstractmethod
    def search_files(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError

    @abstractmethod
    def get(self, filepath: str):
        raise NotImplementedError


class ArchivalRepository(Repository, ABC):
    """ArchivalRepository is the subclass used for Repository where file access may be bound by
    a variety of reasons (network IO, tar archive etc). As such, ArchivalRepository can implement
    `warmup` method, which is meant to remove the obstacle (e.g. download all files, extract the archive)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def warmup(self, *args, **kwargs):
        pass

    @abstractmethod
    def search_files(self, prefix: str) -> Iterable[str]:
        return super().search_files(prefix)
    
    @abstractmethod
    def get(self, filepath: str):
        return super().get(filepath)
