from pathlib import Path
from typing import Iterable
import os

from .base import Repository


class LocalDirectoryRepository(Repository):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        assert Path(
            path
        ).is_dir(), f"LocalRepository needs {path=} to be a directory, but is not."

    def search_files(self, prefix: str = None) -> Iterable[str]:
        root_path = Path(self.path)
        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                filepath = root_path / dirpath / filename
                relative_path = filepath.relative_to(root_path)
                if prefix is None:
                    yield str(filepath)
                    continue
                if str(relative_path).startswith(prefix):
                    yield str(filepath)
                    continue

    def get(self, filepath: str) -> bytes:
        return (Path(self.path) / filepath).read_bytes()
