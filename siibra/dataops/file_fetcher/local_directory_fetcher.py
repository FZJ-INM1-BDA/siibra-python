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
from typing import Iterable
import os

from .base import Repository


class LocalDirectoryRepository(Repository):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        assert Path(
            path
        ).is_dir(), (
            f"LocalRepository needs path={path!r} to be a directory, but is not."
        )

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
