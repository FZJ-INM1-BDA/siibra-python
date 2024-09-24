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
import requests

from .base import Repository
from ..base import DataOp
from ...commons.conf import SiibraConf
from ...cache import CACHE


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


class RemoteLocalDataOp(DataOp):
    input: None
    output: bytes
    type = "read/remote-local"
    desc = "Read local/remote to bytes"
    KEEP_LOCAL_CACHE_THRESHOLD = 0

    def describe(self, *, filename: str, **kwargs):
        desc = ""
        if filename.startswith("https"):
            desc += f"Reading remote file at {filename} "
            cache_filename = CACHE.build_filename(filename)
            if Path(cache_filename).is_file():
                desc += f"by reading a cached version at {cache_filename}"
                return desc
            if SiibraConf.KEEP_LOCAL_CACHE > self.KEEP_LOCAL_CACHE_THRESHOLD:
                desc += f"as KEEP_LOCAL_CACHE flag is set to {SiibraConf.KEEP_LOCAL_CACHE}, higher than the threshold {self.KEEP_LOCAL_CACHE_THRESHOLD}, a local version will be saved at {cache_filename}"
            return desc
        desc += f"Reading local file at {filename}"
        return desc

    def run(self, _, *, filename, **kwargs):
        assert isinstance(
            filename, str
        ), "remote local da73ta op only takes string as filename kwarg"
        if filename.startswith("https"):
            cache_filename = CACHE.build_filename(filename)
            if Path(cache_filename).is_file():
                with open(cache_filename, "rb") as fp:
                    return fp.read()

            resp = requests.get(filename)
            resp.raise_for_status()
            if SiibraConf.KEEP_LOCAL_CACHE > self.KEEP_LOCAL_CACHE_THRESHOLD:
                with open(cache_filename, "wb") as fp:
                    fp.write(resp.content)
            return resp.content
        with open(filename, "rb") as fp:
            return fp.read()

    @classmethod
    def generate_specs(cls, filename, **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "filename": filename}
