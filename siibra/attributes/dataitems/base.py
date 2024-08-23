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

from dataclasses import dataclass, field
from pathlib import Path
import requests
from typing import List

try:
    from typing import TypedDict
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict

from ...attributes import Attribute
from ...cache import fn_call_cache
from ...dataops.base import DataOp
from ...dataops.file_fetcher import (
    ZipRepository,
    TarRepository,
    LocalDirectoryRepository,
)


class Archive(TypedDict):
    file: str = None
    format: str = None


@fn_call_cache
def get_bytesio_from_url(url: str, archive_options: Archive = None) -> bytes:
    if Path(url).is_file():
        pth = Path(url)
        return LocalDirectoryRepository(pth.parent).get(pth.name)
    # TODO: stream bytesio instead
    if not archive_options:
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.content

    if archive_options["format"] == "zip":
        ArchiveRepoType = ZipRepository
    elif archive_options["format"] == "tar":
        ArchiveRepoType = TarRepository
    else:
        raise NotImplementedError(
            f"{archive_options['format']} is not a supported archive format yet."
        )

    filename = archive_options["file"]
    assert filename, "Data attribute 'file' field not populated!"
    repo = ArchiveRepoType(url)
    return repo.get(filename)


# TODO should be renamed DataProvider
@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None

    # url can be from remote (http) or localfile
    url: str = None
    archive_options: Archive = None

    transformation_ops: List = field(default_factory=list)

    @property
    def retrieval_ops(self):
        assert self.url, "url must be defined"

        if not self.archive_options:
            return [{"type": "src/file", "url": self.url}]

        if self.archive_options["format"] == "tar":
            return [
                {
                    "type": "src/remotetar",
                    "tar": self.url,
                    "filename": self.archive_options["file"],
                }
            ]
        if self.archive_options["format"] == "zip":
            return [
                {
                    "type": "src/remotezip",
                    "tar": self.url,
                    "filename": self.archive_options["file"],
                }
            ]
        raise NotImplementedError

    # TODO cache this step
    def get_data(self, **kwargs):
        result = None
        for step in [
            *self.retrieval_ops,
            *self.transformation_ops,
        ]:
            runner = DataOp.get_runner(step)
            result = runner.run(result, **kwargs)
        return result
