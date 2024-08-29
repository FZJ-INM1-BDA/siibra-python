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
from typing import List, Dict, Any

try:
    from typing import TypedDict
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict

from ...attributes import Attribute
from ...cache import fn_call_cache
from ...dataops.base import DataOp
from ...dataops.file_fetcher import (
    TarDataOp,
    ZipDataOp,
    RemoteLocalDataOp,
)


class Archive(TypedDict):
    file: str = None
    format: str = None


@fn_call_cache
def get_result(steps: List[Dict]):
    result: Any = None
    for step in steps:
        Runner = DataOp.get_runner(step)
        runner = Runner
        result = runner.run(result, **step)
    return result


@dataclass
class DataProvider(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None  # TODO: remove
    url: str = None  # url can be from remote (http) or localfile
    format: str = None
    archive_options: Archive = None

    retrieval_ops: List[Dict] = field(default_factory=list)
    transformation_ops: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if "neuroglancer" in self.format:
            return

        if len(self.retrieval_ops) > 0:
            return

        assert self.url, "url must be defined"

        if self.archive_options is None:
            self.retrieval_ops.append(RemoteLocalDataOp.from_url(self.url))
            return

        if self.archive_options["format"] == "tar":
            self.retrieval_ops.append(
                TarDataOp.from_url(self.url, self.archive_options["file"])
            )
            return
        if self.archive_options["format"] == "zip":
            self.retrieval_ops.append(
                ZipDataOp.from_url(self.url, self.archive_options["file"])
            )
            return

        raise RuntimeError(f"Cannot understand {self.archive_options['format']}")

    def get_data(self):
        return get_result(
            [
                *self.retrieval_ops,
                *self.transformation_ops,
            ]
        )
