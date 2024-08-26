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
from typing import List, Dict

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
def get_result(steps: List[Dict], **kwargs):
    result = None
    for step in steps:
        Runner = DataOp.get_runner(step)
        runner = Runner()
        result = runner.run(result, **kwargs)
    return result


@dataclass
class DataProvider(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None

    # url can be from remote (http) or localfile
    url: str = None
    archive_options: Archive = None

    transformation_ops: List[Dict] = field(default_factory=list)

    @property
    def retrieval_ops(self):
        assert self.url, "url must be defined"

        if not self.archive_options:
            return [RemoteLocalDataOp.from_url(self.url)]
        if self.archive_options["format"] == "tar":
            return [TarDataOp.from_url(self.url, self.archive_options["file"])]
        if self.archive_options["format"] == "zip":
            return [ZipDataOp.from_url(self.url, self.archive_options["file"])]
        raise NotImplementedError

    def get_data(self, **kwargs):
        return get_result(
            [
                *self.retrieval_ops,
                *self.transformation_ops,
            ],
            **kwargs
        )
