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
from copy import deepcopy

try:
    from typing import TypedDict
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict

from ...attributes import Attribute
from ...cache import fn_call_cache
from ...operations.base import DataOp
from ...commons.logger import logger
from ...operations.file_fetcher import (
    TarDataOp,
    ZipDataOp,
    RemoteLocalDataOp,
)


class Archive(TypedDict):
    file: str = None
    format: str = None


def cache_validation_callback(metadata):
    args = metadata["input_args"]
    # args are always as a dict
    # it is serialized as str (thus checking 'force': False) is sufficient
    # it seems joblib is quite smart at serialization. even though in the metadata,
    # it only retains the __repr__, it caches non repr fields too.
    return "'force': False" not in args["steps"]


@fn_call_cache(cache_validation_callback=cache_validation_callback)
def run_steps(steps: List[Dict]):
    if len(steps) == 0:
        return None
    *prev_steps, step = steps
    result = run_steps(prev_steps)
    runner = DataOp.get_runner(step)
    try:
        return runner.run(result, **step)
    except Exception as e:
        logger.warning(f"Error running steps: {str(e)}", steps)
        raise e from e


@dataclass
class DataProvider(Attribute):
    """Base DataProvider Class

    If retrieval_op is defined, will use the provided retrieval_ops, rather than parsing properties etc.
    """

    schema: str = "siibra/attr/data"
    key: str = None  # TODO: remove
    url: str = None  # url can be from remote (http) or localfile

    archive_options: Archive = None

    retrieval_ops: List[Dict] = field(default_factory=list)
    transformation_ops: List[Dict] = field(default_factory=list)

    def assemble_ops(self, **kwargs):
        retrieval_ops, transformation_ops = (
            deepcopy(self.retrieval_ops),
            deepcopy(self.transformation_ops),
        )
        if len(retrieval_ops) > 0:
            return retrieval_ops, transformation_ops

        assert self.url, "url must be defined"

        if self.archive_options is None:
            return [
                RemoteLocalDataOp.generate_specs(filename=self.url)
            ], transformation_ops

        if self.archive_options["format"] == "tar":
            return [
                TarDataOp.generate_specs(
                    url=self.url, filename=self.archive_options["file"]
                )
            ], transformation_ops

        if self.archive_options["format"] == "zip":
            return [
                ZipDataOp.generate_specs(
                    url=self.url, filename=self.archive_options["file"]
                )
            ], transformation_ops

        raise RuntimeError(f"Cannot understand {self.archive_options['format']}")

    def get_data(self, **kwargs):
        retrieval_ops, transformation_ops = self.assemble_ops(**kwargs)
        return run_steps(
            [
                *retrieval_ops,
                *transformation_ops,
            ]
        )

    def describe_data(self):
        return DataOp.describe([*self.retrieval_ops, *self.transformation_ops])
