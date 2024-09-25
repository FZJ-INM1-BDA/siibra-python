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

from dataclasses import dataclass, field, replace
from typing import List, Dict
import json

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
        logger.warning(f"Error running steps: {str(e)}")
        logger.info(json.dumps(steps, indent=2))
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

    override_ops: List[Dict] = field(default_factory=list)
    transformation_ops: List[Dict] = field(default_factory=list)

    @property
    def retrieval_ops(self):

        assert self.url, "url must be defined"

        if self.archive_options is None:
            return [RemoteLocalDataOp.generate_specs(filename=self.url)]

        if self.archive_options["format"] == "tar":
            return [
                TarDataOp.generate_specs(
                    url=self.url, filename=self.archive_options["file"]
                )
            ]

        if self.archive_options["format"] == "zip":
            return [
                ZipDataOp.generate_specs(
                    url=self.url, filename=self.archive_options["file"]
                )
            ]

        raise RuntimeError(f"Cannot understand {self.archive_options['format']}")

    @property
    def ops(self):
        if len(self.override_ops) > 0:
            return self.override_ops
        return [*self.retrieval_ops, *self.transformation_ops]

    @property
    def current_output_type(self):
        return DataOp.get_output_type(self.ops[-1])

    def append_op(self, op: Dict):
        input_type = DataOp.get_input_type(op)
        if input_type != self.current_output_type:
            logger.debug(
                f"append_op {op} potential issue: current output type is {self.current_output_type}, but input type is {input_type}"
            )
        if len(self.override_ops) > 0:
            self.override_ops.append(op)
            return
        self.transformation_ops.append(op)

    def extend_ops(self, ops: List[Dict]):
        for op in ops:
            self.append_op(op)

    def pop_op(self, index=-1):
        if len(self.override_ops) > 0:
            if len(self.override_ops) == 1:
                raise IndexError(f"overridden image provider cannot be fully popped")
            return self.override_ops.pop(index)
        return self.transformation_ops.pop(index)

    def get_data(self):
        return run_steps(self.ops)

    def describe_data(self):
        return DataOp.describe(self.ops)

    def query(self, *arg, **kwargs):
        """
        Returns a copy of the data provider.
        """
        return replace(self)
