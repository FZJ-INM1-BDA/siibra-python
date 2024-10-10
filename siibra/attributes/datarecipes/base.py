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

from dataclasses import dataclass, field, replace, asdict
from typing import List, Dict, Tuple, Type, Optional
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
    """
    Run a list of operations, such as from a data recipe, and return the result.
    """
    if len(steps) == 0:
        return None

    # obtain input for the last ste by recursing into previous steps
    *prev_steps, step = steps
    result = run_steps(prev_steps)

    # execute the last step
    runner = DataOp.get_runner(step)
    try:
        return runner.run(result, **step)
    except Exception as e:
        logger.warning(f"Error running steps: {str(e)}")
        logger.info(json.dumps(steps, indent=2))
        raise e from e


# TODO (2.0) convey that the ops *should* be immutable
# TODO (ASAP) figure out how to populate ops arsed from archive_ops etc
@dataclass
class DataRecipe(Attribute):
    """
    Describes a series of operations which produce a well defined, interopable data element.
    The recipe effectively represents the resulting data element while keeping provenance
    and implementing lazy execution of the undelying operations.
    """

    @classmethod
    def _generate_ops(cls, conf: Dict):
        """From configuration definition of DataRecipe, generates the list of ops."""
        url = conf.get("url")
        archive_options = conf.get("archive_options")
        assert url, "url must be defined"
        if archive_options is None:
            return [RemoteLocalDataOp.generate_specs(filename=url)]
        archive_format = archive_options.get("format")
        if archive_format == "tar":
            return [TarDataOp.generate_specs(url=url, filename=archive_options["file"])]
        if archive_format == "zip":
            return [ZipDataOp.generate_specs(url=url, filename=archive_options["file"])]
        raise RuntimeError(f"Cannot understand {archive_format}")

    def __post_init__(self):
        """
        From configuration definition of DataRecipe, generates the list of ops.
        TODO (2.1) at the moment, this violates the principle that DataRecipe should be immutable. But this is a quick way to get
        DataRecipe to work. In future, migrate this to __new__, and use @datacclass(frozen=True)
        TODO (ASAP) check to use factory pattern
        """

        if len(self._ops) > 0:
            return

        self_dict = asdict(self)
        ops = self._generate_ops(self_dict)
        self._ops.extend(ops)

    schema: str = None
    _ops: List[Dict] = field(default_factory=list)

    url: str = None
    archive_options: Dict = None

    @property
    def ops(self):
        return self._ops

    @ops.setter
    def ops(self, value):
        raise RuntimeError("Please use reconfigure instance method")

    def get_parameters(self, *arg, **kwargs):
        import pandas as pd

        r: List[Tuple[str, Type[DataOp]]] = []
        for op in self.ops:
            param_names, Runner = DataOp.get_parameters(op)
            for param_name in param_names:
                r.append([param_name, Runner])

        return pd.DataFrame(r, columns=["param_name", "Runner"])

    def reconfigure(self, **kwargs) -> "DataRecipe":
        """
        Given a set of kwargs, return a shallow copy of DataRecipe, generating data element conforming to the updated configuration
        """

        steps = []
        for op in self._ops:

            # safe overwrite kwargs
            Cls = DataOp.get_runner_cls(op)
            modified_step = Cls.update_parameters(op, **kwargs)
            steps.append(modified_step)

        return replace(self, _ops=steps)

    def get_data(self):
        return run_steps(self.ops)
