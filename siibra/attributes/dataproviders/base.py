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

    schema: str = None
    ops: List[Dict] = field(default_factory=list)
    update_kwargs: List[Dict] = field(default_factory=list)

    def get_parameters(self, *arg, **kwargs):
        import pandas as pd

        r: List[Tuple[str, Type[DataOp]]] = []
        for op in self.ops:
            param_names, Runner = DataOp.get_parameters(op)
            for param_name in param_names:
                r.append([param_name, Runner])

        return pd.DataFrame(r, columns=["param_name", "Runner"])

    def reconfigure(self, *args, **kwargs) -> "DataRecipe":
        """
        Given a set of kwargs, return a shallow copy of DataRecipe, generating data element conforming to the updated configuration
        """

        return replace(self, updates=[*self.update_kwargs, kwargs])

    def get_data(self):
        return run_steps(self.ops)
