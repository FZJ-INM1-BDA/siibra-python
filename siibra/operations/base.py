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

from typing import (
    List,
    Dict,
    Type,
    Any,
    TYPE_CHECKING,
    Union,
)
import json
import inspect

if TYPE_CHECKING:
    from ..attributes.datarecipes.base import DataRecipe

try:
    from typing import get_origin
except ImportError:
    from typing_extensions import get_origin

from ..commons.logger import logger


class DataOp:
    """
    Base Data Operation class.

    Derived class must override the following:

    - type (class attribute): explicitly set to None in the case of AbstractDataOp

    Derived class should override the following:

    - run (instance method)
        - should accept input as the first positional argument
        - must expect arguments to be passed as kwargs
        - should gracefully handle None to be passed as arguments (noop)
        - must allow **kwargs wildcard keyword arguments
        - If not overriden, considered as noop
    - generate_specs (class method)
        - must allow keyword arguments
        - must avoid "input" as argument key
        - must avoid "force" as argument key
    - input (class attribute, Type)
        - type of input to be passed run(instance method) as the first positional argument
        - If None, considered source DataOp
        - Used to validate DataOp steps (NYI)
    - output (class attribute, Type)
        - type of output to be returned from run(instance method)
        - Used to validate DataOp steps (NYI)
    - desc (class attribute, str|Callable)
        - human readable description of what this step does
    - force (class attribute, bool)
        - set to True to disable caching

    Derived class should not override anyother attributes/methods/class methods.
    """

    input: None
    output: None
    desc: str = "Noop"
    type: str = "baseop/noop"
    force = False

    step_register: Dict[str, Type["DataOp"]] = {}

    def run(self, input, **kwargs):
        return input

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not cls.type:
            logger.debug(
                f"{cls.__name__} does not have type defined. Cannot register it."
            )
            return
        if cls.desc == DataOp.desc:
            logger.warning(
                f"DataOp {cls} does not have unique description for its step."
            )

        assert cls.type not in cls.step_register, f"Already registered {cls.type}"
        cls.step_register[cls.type] = cls

    @classmethod
    def get_runner_cls(cls, step: Dict):
        _type = step.get("type")
        if _type not in cls.step_register:
            logger.warning(f"{_type} not found in register. Noop rather than hard fail")
            return DataOp
        return cls.step_register[_type]

    @classmethod
    def get_runner(cls, step: Dict):
        return cls.get_runner_cls(step)()

    @classmethod
    def describe(cls, steps: List[Dict], detail=False):
        descs = []
        for idx, step in enumerate(steps, start=1):
            runner = cls.get_runner(step)
            descs.append(
                f"{idx} - {runner.desc(**step, detail=detail) if callable(runner.desc) else runner.desc.format(**step)}"
            )
        return "\n".join(descs)

    @classmethod
    def generate_specs(cls, force=False, **kwargs):
        return {"type": cls.type, "force": force or cls.force}

    @staticmethod
    def get_types(cls: Type, key: str):
        # This is necessary, as some type are stored on superclass
        for k in cls.__mro__:
            try:
                t = k.__annotations__[key]
                if get_origin(t) is not Union:
                    return [t]
                return [t]
            except KeyError:
                continue
        return []

    @classmethod
    def get_output_type(cls, step: Dict):
        Cls = cls.get_runner_cls(step)
        return cls.get_types(Cls, "output")

    @classmethod
    def get_input_type(cls, step: Dict):
        Cls = cls.get_runner_cls(step)
        return cls.get_types(Cls, "input")

    @classmethod
    def update_parameters(cls, step: Dict, **kwargs):
        """Return a shallow copy of the dictionary, updating with/overwriting the relevant values in kwargs"""
        param_names, Runner = cls.get_parameters(step)
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in param_names
        }
        return Runner.generate_specs(
            **{
                **step,
                **filtered_kwargs,
            }
        )

    @classmethod
    def get_parameters(cls, step: Dict):
        Runner = cls.get_runner_cls(step)
        signatures = inspect.signature(Runner.generate_specs)
        adjustable_param_names: List[str] = []
        for parameter in signatures.parameters.values():
            # see: https://docs.python.org/3/library/inspect.html#inspect.Parameter.kind
            # only allow KEYWORD_ONLY or POSITIONAL_OR_KEYWORD
            if parameter.kind not in (
                parameter.KEYWORD_ONLY,
                parameter.POSITIONAL_OR_KEYWORD,
            ):
                continue
            adjustable_param_names.append(parameter.name)

        return adjustable_param_names, Runner


class Merge(DataOp):
    input: None
    Output: List[Any]
    type = "baseop/merge"
    desc = "Merge multiple srcs into a single src, output a list"

    def run(self, input, *, srcs: List[List[Dict]], **kwargs) -> List[Any]:
        from ..attributes.datarecipes.base import run_steps

        return [run_steps(src) for src in srcs]

    @classmethod
    def generate_specs(cls, *, srcs: List[List[Dict]], **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "srcs": srcs}

    def desc(self, *, srcs: List[List[Dict]], detail=False, **kwargs):
        if not detail:
            return f"Merge {len(srcs)} into a single src, output a list"
        return_desc = (
            f"Merge the following {len(srcs)} srcs into a single src, output a list"
        )
        return_desc += "\n\n"
        for src in srcs:
            return_desc += " - " + json.dumps(src, indent=2)
            return_desc += "\n"
        return_desc += "\n"
        return return_desc

    @classmethod
    def spec_from_datarecipes(cls, datarecipes: List["DataRecipe"]):
        srcs: List[Dict] = []
        for dv in datarecipes:
            srcs.append(dv.ops)
        return cls.generate_specs(srcs=srcs)


class FromInstance(DataOp):
    """Utility Source Operation. This operation returns the instance provided, to be used by the next operation."""

    input: None
    output: Any
    type = "baseop/of"
    desc = "Output an {instance}"

    def run(self, _, instance, **kwargs):
        return instance

    @classmethod
    def generate_specs(cls, instance, **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "instance": instance}
