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

from typing import List, Dict, Type, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ..attributes.dataproviders.base import DataProvider

from ..commons.logger import logger

src_remote_tar = {
    "type": "src/remotetar",
    "tar": "http://foo/bar.tar",
    "filename": "bazz/bud.txt",
}

src_local_tar = {
    "type": "src/localtar",
    "tar": "/tmp/bar.tar",
    "filename": "bazz/bud.txt",
}

src_remote_file = {"type": "src/file", "url": "http://foo/bud.txt"}

src_local_file = {"type": "src/file", "url": "/foo/bud.txt"}

codec_gzip = {
    "type": "codec/gzip",
    "op": "decompress",
}


codec_slice = {"type": "codec/slice", "offset": 200, "bytes": 24}

read_nib = {"type": "read/nifti"}

read_csv = {"type": "read/csv"}

codec_vol_slice = {
    "type": "codec/vol/slice",
    "param": [],
}

codec_vol_mask = {"type": "codec/vol/mask", "threshold": 0.1}

codec_vol_extract_label = {"type": "codec/vol/extractlabel", "labels": [1, 5, 11]}

codec_vol_to_bytes = {"type": "codec/vol/tobytes"}

dst_return = {"type": "dst/return"}

dst_save_to_file = {"type": "dst/save", "filename": "foo.nii.gz"}

dst_upload = {
    "type": "dst/upload/dataproxy",
    "token": "ey...",
    "bucket": "mybucket",
    "filename": "foobar.nii.gz",
}

"""
proposal, DataProvider have a series of "step"'s, which describe everything, such as:

- how data is retrieved
- how data is transformed
- how data is saved/returned

to note:

- In the current implementation, Step is modelled as a class. So each subclass implement a `run` method.
- The run method takes a positional argument (input), a cfg kwarg, and arbitary kwargs.
- Each step *should* have proper type annotation (so static/runtime type check can be performed, to avoid runtime issue)
- The steps may choose to ignore the input (would be the case for source steps, such as fetching )
- similarly, the steps may choose to not return anything (i.e. return None), if it is a final step (e.g. save to a file)

we can also create a text description on a "dry-run". e.g.

1. fetch tar file from http://, extract and get filename
2. gunzip the file
3. read it as nifti with nibabel
4. mask it with threshold 0.3
5. convert the nifti to bytes
6. gzip the result
7. save to my.nii.gz

potentially, we can do things like forking/merging etc (need to be a bit careful about mutability of data)

"""


class DataOp:
    """
    Base Data Operation class. type *must* be overriden (or set to None). desc *should* be overriden, or will trigger warning messages.
    Set force=True to disable cache. User could also pass "force": True in generate_specs to manually bypass the cache.
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
    def get_runner(cls, step: Dict):
        _type = step.get("type")
        if _type not in cls.step_register:
            logger.warning(f"{_type} not found in register. Noop rather than hard fail")
            return DataOp()
        return cls.step_register[_type]()

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


class Merge(DataOp):
    input: None
    Output: List[Any]
    type = "baseop/merge"
    desc = "Merge multiple srcs into a single src, output a list"

    def run(self, input, *, srcs: List[List[Dict]], **kwargs) -> List[Any]:
        from ..attributes.dataproviders.base import run_steps

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
    def spec_from_dataproviders(cls, dataproviders: List["DataProvider"]):
        assert all(
            len(dp.transformation_ops) == 0 for dp in dataproviders
        ), f"Expected no transformops to be in data providers"
        srcs: List[Dict] = []
        for dv in dataproviders:
            retrival, transformer = dv.assemble_ops()
            src = [*retrival, *transformer]
            srcs.append(src)
        return cls.generate_specs(srcs=srcs)


# TODO: Rename the class
class Of(DataOp):
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
