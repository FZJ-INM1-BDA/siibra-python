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

from typing import List, Dict, Type

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
    input: None
    output: None
    desc: str = "Noop"

    step_register: Dict[str, Type["DataOp"]] = {}

    def run(self, input, **kwargs):
        return input

    def __init_subclass__(cls, type: str = None) -> None:
        if not type:
            return
        assert type not in cls.step_register, f"Already registered {type}"
        cls.step_register[type] = cls

    @classmethod
    def get_runner(cls, step: Dict):
        _type = step.get("type")
        assert _type in cls.step_register, f"{_type} not found in step register"
        return cls.step_register[_type]

    @classmethod
    def describe(cls, steps: List[Dict]):
        descs = []
        for idx, step in enumerate(steps, start=1):
            runner = cls.get_runner(step)
            descs.append(f"{idx} - {runner.desc.format(step)}")
        return "\n".join(descs)
