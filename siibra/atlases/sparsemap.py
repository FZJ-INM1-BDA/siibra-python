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

from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, TYPE_CHECKING
import numpy as np
import json
from pathlib import Path
import nibabel as nib
import requests
import gzip

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .parcellationmap import Map
from .region import Region
from ..retrieval.volume_fetcher import FetchKwargs, SIIBRA_MAX_FETCH_SIZE_GIB
from ..retrieval.file_fetcher.io.base import PartialReader

if TYPE_CHECKING:
    from ..attributes.locations import BoundingBox


class SparseIndex:
    HEADER = """MESI-UTF8-V0"""

    META_SUFFIX = ".mesi.meta.txt"
    PROBS_SUFFIX = ".mesi.probs.txt"
    VOXEL_SUFFIX = ".mesi.voxel.nii.gz"

    UINT32_MAX = 4_294_967_295

    readable = False
    writable = False

    def __new__(cls, *args, mode: Literal["r", "w"] = "r", **kwargs):
        if mode == "r":
            instance = object.__new__(ReadableSparseIndex)
            return instance
        if mode == "w":
            instance = object.__new__(WritableSparseIndex)
            return instance
        return super().__new__(cls)

    def __init__(self, filepath: Union[str, Path], mode: Literal["r", "w"] = "r"):
        self.filepath = None
        self.url = None
        self.mode = mode

        if isinstance(filepath, str) and filepath.startswith("https://"):
            self.url = filepath
            return
        self.filepath = Path(filepath)

    def read(self, pos: Union[List[List[int]], np.ndarray]):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def add_img(self, nii: nib.Nifti1Image, regionname: str):
        raise NotImplementedError

    def get_boundingbox_extrema(self, regionname: str, **kwargs) -> List[int]:
        raise NotImplementedError

    @property
    def affine(self):
        raise NotImplementedError


class WritableSparseIndex(SparseIndex):
    writable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.filepath is None:
            raise RuntimeError("WritableSparseIndex must point to a local file")

        # regionname -> regionalias (implicitly from str of element index in list, i.e. '0', '1', '2', etc)
        self._region_name_mapping: List[str] = []

        # regionalias -> prob
        self._probs: List[Dict[str, float]] = []

        # regionalias (implicitly from element indx in list) -> prob
        self._bbox: List[List[int]] = []

        # voxel coord -> element index in self.probs
        self._voxels: Dict[Tuple[int, int, int], int] = {}

        self._shape = None
        self._affine = None

    def save(self):
        if self._affine is None:
            raise RuntimeError("No image has been added yet")

        basename = self.filepath
        basename.parent.mkdir(parents=True, exist_ok=True)

        # newline-separated json objects on all region metadata
        regionmeta = "\n".join(
            json.dumps({"regionname": regionname, "bbox": bbox})
            for regionname, bbox in zip(self._region_name_mapping, self._bbox)
        )

        # write metadata
        with open(basename.with_suffix(self.META_SUFFIX), "w") as fp:
            fp.write(self.HEADER)
            fp.write("\n")
            fp.write(regionmeta)

        current_offset = 0

        # offset/bytes
        # only recording offset, since the bytes can be calculated by getting the next contiguous offset
        # inherits element-wise index of self.probs
        offset_record: List[Tuple[int, int]] = []

        with open(basename.with_suffix(self.PROBS_SUFFIX), "w") as fp:

            for prob in self._probs:
                str_to_write = json.dumps(prob) + "\n"
                byte_count = len(str_to_write.encode("utf-8"))
                fp.write(str_to_write)

                # purely for ascethic and easy debugging purpose

                offset_record.append((current_offset, byte_count))
                current_offset += byte_count

        # saving voxel map, containing offset
        lut = np.zeros(self._shape, dtype=np.uint64, order="C")
        for (x, y, z), list_idx in self._voxels.items():
            offset, bytes_used = offset_record[list_idx]
            assert offset < self.UINT32_MAX, "offset > unit32 max"
            # print(offset << 32)
            lut[x, y, z] = np.uint64(offset << 32) + np.uint64(bytes_used)

        nii = nib.Nifti1Image(lut, affine=self._affine, dtype=np.uint64)
        nii.to_filename(basename.with_suffix(self.VOXEL_SUFFIX))

    def add_img(self, nii: nib.Nifti1Image, regionname: str):
        if self._shape is None:
            self._shape = nii.dataobj.shape
        else:
            assert np.all(
                self._shape == nii.dataobj.shape
            ), "Sparse index from different shape not supported is None"

        if self._affine is None:
            self._affine = nii.affine
        else:
            assert np.all(
                self._affine == nii.affine
            ), "Sparse index from different affine is not supported"

        assert (
            regionname not in self._region_name_mapping
        ), f"{regionname} has already been mapped"

        # eventually, the alias will become dictionary keys. JSON keys *must* be str
        regionalias = str(len(self._region_name_mapping))
        self._region_name_mapping.append(regionname)

        imgdata = np.asanyarray(nii.dataobj)
        X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
        for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
            coord_id = (x, y, z)
            prob = prob.astype("float")

            if coord_id not in self._voxels:
                self._probs.append({regionalias: prob})
                self._voxels[coord_id] = len(self._probs) - 1

            if coord_id in self._voxels:
                probidx = self._voxels[coord_id]
                self._probs[probidx][regionalias] = prob

        self._bbox.append(
            np.array([X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max()]).tolist()
        )

    @property
    def affine(self):
        if self._affine is None:
            raise RuntimeError(
                "You must call .add_img first, before the affine is populated."
            )
        return self._affine


class ReadableSparseIndex(SparseIndex):
    readable = True

    def __init__(self, filepath: Union[str, Path], **kwargs):
        super().__init__(filepath)

        self._readable_nii = None
        self._readable_meta = None
        self._affine = None

        if self.url:
            session = requests.Session()
            resp = session.get(self.url + self.VOXEL_SUFFIX)
            resp.raise_for_status()
            content = resp.content
            try:
                content = gzip.decompress(content)
            except gzip.BadGzipFile:
                ...
            self._readable_nii = nib.Nifti1Image.from_bytes(content)

            resp = session.get(self.url + self.META_SUFFIX)
            resp.raise_for_status()
            self._readable_meta = resp.content.decode("utf-8")
        if self.filepath:
            self._readable_nii = nib.load(self.filepath.with_suffix(self.VOXEL_SUFFIX))
            with open(self.filepath.with_suffix(self.META_SUFFIX), "r") as fp:
                self._readable_meta = fp.read()

    @property
    def affine(self):
        if self._affine is None:
            self._affine = self._readable_nii.affine
        return self._affine

    def _decode_regionalias(self, alias: str):
        if not self.readable:
            raise RuntimeError("Cannot decode a non-readable SparseIndex")
        lines = self._readable_meta.splitlines()
        return json.loads(lines[int(alias) + 1]).get("regionname")

    def read(self, pos: Union[List[List[int]], np.ndarray]):

        probreader = PartialReader(str(self.url or self.filepath) + self.PROBS_SUFFIX)
        probreader.open()

        pos = np.array(pos)
        assert (
            len(pos.shape) == 2 and pos.shape[1] == 3
        ), f"Expecting Nx3 array, but got {pos.shape}"

        nii = np.array(self._readable_nii.dataobj, dtype=np.uint64)
        x, y, z = pos.T

        result = []
        for val in nii[x, y, z].tolist():
            offset = val >> 32
            bytes_to_read = val & self.UINT32_MAX
            probreader.seek(int(offset))
            decoded = probreader.read(int(bytes_to_read))
            prob = {
                self._decode_regionalias(key): prob
                for key, prob in json.loads(decoded).items()
            }
            result.append(prob)

        probreader.close()
        return result

    def get_boundingbox_extrema(self, regionname: str, **kwargs) -> List[int]:
        for line in self._readable_meta.splitlines()[1:]:
            parsed_line = json.loads(line)
            if regionname == parsed_line.get("regionname"):
                return parsed_line.get("bbox")


@dataclass(repr=False, eq=False)
class SparseMap(Map):

    def fetch(
        self,
        region: Union[str, Region] = None,
        frmt: str = None,
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        if region is None:
            assert len(self.regions) == 1, ValueError(
                "To fetch a volume from a SparseMap, please provide a region "
                "name mapped in this SparseMap."
            )
            matched = self.regions[0]
        else:
            if isinstance(region, Region):
                matched = region.name
            else:
                matched = self.parcellation.get_region(region).name

        assert (
            matched in self.regions
        ), f"Statistical map of region '{matched}' is not available in '{self.name}'."

        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
        )
        return super().fetch(region=matched, frmt=frmt, **fetch_kwargs)
