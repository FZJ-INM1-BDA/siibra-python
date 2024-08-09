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
from typing import List, Dict, Tuple, Union
import numpy as np
import json
from pathlib import Path
from nibabel import Nifti1Image, load as nib_load
import gzip
import os
import pandas as pd
import requests
from gzip import decompress
from abc import ABC, abstractmethod
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .parcellationmap import Map
from ..cache import fn_call_cache
from ..commons_new.logger import siibra_tqdm, logger
from ..commons_new.iterable import assert_ooo


class ABCSparseIndex(ABC):

    @abstractmethod
    def save(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_img(self, nii: Nifti1Image, regionname: str, **kwargs) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def read(self, pos: np.ndarray, **kwargs) -> List[Dict[str, float]]:
        raise NotImplementedError
    
    @abstractmethod
    def get_bbox(self, regionname: str, **kwargs) -> List[int]:
        raise NotImplementedError


@dataclass(repr=False)
class SparseIndex:
    probs: List[Dict[str, float]] = field(default_factory=list)
    bboxes: Dict = field(default_factory=dict)
    voxels: np.ndarray = field(default_factory=np.ndarray)
    affine: np.ndarray = field(default_factory=np.ndarray)
    shape: Tuple[int] = field(default_factory=tuple)

    _SUFFIXES = {
        "probs": ".sparseindex.probs.txt.gz",
        "bboxes": ".sparseindex.bboxes.csv.gz",
        "voxels": ".sparseindex.voxels.nii.gz",
    }

    @classmethod
    def load(cls, filepath_or_url: str) -> "SparseIndex":
        """
        Loads a precomputed SparseIndex to the memory.

        Parameters
        ----------
        filepath_or_url: str
            Path/url to the SparseIndex files
            (eg. https://url_to_files/basefilename):
            - basefilename.sparseindex.probs.txt.gz
            - basefilename.sparseindex.bboxes.csv.gz
            - basefilename.sparseindex.voxels.nii.gz

        Returns
        -------
        SparseIndex
        """

        spindtxt_decoder = lambda b: decompress(b).decode("utf-8").strip().splitlines()

        probsfile = filepath_or_url + SparseIndex._SUFFIXES["probs"]
        bboxfile = filepath_or_url + SparseIndex._SUFFIXES["bboxes"]
        voxelfile = filepath_or_url + SparseIndex._SUFFIXES["voxels"]
        assert all(os.path.isfile(f) for f in [probsfile, bboxfile, voxelfile])

        voxels_nii = Nifti1Image.from_bytes(requests.get(voxelfile).content)
        affine = voxels_nii.affine
        shape = voxels_nii.shape
        voxels = np.asanyarray(voxels_nii.dataobj)

        probs = []
        lines_probs = spindtxt_decoder(requests.get(probsfile).content)
        for line in siibra_tqdm(
            lines_probs,
            total=len(lines_probs),
            desc="Loading sparse index",
            unit="voxels",
        ):
            fields = line.strip().split(" ")
            mapindices = list(map(int, fields[0::2]))
            values = list(map(float, fields[1::2]))
            D = dict(zip(mapindices, values))
            probs.append(D)

        bboxes = {}
        bbox_table = pd.read_csv(
            requests.get(bboxfile).content, sep=";", compression="gzip", index_col=0
        )
        bboxes = bbox_table.T.to_dict("list")

        return cls.__init__(
            probs=probs,
            bboxes=bboxes,
            voxels=voxels,
            affine=affine,
            shape=shape,
        )

    def save(self, base_filename: str, folder: str = ""):
        """
        Save SparseIndex (3x) files to under the folder `folder`
        with base_filename. If SparseIndex is not cached, siibra will first
        create it first.
        Parameters
        ----------
        base_filename: str
            The files that will be created as:
            - base_filename.sparseindex.probs.txt.gz
            - base_filename.sparseindex.bboxes.txt.gz
            - base_filename.sparseindex.voxels.nii.gz
        folder: str, default=""
        """

        fullpath = os.path.join(folder, base_filename)

        if folder and not os.path.isdir(folder):
            os.makedirs(folder)

        Nifti1Image(self.voxels, self.affine).to_filename(
            fullpath + SparseIndex._SUFFIXES["voxels"]
        )
        with gzip.open(fullpath + SparseIndex._SUFFIXES["probs"], "wt") as f:
            for D in self.probs:
                f.write("{}\n".format(" ".join(f"{r} {p}" for r, p in D.items())))

        bboxtable = pd.DataFrame(
            self.bboxes.values(),
            index=self.bboxes.keys(),
            columns=["x0", "y0", "z0", "x1", "y1", "z1"],
        )
        bboxtable.to_csv(
            fullpath + SparseIndex._SUFFIXES["bboxes"], sep=";", compression="gzip"
        )


def add_img(spind: dict, nii: "Nifti1Image", regionname: str):
    imgdata = np.asanyarray(nii.dataobj)
    X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
    for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
        coord_id = spind["voxels"][x, y, z]
        if coord_id >= 0:
            # Coordinate already seen. Just add observed value.
            assert regionname not in spind["probs"][coord_id]
            assert len(spind["probs"]) > coord_id
            spind["probs"][coord_id][regionname] = prob
        else:
            # New coordinate. Append entry with observed value.
            coord_id = len(spind["probs"])
            spind["voxels"][x, y, z] = coord_id
            spind["probs"].append({regionname: prob})

    spind["bboxes"][regionname] = (X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max())
    return spind


@fn_call_cache
def build_sparse_index(parcmap: "SparseMap") -> SparseIndex:
    added_image_count = 0
    spind = {"voxels": {}, "probs": [], "bboxes": {}}
    mapaffine: np.ndarray = None
    mapshape: Tuple[int] = None
    for region in siibra_tqdm(
        parcmap.regions,
        unit="map",
        desc=f"Building sparse index from {len(parcmap.regions)} volumetric maps",
    ):
        vol = assert_ooo(parcmap.find_volumes(region=region))
        nii = vol.fetch()
        if added_image_count == 0:
            mapaffine = nii.affine
            mapshape = nii.shape
            spind["voxels"] = np.zeros(nii.shape, dtype=np.int32) - 1
        else:
            if (nii.shape != mapshape) or ((mapaffine - nii.affine).sum() != 0):
                raise RuntimeError(
                    "Building sparse maps from volumes with different voxel "
                    "spaces is not yet supported in siibra."
                )
        add_img(spind, nii, region)
        added_image_count += 1
    return SparseIndex(
        probs=spind["probs"],
        bboxes=spind["bboxes"],
        voxels=spind["voxels"],
        affine=mapaffine,
        shape=mapshape,
    )


class MESI(ABCSparseIndex):

    HEADER = """MESI-UTF8-V0"""

    META_SUFFIX = ".mesi.meta.txt"
    PROBS_SUFFIX = ".mesi.probs.txt"
    VOXEL_SUFFIX = ".mesi.voxel.nii.gz"

    UINT32_MAX = 4_294_967_295

    def __init__(self, base_filename: str, folder: str="", mode=Literal["r", "w"]):
        self.base_filename = base_filename
        self.folder = folder
        self.mode = mode

        # regionname -> regionalias (implicitly from str of element index in list, i.e. '0', '1', '2', etc)
        self.region_name_mapping: List[str] = []

        # regionalias -> prob
        self.probs: List[
            Dict[str, float]
        ] = []

        # regionalias (implicitly from element indx in list) -> prob
        self.bbox: List[List[int]] = []

        # voxel coord -> element index in self.probs
        self.voxels: Dict[
            Tuple[int, int, int],
            int
        ] = {}

        self._shape = None
        self._affine = None
        self._readable_nii = None
        self._readable_meta = None
    
    @property
    def shape(self):
        if self.readable:
            return self.readable_nii.dataobj.shape
        return self._shape

    @shape.setter
    def shape(self, value):
        if not self.writable:
            logger.warning(f"Attempting to set shape in non writable sparseindex. Ignored.")
            return
        self._shape = value
    
    @property
    def affine(self):
        if self.readable:
            return self.readable_nii.affine
        return self._affine

    @affine.setter
    def affine(self, value):
        if not self.writable:
            logger.warning(f"Attempting to set affine in non writable sparseindex. Ignored.")
            return
        self._affine = value

    @property
    def readable_nii(self) -> Nifti1Image:
        if not self.readable:
            return
        
        filename = Path(self.folder) / (self.base_filename + self.VOXEL_SUFFIX)
        if not self._readable_nii:
            self._readable_nii = nib_load(filename)
        return self._readable_nii

    @property
    def readable_meta(self):
        if not self.readable:
            return
        
        filename = Path(self.folder) / (self.base_filename + self.META_SUFFIX)
        if not self._readable_meta:
            with open(filename, "r") as fp:
                self._readable_meta = fp.read()
        return self._readable_meta

    @property
    def readable(self):
        return self.mode == "r"
    
    @property
    def writable(self):
        return self.mode == "w"
    
    def _decode_regionalias(self, alias: str):
        if not self.readable:
            raise RuntimeError(f"Cannot decode a non-readable SparseIndex")
        lines = self.readable_meta.splitlines()
        return json.loads(lines[int(alias) + 1]).get("regionname")
    
    def read(self, pos: Union[List[List[int]], np.ndarray]):
        if not self.readable:
            raise RuntimeError(f"SparseIndex not readable")
        
        pos = np.array(pos)
        assert len(pos.shape) == 2 and pos.shape[1] == 3, f"Expecting Nx3 array, but got {pos.shape}"
        
        probfile = Path(self.folder) / f"{self.base_filename}{self.PROBS_SUFFIX}"
        fp = open(probfile, "r")

        nii = np.array(self.readable_nii.dataobj, dtype=np.uint64)
        x, y, z = pos.T
        
        result = []
        for val in nii[x, y, z].tolist():
            offset = val >> 32
            bytes_to_read = val & self.UINT32_MAX
            fp.seek(int(offset))
            decoded = fp.read(int(bytes_to_read))
            prob = {
                self._decode_regionalias(key): prob
                for key, prob in json.loads(decoded).items()
            }
            result.append(prob)
        return result

    def save(self):
        if not self.writable:
            raise RuntimeError(f"Readable only")

        if self.affine is None:
            raise RuntimeError(f"No image has been added yet")
        
        _dir = Path(self.folder)
        _dir.mkdir(parents=True, exist_ok=True)

        # newline-separated json objects on all region metadata
        regionmeta = "\n".join(json.dumps({ "regionname": regionname, "bbox": bbox })
            for regionname, bbox
            in zip(self.region_name_mapping, self.bbox))
        
        # write metadata
        with open(_dir / (self.base_filename + self.META_SUFFIX), "w") as fp:
            fp.write(self.HEADER)
            fp.write("\n")
            fp.write(regionmeta)

        current_offset = 0

        # offset/bytes
        # only recording offset, since the bytes can be calculated by getting the next contiguous offset
        # inherits element-wise index of self.probs
        offset_record: List[Tuple[int, int]] = []

        with open(_dir / (self.base_filename + self.PROBS_SUFFIX), "w") as fp:
            
            for prob in self.probs:
                str_to_write = json.dumps(prob) + "\n"
                byte_count = len(str_to_write.encode("utf-8"))
                fp.write(str_to_write)

                # purely for ascethic and easy debugging purpose

                offset_record.append((current_offset, byte_count))
                current_offset += byte_count

        # saving voxel map, containing offset
        lut = np.zeros(self.shape, dtype=np.uint64, order="C")
        for (x, y, z), list_idx in self.voxels.items():
            offset, bytes_used = offset_record[list_idx]
            assert offset < self.UINT32_MAX, f"offset > unit32 max"
            # print(offset << 32)
            lut[x, y, z] = np.uint64(offset << 32) + np.uint64(bytes_used)

        nii = Nifti1Image(lut, affine=self.affine, dtype=np.uint64)
        nii.to_filename(_dir / (self.base_filename + self.VOXEL_SUFFIX))

    def add_img(self, nii: "Nifti1Image", regionname: str):
        if not self.writable:
            raise RuntimeError(f"Readable only")

        if self.shape is None:
            self.shape = nii.dataobj.shape
        else:
            assert np.all(self.shape == nii.dataobj.shape), f"Sparse index from different shape not supported is None"

        if self.affine is None:
            self.affine = nii.affine
        else:
            assert np.all(self.affine == nii.affine), f"Sparse index from different affine is not supported"

        assert regionname not in self.region_name_mapping, f"{regionname} has already been mapped"

        # eventually, the alias will become dictionary keys. JSON keys *must* be str
        regionalias = str(len(self.region_name_mapping))
        self.region_name_mapping.append(regionname)

        imgdata = np.asanyarray(nii.dataobj)
        X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
        for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
            coord_id = (x, y, z)
            prob = prob.astype("float")

            if coord_id not in self.voxels:
                self.probs.append({regionalias: prob})
                self.voxels[coord_id] = len(self.probs) - 1

            if coord_id in self.voxels:
                probidx = self.voxels[coord_id]
                self.probs[probidx][regionalias] = prob

        self.bbox.append(
            np.array([X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max()]).tolist()
        )

    def get_bbox(self, regionname: str, **kwargs) -> List[int]:
        for line in self.readable_meta.splitlines()[1:]:
            parsed_line = json.loads(line)
            if regionname == parsed_line.get("regionname"):
                return parsed_line.get("bbox")

@dataclass(repr=False, eq=False)
class SparseMap(Map):
    use_sparse_index: bool = False

    @property
    def _sparse_index(self) -> SparseIndex:
        return build_sparse_index(self)

