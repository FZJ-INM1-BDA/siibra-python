from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Tuple, Union
from functools import cache

import numpy as np

from ..atlases import Region
from ..dataitems import Image
from ..cache import fn_call_cache

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB, siibra_tqdm
from .parcellationmap import Map

if TYPE_CHECKING:
    from ..locations import BBox
    from nibabel import Nifti1Image


@dataclass
class SparseIndex:
    probs: List[float] = field(default_factory=list)
    bboxes: Dict = field(default_factory=dict)
    voxels: np.ndarray = field(default_factory=np.ndarray)
    affine: np.ndarray = field(default_factory=np.ndarray)
    shape: Tuple[int] = field(default_factory=tuple)

    _SUFFIXES = {
        "probs": ".sparseindex.probs.txt.gz",
        "bboxes": ".sparseindex.bboxes.csv.gz",
        "voxels": ".sparseindex.voxels.nii.gz"
    }

    def get_coords(self, regionname: str):
        # Nx3 array with x/y/z coordinates of the N nonzero values of the given mapindex
        coord_ids = [i for i, l in enumerate(self.probs) if regionname in l]
        x0, y0, z0 = self.bboxes[regionname][:3]
        x1, y1, z1 = self.bboxes[regionname][3:]
        return (
            np.array(
                np.where(
                    np.isin(
                        self.voxels[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1],
                        coord_ids,
                    )
                )
            ).T
            + (x0, y0, z0)
        ).T

    def get_mapped_voxels(self, regionname: str):
        # returns the x, y, and z coordinates of nonzero voxels for the map
        # with the given index, together with their corresponding values v.
        x, y, z = [v.squeeze() for v in np.split(self.get_coords(regionname), 3)]
        v = [self.probs[i][regionname] for i in self.voxels[x, y, z]]
        return x, y, z, v

    def _exract_from_sparseindex(self, regionname: str):
        from nibabel import Nifti1Image

        x, y, z, v = self.get_mapped_voxels(regionname)
        result = np.zeros(self.shape, dtype=np.float32)
        result[x, y, z] = v
        return Nifti1Image(dataobj=result, affine=self.affine)

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
        from gzip import decompress
        import os
        import requests
        from pandas import read_csv

        spindtxt_decoder = lambda b: decompress(b).decode('utf-8').strip().splitlines()

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
            unit="voxels"
        ):
            fields = line.strip().split(" ")
            mapindices = list(map(int, fields[0::2]))
            values = list(map(float, fields[1::2]))
            D = dict(zip(mapindices, values))
            probs.append(D)

        bboxes = {}
        bbox_table = read_csv(
            requests.get(bboxfile).content,
            sep=';',
            compression="gzip",
            index_col=0
        )
        bboxes = bbox_table.T.to_dict('list')

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
        import gzip
        import os
        import pandas as pd
        from nibabel import Nifti1Image

        fullpath = os.path.join(folder, base_filename)

        if folder and not os.path.isdir(folder):
            os.makedirs(folder)

        Nifti1Image(self.voxels, self.affine).to_filename(
            fullpath + SparseIndex._SUFFIXES["voxels"]
        )
        with gzip.open(fullpath + SparseIndex._SUFFIXES["probs"], 'wt') as f:
            for D in self.probs:
                f.write(
                    "{}\n".format(
                        " ".join(f"{r} {p}" for r, p in D.items())
                    )
                )

        bboxtable = pd.DataFrame(
            self.bboxes.values(),
            index=self.bboxes.keys(),
            columns=["x0", "y0", 'z0', "x1", "y1", 'z1']
        )
        bboxtable.to_csv(
            fullpath + SparseIndex._SUFFIXES["bboxes"],
            sep=';',
            compression="gzip"
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
def build_sparse_index(parcmap: Map) -> SparseIndex:
    added_image_count = 0
    spind = {"voxels": {}, "probs": [], "bboxes": {}}
    mapaffine: np.ndarray = None
    mapshape: Tuple[int] = None
    for region, attrcol in siibra_tqdm(
        parcmap._index_mapping.items(),
        unit="map",
        desc=f"Building sparse index from {len(parcmap._index_mapping)} volumetric maps",
    ):
        image = attrcol._get(Image)
        nii = image.fetch()
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
        spind = add_img(spind, nii, region)
        added_image_count += 1
    return SparseIndex(
        probs=spind["probs"],
        bboxes=spind["bboxes"],
        voxels=spind["voxels"],
        affine=mapaffine,
        shape=mapshape,
    )


class SparseMap(Map):
    use_sparse_index: bool = False

    @property
    @cache
    def _sparse_index(self) -> SparseIndex:
        return build_sparse_index(self)

    def fetch(
        self,
        region: Union[str, Region] = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        if isinstance(region, Region):
            regionspec = region.name
        else:
            regionspec = region
        matched = self.parcellation.get_region(regionspec)
        assert matched.name in self.regions, (
            f"Statistical map of region '{matched}' is not available. "
            f"Try fetching its descendants: {(r.name for r in matched.descendants)}"
        )

        if self.use_sparse_index:
            nii = self._sparse_index._exract_from_sparseindex(matched.name)

        nii = super().fetch(region=matched.name, frmt=frmt)

        if bbox:
            from ..retrieval_new.image_fetcher.nifti import extract_voi

            nii = extract_voi(nii, bbox)

        if resolution_mm:
            from ..retrieval_new.image_fetcher.nifti import resample

            nii = resample(nii, resolution_mm)

        return nii
