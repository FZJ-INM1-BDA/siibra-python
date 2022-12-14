# Copyright 2018-2021
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

from . import volume

from ..commons import logger, MapType
from ..retrieval import requests, cache
from ..locations import boundingbox

from neuroglancer_scripts.precomputed_io import get_IO_for_existing_dataset
from neuroglancer_scripts.accessor import get_accessor_for_url
from neuroglancer_scripts.mesh import read_precomputed_mesh, affine_transform_mesh
from io import BytesIO
import nibabel as nib
import os
import numpy as np


class NeuroglancerVolumeFetcher(volume.VolumeProvider, srctype="neuroglancer/precomputed"):
    # Number of bytes at which an image array is considered to large to fetch
    MAX_GiB = 0.2

    # Wether to keep fetched data in local cache
    USE_CACHE = False

    @property
    def MAX_BYTES(self):
        return self.MAX_GiB * 1024 ** 3

    def __init__(self, url: str):
        volume.VolumeProvider.__init__(self)
        self.url = url
        self._scales_cached = None
        self._io = None

    _transform_nm = None

    @property
    def transform_nm(self):
        if self._transform_nm is not None:
            return self._transform_nm
        try:
            res = requests.HttpRequest(f"{self.url}/transform.json").get()
        except requests.SiibraHttpRequestError:
            res = None
        if res is not None:
            self._transform_nm = np.array(res)
            return self._transform_nm

        self._transform_nm = np.identity(1)
        logger.warn(f"No transform.json found at {self.url}, using identity.")
        return self._transform_nm

    @transform_nm.setter
    def transform_nm(self, val):
        self._transform_nm = val

    @property
    def map_type(self):
        if self._io is None:
            self._bootstrap()
        return (
            MapType.LABELLED
            if self._io.info.get("type") == "segmentation"
            else MapType.CONTINUOUS
        )

    @map_type.setter
    def map_type(self, val):
        if val is not None:
            logger.debug(
                "NeuroglancerVolume can determine its own maptype from self._io.info.get('type')"
            )

    def _bootstrap(self):
        accessor = get_accessor_for_url(self.url)
        self._io = get_IO_for_existing_dataset(accessor)
        self._scales_cached = sorted(
            [NeuroglancerScale(self, i) for i in self._io.info["scales"]]
        )

    @property
    def dtype(self):
        if self._io is None:
            self._bootstrap()
        return np.dtype(self._io.info["data_type"])

    @property
    def scales(self):
        if self._scales_cached is None:
            self._bootstrap()
        return self._scales_cached

    def fetch(self, resolution_mm: float = None, voi: boundingbox.BoundingBox = None, **kwargs):
        # the caller has to make sure voi is defined in the correct reference space
        scale = self._select_scale(resolution_mm=resolution_mm, bbox=voi)
        return scale.fetch(voi)

    def get_shape(self, resolution_mm=None):
        scale = self._select_scale(resolution_mm)
        return scale.size

    def is_float(self):
        return self.dtype.kind == "f"

    def _select_scale(self, resolution_mm: float, bbox: boundingbox.BoundingBox = None):
        if resolution_mm is None:
            suitable = self.scales
        elif resolution_mm < 0:
            suitable = [self.scales[0]]
        else:
            suitable = sorted(s for s in self.scales if s.resolves(resolution_mm))

        if len(suitable) > 0:
            scale = suitable[-1]
        else:
            scale = self.scales[0]
            logger.warn(
                f"Requested resolution {resolution_mm} is not available. "
                f"Falling back to the highest possible resolution of "
                f"{', '.join(map('{:.2f}'.format, scale.res_mm))} mm."
            )

        scale_changed = False
        while scale._estimate_nbytes(bbox) > self.MAX_BYTES:
            scale = scale.next()
            scale_changed = True
            if scale is None:
                raise RuntimeError(
                    f"Fetching bounding box {bbox} is infeasible "
                    f"relative to the limit of {self.MAX_BYTES/1024**3}GiB."
                )
        if scale_changed:
            logger.warn(f"Resolution was reduced to {scale.res_mm} to provide a feasible volume size")
        return scale


class NeuroglancerScale:
    """One scale of a NeuroglancerVolume."""

    color_warning_issued = False

    def __init__(self, volume: NeuroglancerVolumeFetcher, scaleinfo: dict):
        self.volume = volume
        self.chunk_sizes = np.array(scaleinfo["chunk_sizes"]).squeeze()
        self.encoding = scaleinfo["encoding"]
        self.key = scaleinfo["key"]
        self.res_nm = np.array(scaleinfo["resolution"]).squeeze()
        self.size = scaleinfo["size"]
        self.voxel_offset = np.array(scaleinfo["voxel_offset"])

    @property
    def res_mm(self):
        return self.res_nm / 1e6

    def resolves(self, resolution_mm):
        """Test wether the resolution of this scale is sufficient to provide the given resolution."""
        return any(r / 1e6 <= resolution_mm for r in self.res_nm)

    def __lt__(self, other):
        """Sort scales by resolution."""
        return all(self.res_nm[i] < other.res_nm[i] for i in range(3))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__} {self.key}"

    def _estimate_nbytes(self, bbox: boundingbox.BoundingBox = None):
        """Estimate the size image array to be fetched in bytes, given a bounding box."""
        if bbox is None:
            bbox_ = boundingbox.BoundingBox((0, 0, 0), self.size, space=None)
        else:
            bbox_ = bbox.transform(np.linalg.inv(self.affine))
        result = self.volume.dtype.itemsize * bbox_.volume
        logger.debug(
            f"Approximate size for fetching resolution "
            f"({', '.join(map('{:.2f}'.format, self.res_mm))}) mm "
            f"is {result/1024**3:.2f} GiB."
        )
        return result

    def next(self):
        """Returns the next scale in this volume, of None if this is the last."""
        my_index = self.volume.scales.index(self)
        if my_index < len(self.volume.scales):
            return self.volume.scales[my_index + 1]
        else:
            return None

    def prev(self):
        """Returns the previous scale in this volume, or None if this is the first."""
        my_index = self.volume.scales.index(self)
        print(f"Index of {self.key} is {my_index} of {len(self.volume.scales)}.")
        if my_index > 0:
            return self.volume.scales[my_index - 1]
        else:
            return None

    @property
    def affine(self):
        scaling = np.diag(np.r_[self.res_nm, 1.0])
        affine = np.dot(self.volume.transform_nm, scaling)
        affine[:3, :] /= 1e6
        return affine

    def _point_to_lower_chunk_idx(self, xyz):
        return (
            np.floor((np.array(xyz) - self.voxel_offset) / self.chunk_sizes)
            .astype("int")
            .ravel()
        )

    def _point_to_upper_chunk_idx(self, xyz):
        return (
            np.ceil((np.array(xyz) - self.voxel_offset) / self.chunk_sizes)
            .astype("int")
            .ravel()
        )

    def _read_chunk(self, gx, gy, gz):
        if self.volume.USE_CACHE:
            cachefile = cache.CACHE.build_filename(
                "{}_{}_{}_{}_{}".format(self.volume.url, self.key, gx, gy, gz),
                suffix=".npy",
            )
            if os.path.isfile(cachefile):
                return np.load(cachefile)

        x0 = gx * self.chunk_sizes[0]
        y0 = gy * self.chunk_sizes[1]
        z0 = gz * self.chunk_sizes[2]
        x1, y1, z1 = np.minimum(self.chunk_sizes + [x0, y0, z0], self.size)
        chunk_czyx = self.volume._io.read_chunk(self.key, (x0, x1, y0, y1, z0, z1))
        if not chunk_czyx.shape[0] == 1 and not self.color_warning_issued:
            logger.warn(
                "Color channel data is not yet supported. Returning first channel only."
            )
            self.color_warning_issued = True
        chunk_zyx = chunk_czyx[0]

        if self.volume.USE_CACHE:
            np.save(cachefile, chunk_zyx)
        return chunk_zyx

    def fetch(self, voi: boundingbox.BoundingBox = None, **kwargs):

        # define the bounding box in this scale's voxel space
        if voi is None:
            bbox_ = boundingbox.BoundingBox((0, 0, 0), self.size, space=None)
        else:
            bbox_ = voi.transform(np.linalg.inv(self.affine))

        for dim in range(3):
            if bbox_.shape[dim] < 1:
                logger.warn(
                    f"Bounding box in voxel space will be enlarged to voxel size 1 along axis {dim}."
                )
                bbox_.maxpoint[dim] = bbox_.maxpoint[dim] + 1

        # extract minimum and maximum the chunk indices to be loaded
        gx0, gy0, gz0 = self._point_to_lower_chunk_idx(tuple(bbox_.minpoint))
        gx1, gy1, gz1 = self._point_to_upper_chunk_idx(tuple(bbox_.maxpoint))

        # create requested data volume, and fill it with the required chunk data
        shape_zyx = np.array([gz1 - gz0, gy1 - gy0, gx1 - gx0]) * self.chunk_sizes[::-1]
        data_zyx = np.zeros(shape_zyx, dtype=self.volume.dtype)
        for gx in range(gx0, gx1):
            x0 = (gx - gx0) * self.chunk_sizes[0]
            for gy in range(gy0, gy1):
                y0 = (gy - gy0) * self.chunk_sizes[1]
                for gz in range(gz0, gz1):
                    z0 = (gz - gz0) * self.chunk_sizes[2]
                    chunk = self._read_chunk(gx, gy, gz)
                    z1, y1, x1 = np.array([z0, y0, x0]) + chunk.shape
                    data_zyx[z0:z1, y0:y1, x0:x1] = chunk

        # determine the remaining offset from the "chunk mosaic" to the
        # exact bounding box requested, to cut off undesired borders
        data_min = np.array([gx0, gy0, gz0]) * self.chunk_sizes
        x0, y0, z0 = (np.array(tuple(bbox_.minpoint)) - data_min).astype("int")
        xd, yd, zd = np.array(bbox_.shape).astype("int")
        offset = tuple(bbox_.minpoint)

        # build the nifti image
        trans = np.identity(4)[[2, 1, 0, 3], :]  # zyx -> xyz
        shift = np.c_[np.identity(4)[:, :3], np.r_[offset, 1]]
        return nib.Nifti1Image(
            data_zyx[z0: z0 + zd, y0: y0 + yd, x0: x0 + xd],
            np.dot(self.affine, np.dot(shift, trans)),
        )


class NeuroglancerMesh(volume.VolumeProvider, srctype="neuroglancer/precompmesh"):
    """
    A surface mesh provided as neuroglancer precomputed mesh.
    """
    def __init__(self, url, volume=None):
        self.volume = volume
        self.url = url
        self.meshinfo = requests.HttpRequest(url=self.url + "/info", func=requests.DECODERS['.json']).data
        self.mesh_key = self.meshinfo.get('mesh')

    @staticmethod
    def _fetch_fragment(url: str, transform_nm: np.ndarray):
        r = requests.HttpRequest(url, func=lambda b: BytesIO(b))
        (vertices_vox, triangles_vox) = read_precomputed_mesh(r.data)
        vertices, triangles = affine_transform_mesh(vertices_vox, triangles_vox, transform_nm)
        vertices /= 1e6
        return vertices, triangles

    def fetch(self, meshindex: int, resolution_mm: float = None, voi=None, hemisphere: str = "left", **kwargs):
        """
        Returns the list of fragment meshes found under the given mesh index.
        Each mesh is  a dictionary with the keys:
        - vertices: an Nx3 array of coordinates (in nanometer)
        - faces: an MX3 array containing connection data of vertices
        - name: name of the fragment
        """
        if resolution_mm is not None:
            logger.warn(f"{self.__class__}.fetch() ignores 'resolution_mm' argument")
        if voi is not None:
            logger.warn(f"{self.__class__}.fetch() ignores 'voi' argument")
        if voi:
            raise RuntimeError("Volume of interests cannot yet be fetched from neuroglancer meshes.")
        try:
            resp = requests.HttpRequest(
                url=f"{self.url}/{self.mesh_key}/{str(meshindex)}:0",
                func=requests.DECODERS['.json']
            ).data
            fragments = resp.get('fragments') if resp else None
        except requests.SiibraHttpRequestError as e:
            print(str(e))
            logger.error(f"Source {self} does not provide a mesh with index {meshindex}.")
            return None
        
        transform_nm = np.array(requests.HttpRequest(f"{self.url}/transform.json").data)

        if hemisphere is None:
            logger.warn("No hemisphere is selected. Returning both in one mesh.")
            hemisphere = "all"

        if (hemisphere.casefold() == "all") or (hemisphere.casefold() == "whole"):
            logger.warn("Currently, whole brain cannot be displayed.")
            name = "whole"
            mesh_fragment_left = self._fetch_fragment(f"{self.url}/{self.mesh_key}/{fragments[0]}", transform_nm)
            mesh_fragment_right = self._fetch_fragment(f"{self.url}/{self.mesh_key}/{fragments[1]}", transform_nm)

            vertices = np.concatenate((mesh_fragment_left[0], mesh_fragment_right[0]))
            faces = np.concatenate((mesh_fragment_left[1], mesh_fragment_right[1] + len(mesh_fragment_left[1])) )
        elif hemisphere.casefold() == "left":
            name = "left"
            (vertices, faces) = self._fetch_fragment(f"{self.url}/{self.mesh_key}/{fragments[0]}", transform_nm)
        elif hemisphere.casefold() == "right":
            name = "right"
            (vertices, faces) = self._fetch_fragment(f"{self.url}/{self.mesh_key}/{fragments[1]}", transform_nm)

        logger.warn("Labels are not yet implemented.")
        return dict(zip(['verts', 'faces', 'name'], [vertices, faces, name]))
        
        

    @property
    def variants(self):
        return list(self._loaders.keys()) # rewrite the code above to have _loaders instead

    def fetch_iter(self):
        return (self.fetch(v) for v in self.variants)
    
 
    def find_layer_thickness(self, meshindex_0: int = 0, meshindex_1: int = None):
        """
        Returns a dictionary with keys as the hemisphere and
        the value of the thickness of the given layers.
        """
        # TODO: implement cache check
        print(type(self))
        mesh_0 = self.fetch(meshindex=meshindex_0)
        if meshindex_1 is None:
            meshindex_1 = 7
            logger.warn(f"Second layer is not given. Automatically selecting non-cortical layer.")
        mesh_1 = self.fetch(meshindex=meshindex_1)

        return np.linalg.norm(mesh_0["verts"] - mesh_1["verts"], axis=1)
