# Copyright 2018-2025
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

from io import BytesIO
import os
from typing import Union, Dict, Tuple
import json

import numpy as np
import nibabel as nib
from neuroglancer_scripts.precomputed_io import get_IO_for_existing_dataset, PrecomputedIO
from neuroglancer_scripts.accessor import DataAccessError
from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.mesh import read_precomputed_mesh, affine_transform_mesh
from nilearn.image.resampling import BoundingBoxError

from . import provider as _provider
from ...retrieval import requests, cache
from ...locations import boundingbox as _boundingbox
from ...commons import (
    logger,
    MapType,
    merge_meshes,
    SIIBRA_MAX_FETCH_SIZE_BYTES,
    QUIET,
    resample_img_to_img
)


def shift_ng_transfrom(
    transform_nm: np.ndarray, scale_resolution_nm: np.ndarray, max_resolution_nm: np.ndarray
) -> np.ndarray:
    """
    Helper method to get nifti standard affine.

    transfrorm.json stored with neuroglancer precomputed images and meshes
    are meant to be used for neuroglancer viewers and hence they are not
    representative of the affine in other tools. This method shifts back
    half a voxel in each axis.
    (see https://neuroglancer-scripts.readthedocs.io/en/latest/neuroglancer-info.html#different-conventions-for-coordinate-transformations)

    Parameters
    ----------
    transform_nm: np.ndarray
        Transform array created for displaying an image correctly from
        neuroglancer precomputed format in neuroglancer viewer.
    max_resolution_nm: np.ndarray
        The voxel resolution of the highest level of resolution.

    Returns
    -------
    np.ndarray
        Standard affine in nm
    """
    scaling = np.diag(np.r_[scale_resolution_nm, 1.0])
    affine = np.dot(transform_nm, scaling)
    affine[:3, 3] += (max_resolution_nm * 0.5)
    return affine


class NeuroglancerProvider(_provider.VolumeProvider, srctype="neuroglancer/precomputed"):

    def __init__(self, url: Union[str, Dict[str, str]]):
        _provider.VolumeProvider.__init__(self)
        self._init_url = url
        # TODO duplicated code to giftimesh
        if isinstance(url, str):  # one single image to load
            self._fragments = {None: NeuroglancerVolume(url)}
        elif isinstance(url, dict):  # assuming multiple for fragment images
            self._fragments = {n: NeuroglancerVolume(u) for n, u in url.items()}
        else:
            raise ValueError(f"Invalid url specified for {self.__class__.__name__}: {url}")

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url

    def fetch(
        self,
        fragment: str = None,
        resolution_mm: float = -1,
        voi: _boundingbox.BoundingBox = None,
        max_bytes: float = SIIBRA_MAX_FETCH_SIZE_BYTES,
        **kwargs,
    ) -> nib.Nifti1Image:
        """
        Fetch 3D image data from neuroglancer volume.

        Parameters
        ----------
        fragment: str, optional
            The name of a fragment volume to fetch, if any. For example,
            some volumes are split into left and right hemisphere fragments.
            See :func:`~siibra.volumes.Volume.fragments`
        resolution_mm: float, default: -1 (i.e, the highest possible given max_bytes)
            Desired resolution in millimeters.
        voi: BoundingBox
            optional specification of a volume of interest to fetch.
        max_bytes: float: Default: NeuroglancerVolume.MAX_BYTES
            Maximum allowable size (in bytes) for downloading the image. siibra
            will attempt to find the highest resolution image with a size less
            than this value.
        """

        result = None

        if 'index' in kwargs:
            index = kwargs.pop('index')
            if fragment is not None:
                assert fragment == index.fragment
            fragment = index.fragment

        if len(self._fragments) > 1:
            if fragment is None:
                logger.info(
                    f"Merging fragments [{', '.join(self._fragments.keys())}]. "
                    f"You can select one using `fragment` kwarg."
                )
                result = self._merge_fragments(
                    resolution_mm=resolution_mm,
                    voi=voi,
                    max_bytes=max_bytes,
                    **kwargs
                )
            else:
                matched_names = [n for n in self._fragments if fragment.lower() in n.lower()]
                if len(matched_names) != 1:
                    raise ValueError(
                        f"Requested fragment '{fragment}' could not be matched uniquely "
                        f"to [{', '.join(self._fragments)}]"
                    )
                else:
                    result = self._fragments[matched_names[0]].fetch(
                        resolution_mm=resolution_mm, voi=voi, **kwargs
                    )
        else:
            assert len(self._fragments) > 0
            fragment_name, ngvol = next(iter(self._fragments.items()))
            if fragment is not None:
                assert fragment.lower() in fragment_name.lower()
            result = ngvol.fetch(
                resolution_mm=resolution_mm,
                voi=voi,
                max_bytes=max_bytes,
                **kwargs
            )

        # if a label is specified, mask the resulting image.
        if result is not None:
            if 'label' in kwargs:
                label = kwargs['label']
            elif ('index') in kwargs:
                label = kwargs['index'].label
            else:
                label = None
            if label is not None:
                result = nib.Nifti1Image(
                    (np.asanyarray(result.dataobj) == label).astype('uint8'),
                    result.affine,
                    dtype='uint8',
                )
            # result.set_qform(result.affine)  # TODO: needs to be investigated further

        return result

    def get_boundingbox(self, **fetch_kwargs) -> "_boundingbox.BoundingBox":
        """
        Return the bounding box in physical coordinates of the union of
        fragments in this neuroglancer volume.

        Parameters
        ----------
        fetch_kwargs:
            key word arguments that are used for fetching volumes,
            such as voi or resolution_mm.
        """
        bbox = None
        for frag in self._fragments.values():
            if len(frag.shape) > 3:
                logger.warning(
                    f"N-D Neuroglancer volume has shape {frag.shape}, but "
                    f"bounding box considers only {frag.shape[:3]}"
                )
            resolution_mm = fetch_kwargs.get("resolution_mm")
            if resolution_mm is None:
                affine = frag.affine
                shape = frag.shape[:3]
            else:
                scale = frag._select_scale(resolution_mm=resolution_mm)
                affine = scale.affine
                shape = scale.size[:3]
            next_bbox = _boundingbox.BoundingBox(
                (0, 0, 0), shape, space=None
            ).transform(affine)
            bbox = next_bbox if bbox is None else bbox.union(next_bbox)
        return bbox

    def _merge_fragments(
        self,
        resolution_mm: float = -1,
        voi: _boundingbox.BoundingBox = None,
        max_bytes: float = SIIBRA_MAX_FETCH_SIZE_BYTES,
    ) -> nib.Nifti1Image:
        with QUIET:
            if voi is not None:
                bbox = voi
            else:
                bbox = self.get_boundingbox(clip=False, background=0, resolution_mm=resolution_mm)

        num_conflicts = 0
        result = None
        for frag_name, frag_vol in self._fragments.items():
            frag_scale = frag_vol._select_scale(
                resolution_mm=resolution_mm,
                bbox=voi,
                max_bytes=max_bytes,
            )
            img = frag_scale.fetch(voi=voi)
            if img is None:
                logger.debug(f"Fragment {frag_name} did not provide content for fetching.")
                continue
            if result is None:
                # build the empty result image with its own affine and voxel space
                transl = np.identity(4)
                transl[:3, -1] = list(bbox.minpoint.transform(np.linalg.inv(img.affine)))
                result_affine = np.dot(img.affine, transl)  # adjust global bounding box offset to get global affine
                voxdims = np.asanyarray(np.ceil(
                    bbox.transform(np.linalg.inv(result_affine)).shape  # transform to the voxel space
                ), dtype="int")
                result_arr = np.zeros(voxdims, dtype=img.dataobj.dtype)
                result = nib.Nifti1Image(dataobj=result_arr, affine=result_affine)

            # resample to merge template and update it
            try:
                resampled_img = resample_img_to_img(source_img=img, target_img=result)
            except BoundingBoxError:
                logger.debug(f"Bounding box outside the fragment {frag_name}.")
                continue
            arr = np.asanyarray(resampled_img.dataobj)
            nonzero_voxels = arr != 0
            num_conflicts += np.count_nonzero(result_arr[nonzero_voxels])
            result_arr[nonzero_voxels] = arr[nonzero_voxels]

        if num_conflicts > 0:
            num_voxels = np.count_nonzero(result_arr)
            logger.warning(
                f"Merging fragments required to overwrite {num_conflicts} "
                f"conflicting voxels ({num_conflicts / num_voxels * 100.:2.3f}%)."
            )

        return result


class NeuroglancerVolume:

    USE_CACHE = False  # Whether to keep fetched data in local cache
    MAX_BYTES = SIIBRA_MAX_FETCH_SIZE_BYTES  # Number of bytes at which an image array is considered to large to fetch

    def __init__(self, url: str):
        assert isinstance(url, str)
        self.url = url
        self._scales_cached = None
        self._info = None
        self._transform_nm = None
        self._io: PrecomputedIO = None

    @property
    def transform_nm(self) -> np.ndarray:
        """
        This is the transformation matrix created to cater neuroglancer viewer
        for a neuroglancer precomputed images.
        """
        if self._transform_nm is not None:
            return self._transform_nm
        try:
            res = requests.HttpRequest(f"{self.url}/transform.json").get()
        except requests.SiibraHttpRequestError:
            res = None
        if res is not None:
            self._transform_nm = np.array(res)
            return self._transform_nm

        self._transform_nm = np.identity(4)
        logger.warning(f"No transform.json found at {self.url}, using identity.")
        return self._transform_nm

    @transform_nm.setter
    def transform_nm(self, val):
        self._transform_nm = val

    @property
    def io(self) -> PrecomputedIO:
        if self._io is None:
            accessor = HttpAccessor(self.url)
            self._io = get_IO_for_existing_dataset(accessor)
        return self._io

    @property
    def map_type(self):
        if self._info is None:
            self._bootstrap()
        return (
            MapType.LABELLED
            if self._info.get("type") == "segmentation"
            else MapType.STATISTICAL
        )

    @map_type.setter
    def map_type(self, val):
        if val is not None:
            logger.debug(
                "NeuroglancerVolume can determine its own maptype from self._info.get('type')"
            )

    def _bootstrap(self):
        self._info = requests.HttpRequest(f"{self.url}/info", func=lambda b: json.loads(b.decode())).get()
        self._scales_cached = sorted(
            [NeuroglancerScale(self, i) for i in self._info["scales"]]
        )

    @property
    def dtype(self):
        if self._info is None:
            self._bootstrap()
        return np.dtype(self._info["data_type"])

    @property
    def scales(self):
        if self._scales_cached is None:
            self._bootstrap()
        return self._scales_cached

    @property
    def shape(self):
        # return the shape of the scale 0 array
        return self.scales[0].size

    @property
    def affine(self):
        # return the affine matrix of the scale 0 data
        return self.scales[0].affine

    def fetch(
        self,
        resolution_mm: float = -1,
        voi: _boundingbox.BoundingBox = None,
        max_bytes: float = MAX_BYTES,
        **kwargs
    ):
        # the caller has to make sure voi is defined in the correct reference space
        scale = self._select_scale(resolution_mm=resolution_mm, bbox=voi, max_bytes=max_bytes)
        return scale.fetch(voi=voi, **kwargs)

    def get_shape(self, resolution_mm=None, max_bytes: float = MAX_BYTES):
        scale = self._select_scale(resolution_mm=resolution_mm, max_bytes=max_bytes)
        return scale.size

    def is_float(self):
        return self.dtype.kind == "f"

    def _select_scale(
        self,
        resolution_mm: float,
        max_bytes: float = MAX_BYTES,
        bbox: _boundingbox.BoundingBox = None
    ) -> 'NeuroglancerScale':
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
            xyz_res = ['{:.6f}'.format(r).rstrip('0') for r in scale.res_mm]
            if all(r.startswith(str(resolution_mm)) for r in xyz_res):
                logger.info(f"Closest resolution to requested is {', '.join(xyz_res)} mm.")
            else:
                logger.warning(
                    f"Requested resolution {resolution_mm} is not available. "
                    f"Falling back to the highest possible resolution of "
                    f"{', '.join(xyz_res)} mm."
                )

        scale_changed = False
        while scale._estimate_nbytes(bbox) > max_bytes:
            scale = scale.next()
            scale_changed = True
            if scale is None:
                raise RuntimeError(
                    f"Fetching bounding box {bbox} is infeasible "
                    f"relative to the limit of {max_bytes / 1024**3}GiB."
                )
        if scale_changed:
            logger.warning(
                f"Resolution was reduced to {scale.res_mm} to provide a "
                f"feasible volume size of {max_bytes / 1024**3} GiB. Set `max_bytes` to"
                f" fetch in the resolution requested."
            )
        return scale


class NeuroglancerScale:
    """One scale of a NeuroglancerVolume."""

    color_warning_issued = False

    def __init__(self, volume: NeuroglancerVolume, scaleinfo: dict):
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
        """Test whether the resolution of this scale is sufficient to provide the given resolution."""
        return all(r / 1e6 <= resolution_mm for r in self.res_nm)

    def __lt__(self, other):
        """Sort scales by resolution."""
        return all(self.res_nm[i] < other.res_nm[i] for i in range(3))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.__class__.__name__} {self.key}"

    def _estimate_nbytes(self, bbox: _boundingbox.BoundingBox = None):
        """Estimate the size image array to be fetched in bytes, given a bounding box."""
        if bbox is None:
            bbox_ = _boundingbox.BoundingBox((0, 0, 0), self.size, space=None)
        else:
            bbox_ = bbox.transform(np.linalg.inv(self.affine))
        result = self.volume.dtype.itemsize * bbox_.volume
        logger.debug(
            f"Approximate size for fetching resolution "
            f"({', '.join(map('{:.6f}'.format, self.res_mm))}) mm "
            f"is {result / 1024**3:.5f} GiB."
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
        affine_ = shift_ng_transfrom(
            transform_nm=self.volume.transform_nm,
            scale_resolution_nm=self.res_nm,
            max_resolution_nm=self.volume.scales[0].res_nm[0],
        )
        affine_[:3, :] /= 1e6
        return affine_

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

    def _read_chunk(self, gx, gy, gz, channel: int = None):
        if any(v < 0 for v in (gx, gy, gz)):
            raise DataAccessError('Negative tile index observed - you have likely requested fetch() with a voi specification ranging outside the actual data.')
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
        chunk_czyx = self.volume.io.read_chunk(self.key, (x0, x1, y0, y1, z0, z1))
        if channel is None:
            channel = 0
            if chunk_czyx.shape[0] > 1 and not self.color_warning_issued:
                logger.warning(
                    f"The volume has {chunk_czyx.shape[0]} color channels. "
                    "Returning the first channel now but you can specify one "
                    "with 'channel' keyword."
                )
                self.color_warning_issued = True
        elif channel + 1 > chunk_czyx.shape[0]:
            raise ValueError(f"There are only {chunk_czyx.shape[0]} color channels.")
        chunk_zyx = chunk_czyx[channel]

        if self.volume.USE_CACHE:
            np.save(cachefile, chunk_zyx)
        return chunk_zyx

    def fetch(self, voi: _boundingbox.BoundingBox = None, **kwargs):

        # define the bounding box in this scale's voxel space
        if voi is None:
            bbox_ = _boundingbox.BoundingBox((0, 0, 0), self.size, space=None)
        else:
            bbox_ = voi.transform(np.linalg.inv(self.affine), space=None)

        # extract minimum and maximum the chunk indices to be loaded
        gx0, gy0, gz0 = self._point_to_lower_chunk_idx(tuple(bbox_.minpoint))
        gx1, gy1, gz1 = self._point_to_upper_chunk_idx(tuple(bbox_.maxpoint))

        # create requested data volume, and fill it with the required chunk data
        shape_zyx = np.array([gz1 - gz0, gy1 - gy0, gx1 - gx0]) * self.chunk_sizes[::-1]
        data_zyx = None
        for gx in range(gx0, gx1):
            x0 = (gx - gx0) * self.chunk_sizes[0]
            for gy in range(gy0, gy1):
                y0 = (gy - gy0) * self.chunk_sizes[1]
                for gz in range(gz0, gz1):
                    try:
                        chunk = self._read_chunk(gx, gy, gz, kwargs.get("channel"))
                    except DataAccessError:
                        logger.debug(f"voi: {voi}", exc_info=1)
                        continue
                    if data_zyx is None:
                        data_zyx = np.zeros(shape_zyx, dtype=self.volume.dtype)
                    z0 = (gz - gz0) * self.chunk_sizes[2]
                    z1, y1, x1 = np.array([z0, y0, x0]) + chunk.shape
                    data_zyx[z0:z1, y0:y1, x0:x1] = chunk

        # no voxel values in voi
        if data_zyx is None:
            return None

        # determine the remaining offset from the "chunk mosaic" to the
        # exact bounding box requested, to cut off undesired borders
        data_min = np.array([gx0, gy0, gz0]) * self.chunk_sizes
        x0, y0, z0 = (np.array(bbox_.minpoint) - data_min).astype("int")
        xd, yd, zd = np.ceil(bbox_.maxpoint).astype(int) - np.floor(bbox_.minpoint).astype(int)
        offset = tuple(bbox_.minpoint)
        if voi is not None:
            logger.debug(
                f"Input: {voi.minpoint.coordinate}, {voi.maxpoint.coordinate}.\nVoxel space: {bbox_.minpoint.coordinate}, {bbox_.maxpoint.coordinate}"
            )

        # build the nifti image
        trans = np.identity(4)[[2, 1, 0, 3], :]  # zyx -> xyz
        shift = np.c_[np.identity(4)[:, :3], np.r_[offset, 1]]
        return nib.Nifti1Image(
            data_zyx[z0: z0 + zd, y0: y0 + yd, x0: x0 + xd],
            np.dot(self.affine, np.dot(shift, trans)),
        )


class NeuroglancerMesh(_provider.VolumeProvider, srctype="neuroglancer/precompmesh"):
    """
    A surface mesh provided as neuroglancer precomputed mesh.
    """

    @staticmethod
    def _fragmentinfo(url: str) -> Dict[str, Union[str, np.ndarray, Dict]]:
        """Prepare basic mesh fragment information from url."""
        return {
            "url": url,
            "transform_nm": np.array(requests.HttpRequest(f"{url}/transform.json").data),
            "info": requests.HttpRequest(url=f"{url}/info", func=requests.DECODERS['.json']).data
        }

    # TODO check resource typing?
    def __init__(self, resource: Union[str, dict], volume=None):
        self.volume = volume
        self._init_url = resource
        if isinstance(resource, str):
            self._meshes = {None: self._fragmentinfo(resource)}
        elif isinstance(resource, dict):
            self._meshes = {n: self._fragmentinfo(u) for n, u in resource.items()}
        else:
            raise ValueError(f"Resource specification not understood for {self.__class__.__name__}: {resource}")

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url

    def get_boundingbox(self, clip=False, background=0.0, **fetch_kwargs) -> '_boundingbox.BoundingBox':
        """
        Bounding box calculation is not yet implemented for meshes.
        """
        raise NotImplementedError(
            f"Bounding box access to {self.__class__.__name__} objects not yet implemented."
        )

    def _get_fragment_info(self, meshindex: int) -> Dict[str, Tuple[str, ]]:
        # extract available fragment urls with their names for the given mesh index
        result = {}

        for name, spec in self._meshes.items():
            mesh_key = spec.get('info', {}).get('mesh')
            meshurl = f"{spec['url']}/{mesh_key}/{str(meshindex)}:0"
            transform = spec.get('transform_nm')
            try:
                meshinfo = requests.HttpRequest(url=meshurl, func=requests.DECODERS['.json']).data
            except requests.SiibraHttpRequestError:
                continue
            fragment_names = meshinfo.get('fragments')

            if len(fragment_names) == 0:
                raise RuntimeError(f"No fragments found at {meshurl}")
            elif len(self._meshes) > 1:
                # multiple meshes were configured, so we expect only one fragment under each mesh url
                if len(fragment_names) > 1:
                    raise RuntimeError(
                        f"{self.__class__.__name__} was configured with multiple mesh fragments "
                        f"({', '.join(self._meshes.keys())}), but unexpectedly even more fragmentations "
                        f"were found at {spec['url']}."
                    )
                result[name] = (f"{spec['url']}/{mesh_key}/{fragment_names[0]}", transform)
            else:
                # only one mesh was configures, so we might still
                # see multiple fragments under the mesh url
                for fragment_name in fragment_names:
                    result[fragment_name] = (f"{spec['url']}/{mesh_key}/{fragment_name}", transform)

        return result

    def _fetch_fragment(self, url: str, transform_nm: np.ndarray):
        r = requests.HttpRequest(url, func=lambda b: BytesIO(b))
        (vertices_vox, triangles_vox) = read_precomputed_mesh(r.data)
        vertices, triangles = affine_transform_mesh(vertices_vox, triangles_vox, transform_nm)
        vertices /= 1e6
        return {'verts': vertices, 'faces': triangles}

    def fetch(self, label: int, fragment: str):
        """
        Fetches a particular mesh. Each mesh is a dictionary with keys:

        Parameters
        ----------
        label: int
            Label of the volume
        fragment: str, default: None
            A fragment name can be specified to choose from multiple fragments.

            Note
            ----
            If not specified, multiple fragments will be merged into one mesh.
            In such a case, the verts and faces arrays of different fragments
            are appended to one another.
        Returns
        -------
        dict
            - 'verts': An Nx3 array of vertex coordinates (in nanometer)
            - 'faces': an MX3 array containing connection data of vertices
            - 'name': Name of the of the mesh variant
        """

        # extract fragment information for the requested mesh
        fragment_infos = self._get_fragment_info(label)

        if fragment is None:

            # no fragment specified, return merged fragment meshes
            if len(fragment_infos) == 1:
                url, transform = next(iter(fragment_infos.values()))
                return self._fetch_fragment(url, transform)
            else:
                logger.info(
                    f"Fragments [{', '.join(fragment_infos.keys())}] are merged during fetch(). "
                    "You can select one of them using the 'fragment' parameter."
                )
                return merge_meshes([self._fetch_fragment(u, t) for u, t in fragment_infos.values()])

        else:

            # match fragment to available fragments
            matched = [
                info for name, info in fragment_infos.items()
                if fragment.lower() in name
            ]
            if len(matched) == 1:
                url, transform = next(iter(matched))
                return self._fetch_fragment(url, transform)
            else:
                raise ValueError(
                    f"The requested mesh fragment name '{fragment}' could not be resolved. "
                    f"Valid names are: {', '.join(fragment_infos.keys())}"
                )


class NeuroglancerSurfaceMesh(NeuroglancerMesh, srctype="neuroglancer/precompmesh/surface"):
    """
    Only shadows NeuroglancerMesh for the special surface srctype,
    which provides a mesh urls plus a mesh index for identifying the surface.
    Behaves like NeuroglancerMesh otherwise.

    TODO this class might be replaced by implementing a default label index for the parent class.
    """
    def __init__(self, spec: str, **kwargs):
        # Here we expect a string of the form "<url> <labelindex>",
        # and use this to set the url and label index in the parent class.
        assert ' ' in spec
        url, labelindex, *args = spec.split(' ')
        assert labelindex.isnumeric()
        self.label = int(labelindex)
        NeuroglancerMesh.__init__(self, resource=url, **kwargs)

    @property
    def fragments(self, meshindex=1):
        """
        Returns the set of fragment names available
        for the mesh with the given index.
        """
        return set(self._get_fragment_info(self.label))

    def fetch(self, **kwargs):
        if 'fragment' not in kwargs:
            kwargs['fragment'] = None
        return super().fetch(label=self.label, **kwargs)
