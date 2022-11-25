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

from .mesh import GiftiSurface, NeuroglancerMesh


from .. import logger
from ..commons import MapType
from ..retrieval import HttpRequest, ZipfileRequest, CACHE, SiibraHttpRequestError
from ..core.location import BoundingBox, PointSet

import numpy as np
import nibabel as nib
import os
from abc import ABC, abstractmethod
from neuroglancer_scripts.precomputed_io import get_IO_for_existing_dataset
from neuroglancer_scripts.accessor import get_accessor_for_url


class ColorVolumeNotSupported(NotImplementedError):
    pass


class Volume:

    def __init__(self, name="", space_info: dict = {}, urls: dict = {}):
        self.name = name
        self.space_info = space_info
        self._providers = {}
        for srctype, url in urls.items():
            if srctype == "neuroglancer/precomputed":
                self._providers[srctype] = NeuroglancerVolume(url)
            elif srctype == "nii":
                self._providers[srctype] = NiftiVolume(url)
            elif srctype == "neuroglancer/precompmesh":
                self._providers[srctype] = NeuroglancerMesh(url)
            elif srctype == "gii-mesh":
                self._providers[srctype] = GiftiSurface(url)
            else:
                logger.warn(f"Volume source type {srctype} not yet supported, ignoring this specification.")


class ImageProvider(ABC):

    def __init__(self, url: str):
        self.url = url

    @abstractmethod
    def fetch(self, resolution_mm=None, voi=None):
        """
        Provide access to image data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement fetch(), use a derived class."
        )

    @abstractmethod
    def get_shape(self, resolution_mm=None):
        """
        Return the shape of the image volume.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_shape(), use a derived class."
        )

    @abstractmethod
    def is_float(self):
        """
        Return True if the data type of the volume is a float type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement is_float(), use a derived class."
        )

    def is_4D(self):
        return len(self.get_shape()) == 4


class NiftiVolume(ImageProvider):

    def __init__(self, url, zipped_file=None):
        ImageProvider.__init__(self, url)
        if os.path.isfile(self.url):
            if zipped_file is None:
                self._image_loader = lambda fn=self.url: nib.load(fn)
            else:
                raise NotImplementedError("loading niftis from zip containers not yet supported")
        else:
            if zipped_file is None:
                self._image_loader = lambda u=url: HttpRequest(u).data
            else:
                self._image_loader = lambda u=url: ZipfileRequest(u, zipped_file).data

    @property
    def image(self):
        return self._image_loader()

    def fetch(self, resolution_mm=None, voi=None):
        """
        Loads and returns a Nifti1Image object

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for neuroglancer volumes.
        voi : BoundingBox
            optional bounding box
        """
        img = None
        if resolution_mm is not None:
            raise NotImplementedError(
                f"NiftiVolume does not support to specify image resolutions (but {resolution_mm} was given)"
            )

        img = self.image
        bb_vox = None
        if voi is not None:
            bb_vox = voi.transform_bbox(np.linalg.inv(img.affine))

        if bb_vox is not None:
            (x0, y0, z0), (x1, y1, z1) = bb_vox.minpoint, bb_vox.maxpoint
            shift = np.identity(4)
            shift[:3, -1] = bb_vox.minpoint
            img = nib.Nifti1Image(
                dataobj=img.dataobj[x0:x1, y0:y1, z0:z1],
                affine=np.dot(img.affine, shift),
            )

        return img

    def get_shape(self, resolution_mm=None):
        if resolution_mm is not None:
            raise NotImplementedError(
                "NiftiVolume does not support to specify different image resolutions"
            )
        try:
            return self.image.shape
        except AttributeError as e:
            logger.error(
                f"Invalid object type {type(self.image)} of image for {self} {self.name}"
            )
            raise (e)

    def is_float(self):
        return self.image.dataobj.dtype.kind == "f"

    def find_peaks(self, min_distance_mm=5):
        """
        Find peaks in the image data.

        Arguments:
        ----------
        min_distance_mm : float
            Minimum distance between peaks in mm

        Returns:
        --------
        peaks: PointSet
        """

        from skimage.feature.peak import peak_local_max
        from ..commons import affine_scaling

        img = self.fetch()
        dist = int(min_distance_mm / affine_scaling(img.affine) + 0.5)
        voxels = peak_local_max(
            img.get_fdata(),
            exclude_border=False,
            min_distance=dist,
        )
        return (
            PointSet(
                [np.dot(img.affine, [x, y, z, 1])[:3] for x, y, z in voxels],
                space=self.space,
            ),
            img,
        )


class NeuroglancerVolume(ImageProvider):
    # Number of bytes at which an image array is considered to large to fetch
    MAX_GiB = 0.2

    # Wether to keep fetched data in local cache
    USE_CACHE = False

    @property
    def MAX_BYTES(self):
        return self.MAX_GiB * 1024 ** 3

    def __init__(self, url: str):
        ImageProvider.__init__(self, url=url)
        self._scales_cached = None
        self._io = None

    _transform_nm = None

    @property
    def transform_nm(self):
        if self._transform_nm is not None:
            return self._transform_nm
        transform_in_detail = self.detail.get("neuroglancer/precomputed", {}).get(
            "transform"
        )
        if transform_in_detail:
            logger.debug(
                "transform defined in detail attribute, using detail to set transform"
            )
            self._transform_nm = np.array(transform_in_detail)
            return self._transform_nm

        try:
            res = HttpRequest(f"{self.url}/transform.json").get()
        except SiibraHttpRequestError:
            res = None
        if res is not None:
            logger.debug(
                "Found global affine transform file, intrepreted in nanometer space."
            )
            self._transform_nm = np.array(res)
            return self._transform_nm

        self._transform_nm = np.identity(1)
        logger.debug("Fall back, using identity")
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

    def fetch(self, resolution_mm=None, voi: BoundingBox = None):
        if voi is not None:
            assert voi.space == self.space
        scale = self._select_scale(resolution_mm, voi)
        logger.debug(
            f"Fetching resolution "
            f"{', '.join(map('{:.2f}'.format, scale.res_mm))} mm "
        )
        return scale.fetch(voi)

    def get_shape(self, resolution_mm=None):
        scale = self._select_scale(resolution_mm)
        return scale.size

    def is_float(self):
        return self.dtype.kind == "f"

    def _select_scale(self, resolution_mm, bbox: BoundingBox = None):

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

        while scale._estimate_nbytes(bbox) > self.MAX_BYTES:
            scale = scale.next()
            if scale is None:
                raise RuntimeError(
                    f"Fetching bounding box {bbox} is infeasible "
                    f"relative to the limit of {self.MAX_BYTES/1024**3}GiB."
                )

        return scale


class NeuroglancerScale:
    """One scale of a NeuroglancerVolume."""

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

    @property
    def space(self):
        """forward the corresponding volume's coordinate space."""
        return self.volume.space

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

    def _estimate_nbytes(self, bbox: BoundingBox = None):
        """Estimate the size image array to be fetched in bytes, given a bounding box."""
        if bbox is None:
            bbox_ = BoundingBox((0, 0, 0), self.size, self.space)
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
            cachefile = CACHE.build_filename(
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
        if not chunk_czyx.shape[0] == 1:
            logger.warn(
                "Color channel data is not yet supported. Returning first channel only."
            )
        chunk_zyx = chunk_czyx[0]

        if self.volume.USE_CACHE:
            np.save(cachefile, chunk_zyx)
        return chunk_zyx

    def fetch(self, voi: BoundingBox = None):

        # define the bounding box in this scale's voxel space
        if voi is None:
            bbox_ = BoundingBox((0, 0, 0), self.size, self.space)
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
