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

from typing import ClassVar, List

from ..openminds.common import CommonConfig
from .. import logger
from ..commons import MapType
from ..core.datasets import Dataset
from ..core.space import Space, BoundingBox
from ..core.concept import main_openminds_registry
from ..retrieval import HttpRequest, ZipfileRequest, CACHE, SiibraHttpRequestError

import numpy as np
import nibabel as nib
import os
from abc import ABC, abstractmethod, abstractproperty
from neuroglancer_scripts.precomputed_io import get_IO_for_existing_dataset
from neuroglancer_scripts.accessor import get_accessor_for_url


from ..openminds.core.v4.data import file, contentType

class ContentType(contentType.Model):
    ...

ng_volume = ContentType(**{
    "https://openminds.ebrains.eu/vocab/name": "application/vnd.neuroglancer.precomputed",
    "@id": "https://openminds.ebrains.eu/core/ContentType/neuroglancer.precomputed",
    "@type": "https://openminds.ebrains.eu/core/ContentType",
})

ng_mesh = ContentType(**{
    "https://openminds.ebrains.eu/vocab/name":'application/vnd.neuroglancer.precompmesh',
    "@id":'https://openminds.ebrains.eu/core/ContentType/neuroglancer.precompmesh',
    "@type":'https://openminds.ebrains.eu/core/ContentType',
})

fall_back = ContentType(**{
    "https://openminds.ebrains.eu/vocab/name":'application/unknown',
    "@id":'https://openminds.ebrains.eu/core/ContentType/unknown',
    "@type":'https://openminds.ebrains.eu/core/ContentType',
})

# existing
gii = ContentType(**{
    "https://openminds.ebrains.eu/vocab/name": 'application/vnd.gifti',
    "@id": 'https://openminds.ebrains.eu/instances/contentTypes/application/vnd.gifti',
    "@type": 'https://openminds.ebrains.eu/core/ContentType',
})
nii = ContentType(**{
    "https://openminds.ebrains.eu/vocab/name": 'application/vnd.nifti.1',
    "@id": 'https://openminds.ebrains.eu/instances/contentTypes/application/vnd.nifti.1',
    "@type": 'https://openminds.ebrains.eu/core/ContentType',
})

class File(file.Model):
    @classmethod
    def parse_legacy(Cls, json_input) -> List['File']:
        assert json_input.get('@type') == 'fzj/tmp/volume_type/v0.0.1'

        is_mesh = False
        is_volume = False
        content_type = fall_back

        volume_type = json_input.get('volume_type')
        
        if volume_type == 'nii':
            is_volume = True
            content_type = nii

        if volume_type == 'neuroglancer/precomputed':
            is_volume = True
            content_type = ng_volume
        if volume_type == 'neuroglancer/precompmesh':
            is_mesh = True
            content_type = ng_mesh
        if volume_type == 'gii' or volume_type == 'gii' or volume_type == 'gii-label':
            is_mesh = True
            content_type = gii
        
        content_description = None
        file_repository = None
        hash = None
        is_part_of = None
        special_usage_role = None
        storage_size = None

        base_id = os.path.basename(json_input.get('@id'))
        url_val = json_input.get('url')
        if type(url_val) == str:
            urls = [ (None, url_val) ]
        elif type(url_val) == dict:
            urls = [(key, val) for key, val in url_val.items()]
            assert all([type(url) == str for _, url in urls])
        elif url_val is None:
            # TODO some dataset has None as url attr
            # e.g. big brain collect volume src
            return []
        else:
            raise ValueError(f'cannot parse type of url: {type(url_val)}')

        # expect the mesh to be either volume or mesh
        # as this will populate the data_type which has a min 1 requirement
        assert is_volume or is_mesh, f'Expecting the volume to be either volume or mesh, but is neither: {json_input}'
        
        # TODO does not parse threesurfer gii files at all
        # use new schema (0.3a5)
        return [Cls(
            name=json_input.get('name', 'Unnamed file'),
            id=f"https://openminds.ebrains.eu/core/File/{base_id}{'/' + key if key else ''}",
            type="https://openminds.ebrains.eu/core/File",
            iri=url,
            data_type=[
                *(['https://openminds.ebrains.eu/instances/dataType/voxelData'] if is_volume else []),
                *(['https://openminds.ebrains.eu/instances/dataType/3DComputerGraphic'] if is_mesh else [])
            ],
            format={
                '@id': content_type and content_type.id or 'UnknownContentType'
            },
        ) for key, url in urls]

    Config = CommonConfig


gbyte_feasible = 0.1


class VolumeSrc(File):

    _SPECIALISTS: ClassVar[dict] = {}
    _space_id = None
    _volume_type = None
    _map_type = None
    _detail = None
    _legacy_json = None

    _SPECIALISTS = {}
    _SURFACE_TYPES = ['gii', 'neuroglancer/precompmesh']

    def __init__(self, space=None, detail=None, **data):
        """
        Construct a new volume source.

        Parameters
        ----------
        identifier : str
            A unique identifier for the source
        name : str
            A human-readable name
        volume_type : str
            Type of volume source, clarifying the data format. Typical names: "nii", "neuroglancer/precomputed".
        url : str
            The URL to the volume src, typically a url to the corresponding image or tilesource.
        space : Space
            Reference space in which this volume is defined
        detail : dict
            Detailed information. Currently only used to store a transformation matrix  for neuroglancer tilesources.
        zipped_file : str
            The filename to be extracted from a zip file. If given, the url is
            expected to point to a downloadable zip archive. Currently used to
            extreact niftis from zip archives, as for example in case of the
            MNI reference templates.
        """
        File.__init__(self, **data)
        
        self._detail = {} if detail is None else detail

    def __init_subclass__(cls, volume_type=None):
        """Called when this class gets subclassed by cls."""
        cls._volume_type = volume_type
        if volume_type is not None:
            assert volume_type not in VolumeSrc._SPECIALISTS.keys()
            VolumeSrc._SPECIALISTS[volume_type] = cls

    def __str__(self):
        return f"{self._volume_type} {self.iri}"

    @property
    def is_volume(self):
        """Overwrite Dataset's default 'false'"""
        return True

    @property
    def is_surface(self):
        """Overwrite Dataset's default 'false'"""
        return self._volume_type in self._SURFACE_TYPES

    @property
    def space(self) -> Space:
        return main_openminds_registry[self._space_id]

    @classmethod
    def parse_legacy(Cls, json_input) -> List['VolumeSrc']:
        """
        Provides an object hook for the json library to construct a VolumeSrc
        object from a json stream.
        """
        if json_input.get("@type") != "fzj/tmp/volume_type/v0.0.1":
            raise NotImplementedError(
                f"Cannot build VolumeSrc from this json spec: {json_input}"
            )

        volume_type = json_input.get("volume_type")

        # decide if object shoulc be generated with a specialized derived class
        VolumeClass = Cls._SPECIALISTS.get(volume_type, VolumeSrc)


        # if special class exist, return
        if VolumeClass is not Cls:
            return VolumeClass.parse_legacy(json_input)
            
            
        result = super().parse_legacy(json_input)
            
        for r in result:
            r._legacy_json = json_input
            r._volume_type = volume_type
            if json_input.get('space_id'):
                r._space_id = Space.parse_legacy_id(json_input.get('space_id'))

        if VolumeClass is VolumeSrc:
            logger.error(f"Volume will be generated as plain VolumeSrc: {json_input}")
            return result
        
        maptype = json_input.get("map_type", None)
        if maptype is not None:
            for r in result:
                r._map_type = MapType[maptype.upper()]
        return result


class ImageProvider(ABC):
    @abstractmethod
    def fetch(self, resolution_mm=None, voi=None, mapindex=None):
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

    @abstractproperty
    def is_float(self) -> bool:
        """
        Return True if the data type of the volume is a float type.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement is_float(), use a derived class."
        )

    @property
    def is_4D(self) -> bool:
        return len(self.get_shape()) == 4


class LocalNiftiVolume(ImageProvider):

    volume_type = "nii"

    def __init__(self, name: str, img: nib.Nifti1Image, space):
        """ Create a new local nifti volume from a Nifti1Image object.

        Args:
            name ([str]): A human-readable name
            img (Nifti1Image): 3D image
            space (Space): the reference space associated with the Image
        """
        self.name = name
        if isinstance(img, nib.Nifti1Image):
            self.image = img
        elif isinstance(img, str):
            self.image = nib.load(img)
        else:
            raise ValueError(
                f"Cannot create local nifti volume from image parameter {img}"
            )

        if np.isnan(self.image.get_fdata()).any():
            logger.warn(f"Replacing NaN by 0 for {self.name}")
            self.image = nib.Nifti1Image(
                np.nan_to_num(self.image.get_fdata()), self.image.affine
            )
        self.space = space

    def fetch(self, **kwargs):
        return self.image

    def get_shape(self):
        return self.image.shape

    @property
    def is_float(self):
        return self.image.dataobj.dtype.kind == "f"


class RemoteNiftiVolume(ImageProvider, VolumeSrc, volume_type="nii"):

    _image_cached = None
    _image_loader = None
    def __init__(
        self, **data
    ):
        VolumeSrc.__init__(self, **data)
        zipped_file=data.get('zipped_file')
        if zipped_file is None:
            self._image_loader = HttpRequest(self.iri)
        else:
            self._image_loader = ZipfileRequest(self.iri, zipped_file)

    @property
    def image(self):
        return self._image_loader.data

    def fetch(self, resolution_mm=None, voi=None, mapindex=None):
        """
        Loads and returns a Nifti1Image object representing the volume source.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for neuroglancer volumes.
        voi : BoundingBox
            optional bounding box
        """
        shape = self.get_shape()
        img = None
        if resolution_mm is not None:
            raise NotImplementedError(
                f"NiftiVolume does not support to specify image resolutions (but {resolution_mm} was given)"
            )

        if mapindex is None:
            img = self.image
        elif len(shape) != 4 or mapindex >= shape[3]:
            raise IndexError(
                f"Mapindex of {mapindex} provided for fetching from NiftiVolume, but its shape is {shape}."
            )
        else:
            img = nib.Nifti1Image(
                dataobj=self.image.dataobj[:, :, :, mapindex], affine=self.image.affine
            )
        assert img is not None

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

    @property
    def is_float(self):
        return self.image.dataobj.dtype.kind == "f"


class NeuroglancerVolume(
    ImageProvider, VolumeSrc, volume_type="neuroglancer/precomputed"
):
    # Number of bytes at which an image array is considered to large to fetch
    _MAX_GiB = .2
    _transform_nm = None
    _scales_cached = None
    _io = None

    @property
    def MAX_BYTES(self):
        return self._MAX_GiB * 1024**3
    
    def __init__(self, **kwargs):

        VolumeSrc.__init__(self, **kwargs)
        ImageProvider.__init__(self)
        self._scales_cached = None
        self._io = None
        self._transform_nm = np.identity(4)

    def _bootstrap(self):
        accessor = get_accessor_for_url(self.iri)
        self._io = get_IO_for_existing_dataset(accessor)
        self._scales_cached = sorted(
            [NeuroglancerScale(self, i) for i in self._io.info["scales"]]
        )

        try:
            res = HttpRequest(f"{self.iri}/transform.json").get()
        except SiibraHttpRequestError:
            res = None
        if res is not None:
            logger.debug(
                "Found global affine transform file, intrepreted in nanometer space."
            )
            self._transform_nm = np.array(res)

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

    @property
    def is_float(self):
        return self.dtype.kind == "f"

    def _select_scale(self, resolution_mm, bbox: BoundingBox = None):

        if resolution_mm is None:
            suitable = self.scales
        elif resolution_mm < 0:
            suitable = [self.scales[0]]
        else:
            suitable = sorted(s for s in self.scales if s.resolves(resolution_mm))
        if len(suitable)>0:
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


class GiftiSurfaceLabeling(VolumeSrc, volume_type="threesurfer/gii-label"):
    """
    TODO Implement this, surfaces need special handling
    """

    warning_shown = False

    def __init__(self, **data):
        if not self.__class__.warning_shown:
            logger.info(
                f"A {self.__class__.__name__} object was registered, "
                "but this type is not yet explicitly supported."
            )
            self.__class__.warning_shown = True
        VolumeSrc.__init__(self, **data)

    @property
    def is_image_volume(self):
        """ Meshes are not volumes. """
        return False

        
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
            # TODO 02/004 need space to be defined
            bbox_ = bbox.transform(np.linalg.inv(self.affine))
        result = self.volume.dtype.itemsize * bbox_.volume
        logger.debug(
            f"Approximate size for fetching resolution "
            f"({', '.join(map('{:.2f}'.format, self.res_mm))}) mm "
            f"is {result/1024**3:.2f} GiB."
        )
        return result

    def next(self):
        """ Returns the next scale in this volume, of None if this is the last.
        """
        my_index = self.volume.scales.index(self)
        print(f"Index of {self.key} is {my_index} of {len(self.volume.scales)}.")
        if my_index < len(self.volume.scales):
            return self.volume.scales[my_index+1]
        else:
            return None

    def prev(self):
        """ Returns the previous scale in this volume, or None if this is the first. 
        """
        my_index = self.volume.scales.index(self)
        print(f"Index of {self.key} is {my_index} of {len(self.volume.scales)}.")
        if my_index > 0:
            return self.volume.scales[my_index-1]
        else:
            return None

    @property
    def affine(self):
        scaling = np.diag(np.r_[self.res_nm, 1.])
        affine = np.dot(self.volume._transform_nm, scaling)
        affine[:3, :] /= 1e6
        return affine

    def _chunk_of_point(self, xyz):
        return (
            np.floor((np.array(xyz) - self.voxel_offset) / self.chunk_sizes)
            .astype("int")
            .ravel()
        )

    def _read_chunk(self, gx, gy, gz):
        cachefile = CACHE.build_filename(
            "{}_{}_{}_{}_{}".format(self.volume.iri, self.key, gx, gy, gz),
            suffix='.npy'
        )
        if os.path.isfile(cachefile):
            return np.load(cachefile)

        x0 = gx * self.chunk_sizes[0]
        y0 = gy * self.chunk_sizes[1]
        z0 = gz * self.chunk_sizes[2]
        x1, y1, z1 = np.minimum(self.chunk_sizes + [x0, y0, z0], self.size)
        chunk_czyx = self.volume._io.read_chunk(self.key, (x0, x1, y0, y1, z0, z1))
        if not chunk_czyx.shape[0] == 1:
            raise NotImplementedError("Color channel data is not yet supported")
        chunk_zyx = chunk_czyx[0]
        np.save(cachefile, chunk_zyx)
        return chunk_zyx

    def fetch(self, voi: BoundingBox = None):

        # define the bounding box in this scale's voxel space
        if voi is None:
            bbox_ = BoundingBox((0, 0, 0), self.size, self.space)
        else:
            bbox_ = voi.transform(np.linalg.inv(self.affine))

        # extract minimum and maximum the chunk indices to be loaded
        gx0, gy0, gz0 = self._chunk_of_point(tuple(bbox_.minpoint))
        gx1, gy1, gz1 = self._chunk_of_point(tuple(bbox_.maxpoint)) + 1

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
