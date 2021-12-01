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

from .. import logger
from ..commons import MapType
from ..core.datasets import Dataset
from ..core.space import Space, BoundingBox
from ..retrieval import HttpRequest, ZipfileRequest, CACHE

from ctypes import ArgumentError
import numpy as np
import nibabel
from cloudvolume.exceptions import OutOfBoundsError
from cloudvolume import CloudVolume
import os
import json
from abc import ABC, abstractmethod

gbyte_feasible = 0.1


class VolumeSrc(Dataset, type_id="fzj/tmp/volume_type/v0.0.1"):

    _SPECIALISTS = {}
    volume_type = None

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
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
        Dataset.__init__(self, identifier=identifier)
        assert name is not None
        self.name = name
        self.url = url
        if "SIIBRA_URL_MODS" in os.environ and url:
            mods = json.loads(os.environ["SIIBRA_URL_MODS"])
            for old, new in mods.items():
                self.url = self.url.replace(old, new)
            if self.url != url:
                logger.warning(f"Applied URL modification\nfrom {url}\nto   {self.url}")
        self.detail = {} if detail is None else detail
        self.space = space
        self.map_type = None

    def __init_subclass__(cls, volume_type=None):
        """Called when this class gets subclassed by cls."""
        cls.volume_type = volume_type
        if volume_type is not None:
            assert volume_type not in VolumeSrc._SPECIALISTS
            VolumeSrc._SPECIALISTS[volume_type] = cls

    def __str__(self):
        return f"{self.volume_type} {self.url}"

    def get_url(self):
        return self.url

    @property
    def is_image_volume(self):
        return True

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a VolumeSrc
        object from a json stream.
        """
        if obj.get("@type") != "fzj/tmp/volume_type/v0.0.1":
            raise NotImplementedError(
                f"Cannot build VolumeSrc from this json spec: {obj}"
            )

        volume_type = obj.get("volume_type")
        detail = obj.get("detail")
        url = obj.get("url")
        space = Space.REGISTRY[obj.get("space_id")]
        transform_nm = np.identity(4)
        if detail is not None and "neuroglancer/precomputed" in detail:
            if "transform" in detail["neuroglancer/precomputed"]:
                transform_nm = np.array(detail["neuroglancer/precomputed"]["transform"])

        # decide if object shoulc be generated with a specialized derived class
        VolumeClass = cls._SPECIALISTS.get(volume_type, cls)
        kwargs = {
            "transform_nm": transform_nm,
            "zipped_file": obj.get("zipped_file", None),
        }
        if VolumeClass == cls:
            logger.error(f"Volume will be generated as plain VolumeSrc: {obj}")
        result = VolumeClass(
            identifier=obj["@id"],
            name=obj["name"],
            url=url,
            space=space,
            detail=detail,
            **kwargs,
        )
        # result.volume_type = obj.get('volume_type',None)
        maptype = obj.get("map_type", None)
        if maptype is not None:
            result.map_type = MapType[maptype.upper()]
        return result


class ImageProvider(ABC):
    @abstractmethod
    def fetch(self, resolution_mm=None, voi=None, mapindex=None, clip=False):
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


class LocalNiftiVolume(ImageProvider):

    volume_type = 'nii'

    def __init__(self, name: str, img: nibabel.Nifti1Image, space: Space):
        """ Create a new local nifti volume from a Nifti1Image object.

        Args:
            name ([str]): A human-readable name
            img (Nifti1Image): 3D image
            space (Space): the reference space associated with the Image
        """
        self.name = name
        if isinstance(img, nibabel.Nifti1Image):
            self.image = img
        elif isinstance(img, str): 
            self.image = nibabel.load(img)
        else:
            raise ValueError(f"Cannot create local nifti volume from image parameter {img}") 

        if np.isnan(self.image.get_fdata()).any():
            logger.warn(f'Replacing NaN by 0 for {self.name}')
            self.image = nibabel.Nifti1Image(
                np.nan_to_num(self.image.get_fdata()),
                self.image.affine
            )
        self.space = space

    def fetch(self, **kwargs):
        return self.image

    def get_shape(self):
        return self.image.shape

    def is_float(self):
        return self.image.dataobj.dtype.kind == "f"

    @property
    def is_image_volume(self):
        return True


class RemoteNiftiVolume(ImageProvider, VolumeSrc, volume_type="nii"):

    _image_cached = None

    def __init__(
        self, identifier, name, url, space, detail=None, zipped_file=None, **kwargs
    ):
        VolumeSrc.__init__(self, identifier, name, url, space, detail=detail)
        if zipped_file is None:
            self._image_loader = HttpRequest(url)
        else:
            self._image_loader = ZipfileRequest(url, zipped_file)

    @property
    def image(self):
        return self._image_loader.data

    def fetch(self, resolution_mm=None, voi=None, mapindex=None, clip=False):
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
        clip : Boolean, default: False
            if True, generates an object where the image data array is cropped to its bounding box (with properly adjusted affine matrix)
        """
        if clip and voi:
            raise ArgumentError("voi and clip cannot only be requested independently")
        shape = self.get_shape()
        img = None
        if resolution_mm is not None:
            raise NotImplementedError(
                f"NiftiVolume does not support to specify image resolutions (but {resolution_mm} was given)"
            )

        if mapindex is None:
            img = self.image
        elif len(shape) != 4 or mapindex >= shape[3]:
            raise IndexError(
                f"Mapindex of {mapindex} provided for fetching from NiftiVolume, but its shape is {shape}."
            )
        else:
            img = nibabel.Nifti1Image(
                dataobj=self.image.dataobj[:, :, :, mapindex], affine=self.image.affine
            )
        assert img is not None

        bb_vox = None
        if voi is not None:
            bb_vox = voi.transform_bbox(np.linalg.inv(img.affine))
        elif clip:
            # determine bounding box by cropping the nonzero values
            bb_vox = BoundingBox.from_image(img)

        if bb_vox is not None:
            (x0, y0, z0), (x1, y1, z1) = bb_vox.minpoint, bb_vox.maxpoint
            shift = np.identity(4)
            shift[:3, -1] = bb_vox.minpoint
            img = nibabel.Nifti1Image(
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


class NeuroglancerVolume(
    ImageProvider, VolumeSrc, volume_type="neuroglancer/precomputed"
):

    # Gigabyte size that is considered feasible for ad-hoc downloads of
    # neuroglancer volume data. This is used to avoid accidental huge downloads.
    _cached_volume = None

    def __init__(
        self,
        identifier,
        name,
        url,
        space,
        detail=None,
        transform_nm=np.identity(4),
        **kwargs,
    ):
        """
        ngsite: base url of neuroglancer http location
        transform_nm: optional transform to be applied after scaling voxels to nm
        """
        super().__init__(identifier, name, url, space, detail)
        self.transform_nm = transform_nm
        self.info = HttpRequest(url + "/info", lambda b: json.loads(b.decode())).get()
        self.nbytes = np.dtype(self.info["data_type"]).itemsize
        self.num_scales = len(self.info["scales"])
        self.mip_resolution_mm = {
            i: np.min(v["resolution"]) / (1000 ** 2)
            for i, v in enumerate(self.info["scales"])
        }
        self.resolutions_available = {
            np.min(v["resolution"])
            / (1000 ** 2): {
                "mip": i,
                "GBytes": np.prod(v["size"]) * self.nbytes / (1024 ** 3),
            }
            for i, v in enumerate(self.info["scales"])
        }
        self.helptext = "\n".join(
            [
                "{:7.4f} mm {:10.4f} GByte".format(k, v["GBytes"])
                for k, v in self.resolutions_available.items()
            ]
        )

    @property
    def volume(self):
        """
        We implement this as a property so that the CloudVolume constructor is only called lazily.
        """
        if not self._cached_volume:
            self._cached_volume = CloudVolume(
                self.url, fill_missing=True, progress=False
            )
        return self._cached_volume

    def largest_feasible_resolution(self, voi=None):
        # returns the highest resolution in millimeter that is available and
        # still below the threshold of downloadable volume sizes.
        if voi is None:
            return min(
                [
                    res
                    for res, v in self.resolutions_available.items()
                    if v["GBytes"] < gbyte_feasible
                ]
            )
        else:
            gbytes = {
                res_mm: voi.transform(
                    np.linalg.inv(self.build_affine(res_mm))
                ).volume
                * self.nbytes
                / (1024 ** 3)
                for res_mm in self.resolutions_available.keys()
            }
            return min([res for res, gb in gbytes.items() if gb < gbyte_feasible])

    def _resolution_to_mip(self, resolution_mm, voi):
        """
        Given a resolution in millimeter, try to determine the mip that can
        be applied.

        Parameters
        ----------
        resolution_mm : float or None
            Physical resolution in mm.
            If None, the smallest availalbe resolution is used (lowest image size)
            If -1, tha largest feasible resolution is used.
        """
        mip = None
        if resolution_mm is None:
            mip = self.num_scales - 1
        elif resolution_mm == -1:
            maxres = self.largest_feasible_resolution(voi=voi)
            mip = self.resolutions_available[maxres]["mip"]
            if mip > 0:
                logger.info(
                    f"Due to the size of the volume requested, "
                    f"a reduced resolution of {maxres}mm is used "
                    f"(full resolution: {self.mip_resolution_mm[0]}mm).")
        elif resolution_mm in self.resolutions_available:
            mip = self.resolutions_available[resolution_mm]["mip"]
        if mip is None:
            raise ValueError(
                f"Requested resolution of {resolution_mm} mm not available.\n{self.helptext}"
            )
        return mip

    def build_affine(self, resolution_mm=None, voi=None):
        """
        Builds the affine matrix that maps voxels
        at the given resolution to physical space.

        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm.
            If None, the smallest is used.
            If -1, the largest feasible is used.
        """
        loglevel = logger.getEffectiveLevel()
        logger.setLevel("ERROR")
        mip = self._resolution_to_mip(resolution_mm=resolution_mm, voi=voi)
        effective_res_mm = self.mip_resolution_mm[mip]
        logger.setLevel(loglevel)

        # if a volume of interest is given, apply the offset
        shift = np.identity(4)
        if voi is not None:
            minpoint_vox = voi.minpoint.transform(
                np.linalg.inv(self.build_affine(effective_res_mm)))
            logger.debug(f"Affine matrix respects volume of interest shift {voi.minpoint}")
            shift[:3, -1] = minpoint_vox.coordinate

        # scaling from voxel to nm
        resolution_nm = self.info["scales"][mip]["resolution"]
        scaling = np.identity(4)
        for i in range(3):
            scaling[i, i] = resolution_nm[i]

        # optional transform in nanometer space
        affine = np.dot(self.transform_nm, scaling)

        # warp from nm to mm
        affine[:3, :] /= 1000000.0

        return np.dot(affine, shift)

    def _load_data(self, resolution_mm, voi: BoundingBox):
        """
        Actually load image data.
        TODO: Check amount of data beforehand and raise an Exception if it is over a reasonable threshold.
        NOTE: this function caches chunks as numpy arrays (*.npy) to the
        CACHEDIR defined in the retrieval module.

        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm. If none, the full resolution is used.
        """
        mip = self._resolution_to_mip(resolution_mm, voi=voi)
        effective_res_mm = self.mip_resolution_mm[mip]
        logger.debug(
            f"Loading neuroglancer data at a resolution of {effective_res_mm} mm (mip={mip})"
        )

        maxdims = tuple(np.array(self.volume.mip_shape(mip)[:3]) - 1)
        if voi is None:
            bbox_vox = BoundingBox([0, 0, 0], maxdims, space=None)
        else:
            bbox_vox = voi.transform(
                np.linalg.inv(self.build_affine(effective_res_mm)),
            ).clip(maxdims)

        if bbox_vox is None:
            # zero size bounding box, return empty array
            return np.empty((0, 0, 0))

        # estimate size and check feasibility
        gbytes = bbox_vox._Bbox.volume() * self.nbytes / (1024 ** 3)
        if gbytes > gbyte_feasible:
            # TODO would better do an estimate of the acutal data size
            logger.error(
                "Data request is too large (would result in an ~{:.2f} GByte download, the limit is {}).".format(
                    gbytes, gbyte_feasible
                )
            )
            print(self.helptext)
            raise NotImplementedError(
                f"Request of the whole full-resolution volume in one piece is prohibited as of now due to the estimated size of ~{gbytes:.0f} GByte."
            )

        # ok, retrieve data now.
        cachefile = CACHE.build_filename(
            f"{self.url}{bbox_vox._Bbox.serialize()}{str(mip)}", suffix="npy"
        )
        if os.path.exists(cachefile):
            logger.debug(f"NgVolume loads from cache file {cachefile}")
            return np.load(cachefile)
        else:
            try:
                logger.debug(f"NgVolume downloads (mip={mip}, bbox={bbox_vox}")
                data = self.volume.download(bbox=bbox_vox._Bbox, mip=mip)
                np.save(cachefile, np.array(data))
                return np.array(data)
            except OutOfBoundsError as e:
                logger.error("Bounding box does not match image.")
                print(str(e))
                return np.empty((0, 0, 0))

    def fetch(self, resolution_mm=None, voi=None, mapindex=None, clip=False):
        """
        Compute and return a spatial image for the given mip.

        Parameters:
        -----------
        resolution_mm : desired resolution in mm
        voi : BoundingBox
            optional bounding box
        """
        if clip:
            raise NotImplementedError(
                "Automatic clipping is not yet implemented for neuroglancer volume sources."
            )
        if mapindex is not None:
            raise NotImplementedError(
                f"NgVolume does not support access by map index (but {mapindex} was given)"
            )
        data = self._load_data(resolution_mm=resolution_mm, voi=voi)

        if data.ndim == 4:
            data = data.squeeze(axis=3)
        if data.ndim == 2:
            data = data.reshape(list(data.shape) + [1])

        return nibabel.Nifti1Image(data, self.build_affine(resolution_mm=resolution_mm, voi=voi))

    def get_shape(self, resolution_mm=None):
        mip = self._resolution_to_mip(resolution_mm, voi=None)
        return self.info["scales"][mip]["size"]

    def is_float(self):
        return np.dtype(self.info["data_type"]).kind == "f"

    def __hash__(self):
        return hash(self.url) + hash(self.transform_nm)

    def _enclosing_chunkgrid(self, mip, bbox_phys):
        """
        Produce grid points representing the chunks of the mip
        which enclose a given bounding box. The bounding box is given in
        physical coordinates, but the grid is returned in voxel spaces of the
        given mip.
        """

        # some helperfunctions to produce the smallest range on a grid enclosing another range
        def cfloor(x, s):
            int(np.floor(x / s) * s)

        def cceil(x, s):
            int(np.ceil(x / s) * s) + 1

        def crange(x0, x1, s):
            np.arange(cfloor(x0, s), cceil(x1, s), s)

        # project the bounding box to the voxel grid of the selected mip
        bb = np.dot(
            np.linalg.inv(self.build_affine(self.mip_resolution_mm[mip])), bbox_phys
        )

        # compute the enclosing chunk grid
        chunksizes = self.volume.scales[mip]["chunk_sizes"][0]
        x, y, z = [crange(bb[i][0], bb[i][1], chunksizes[i]) for i in range(3)]
        xx, yy, zz = np.meshgrid(x, y, z)
        return np.vstack([xx.ravel(), yy.ravel(), zz.ravel(), zz.ravel() * 0 + 1])


class DetailedMapsVolume(VolumeSrc, volume_type="detailed maps"):

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)


class GiftiSurfaceLabeling(VolumeSrc, volume_type="threesurfer/gii-label"):
    """
    TODO Implement this, surfaces need special handling
    """

    warning_shown = False

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        if not self.__class__.warning_shown:
            logger.info(
                f"A {self.__class__.__name__} object was registered, "
                "but this type is not yet explicitly supported."
            )
            self.__class__.warning_shown = True
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)

    @property
    def is_image_volume(self):
        """ Meshes are not volumes. """
        return False


class GiftiSurface(VolumeSrc, volume_type="threesurfer/gii"):
    """
    TODO Implement this, surfaces need special handling
    """

    warning_shown = False

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        if not self.__class__.warning_shown:
            logger.info(
                f"A {self.__class__.__name__} object was registered, "
                "but this type is not yet explicitly supported."
            )
            self.__class__.warning_shown = True
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)

    @property
    def is_image_volume(self):
        """ Meshes are not volumes. """
        return False


class NeuroglancerMesh(VolumeSrc, volume_type="neuroglancer/precompmesh"):
    """
    TODO Implement this, surfaces need special handling
    """

    warning_shown = False

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        if not self.__class__.warning_shown:
            logger.info(
                f"A {self.__class__.__name__} object was registered, "
                "but this type is not yet explicitly supported."
            )
            self.__class__.warning_shown = True
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)

    @property
    def is_image_volume(self):
        """ Meshes are not volumes. """
        return False
