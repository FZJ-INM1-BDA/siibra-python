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

from ..commons import logger
from ..retrieval import requests
from ..locations import pointset

from typing import Union
import nibabel as nib
import os
import numpy as np


class NiftiFetcher(volume.VolumeProvider, srctype="nii"):

    def __init__(self, src: Union[str, nib.Nifti1Image]):
        """
        Construct a new NIfTI volume source, from url, local file, or Nift1Image object.
        """
        volume.VolumeProvider.__init__(self)
        self._image_cached = None
        self._src = src
        if isinstance(src, nib.Nifti1Image):
            self._image_cached = src
        elif isinstance(src, str):
            if os.path.isfile(src):
                self._image_loader = lambda fn=self._src: nib.load(fn)
            else:
                self._image_loader = lambda u=src: requests.HttpRequest(u).data
        else:
            raise ValueError(f"Invalid source specification for {self.__class__}: {src}")

    @property
    def image(self):
        if self._image_cached is None:
            self._image_cached = self._image_loader()
        return self._image_cached

    def fetch(self, resolution_mm=None, voi=None, **kwargs):
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
            pointset.PointSet(
                [np.dot(img.affine, [x, y, z, 1])[:3] for x, y, z in voxels],
                space=self.space,
            ),
            img,
        )


class ZipContainedNiftiFetcher(NiftiFetcher, srctype="zip/nii"):

    def __init__(self, src: str):
        """
        Construct a new NIfTI volume source, from url, local file, or Nift1Image object.
        """
        volume.VolumeProvider.__init__(self)
        zipurl, zipped_file = src.split(" ")
        self._image_cached = None
        self._image_loader = lambda u=zipurl: requests.ZipfileRequest(u, zipped_file).data
