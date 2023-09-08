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

from . import location, boundingbox, point, pointset
from ..commons import logger

from nibabel import Nifti1Image
import numpy as np
from typing import Union


class SpatialMap(location.Location):
    """ A Nifti1image with a reference space location attached to it. """

    def __init__(self, niftiimg: Union[str, Nifti1Image], spacespec: str, description: str = ""):
        location.Location.__init__(self, spacespec)
        self.image = Nifti1Image.load(niftiimg) if isinstance(niftiimg, str) else niftiimg
        self._id_cached = None
        self._boundingbox_cached = None
        self.description = description

    @property
    def boundingbox(self):
        if self._boundingbox_cached is None:
            self._boundingbox_cached = boundingbox.BoundingBox.from_image(self.image, self.space)
        return self._boundingbox_cached

    @property
    def affine(self):
        return self.image.affine

    @property
    def id(self):
        # identify this feature map by the hashed array and affine of the nifti image, if needed.
        if self._id_cached is None:
            self._id_cached = hash(self.image.get_fdata().tobytes() + self.affine.tobytes())
        return self._id_cached
    
    def __str__(self):
        space_str = "" if self.space is None else f" defined in {self.space.name}"
        return f"{self.__class__.__name__} {self.description}{space_str}"

    def __iter__(self) -> location.Location:
        """iterate coordinates of the feature map - empty for now"""
        return iter(())

    def contains(self, other: location.Location, fast=False) -> bool:
        """
        tests containedness of another location in this spatial map.
        if fast==True, only bounding box testing is performed, otherwise
        exact voxels are used.
        """
        return self.intersection(other) == other

    def intersection(self, other: location.Location) -> location.Location:
        """ Compute the intersection of this image with an other location. """
        if isinstance(other, (pointset.PointSet, point.Point)):
            arr = np.asanyarray(self.image.dataobj)
            warped = other.warp(self.space)
            assert warped is not None
            phys2vox = np.linalg.inv(self.affine)
            voxels = warped.transform(phys2vox, space=None)
            XYZ = np.atleast_2d(np.array(voxels)).astype('int')
            invalid = np.where(
                np.all(XYZ >= arr.shape, axis=1)
                | np.all(XYZ < 0, axis=1)
            )[0]
            XYZ[invalid] = 0  # set all out-of-bounds vertices to (0, 0, 0)
            arr[0, 0, 0] = 0  # ensure the lower left voxel is not foreground
            inside = np.where(arr[tuple(zip(*XYZ))] != 0)[0]
            result = pointset.PointSet(
                np.atleast_2d(np.array(other))[inside],
                space=other.space,
                labels=inside
            )
            if len(result) == 0:
                return None
            elif len(result) == 1:
                return result[0]
            else:
                return result
        elif isinstance(other, boundingbox.BoundingBox):
            return self.boundingbox.intersection(other)
        else:
            raise NotImplementedError(
                f"Intersection of {self.__class__.__name__} with {other.__class__.__name__} not implemented."
            )

    def transform(self, affine: np.ndarray, space=None):
        """ only modifies the affine matrix and space. """
        return SpatialMap(
            Nifti1Image(self.dataobj, np.dot(affine, self.affine)),
            spacespec=space
        )

    def warp(self, space):
        if self.space.matches(space):
            return self
        logger.warn('Warping of Nifti images requested, but not yet supported')
        return None
