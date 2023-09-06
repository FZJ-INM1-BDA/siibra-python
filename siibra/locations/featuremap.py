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

from . import location, boundingbox
from ..commons import logger
from nibabel import Nifti1Image
import numpy as np
from typing import Union


class FeatureMap(location.Location):

    def __init__(self, niftiimg: Union[str, Nifti1Image], spacespec: str):
        location.Location.__init__(self, spacespec)
        self.image = Nifti1Image.load(niftiimg) if isinstance(niftiimg, str) else niftiimg
        self._id_cached = None
        self._boundingbox_cached = None

    @property
    def boundingbox(self):
        if self._boundingbox_cached is None:
            self._boundingbox_cached = boundingbox.BoundingBox.from_image(
                self.image, self.space
            )
        return self._boundingbox_cached

    @property
    def id(self):
        # identify this feature map by the hashed array and affine of the nifti image, if needed.
        if self._id_cached is None:
            self._id_cached = hash(self.image.get_fdata().tobytes() + self.image.affine.tobytes())
        return self._id_cached

    def __iter__(self) -> location.Location:
        """iterate coordinates of the feature map - empty for now"""
        return iter(())

    def contains(self, other: location.Location) -> bool:
        return self.boundingbox.contains(other)

    def contained_in(self, other: location.Location) -> bool:
        if isinstance(other, Nifti1Image):
            # TODO we assume here the nifti is in the same space, which might be wrong.
            # better to not allow niftiimages and require featuremaps
            other_bbox = boundingbox.BoundingBox.from_image(other, self.space)
            return other_bbox.contains(self.boundingbox)
        else:
            assert isinstance(other, location.Location)
            return other.contains(self.boundingbox)

    def intersection(self, other: location.Location) -> location.Location:
        pass

    def transform(self, affine: np.ndarray, space=None):
        """ only modifies the affine matrix and space. """
        return FeatureMap(
            Nifti1Image(self.niftiimg.dataobj, np.dot(affine, self.niftiimg.affine)),
            spacespec=space
        )

    def warp(self, space):
        if self.space.matches(space):
            return self
        logger.warn('Warping of Nifti images requested, but not yet supported')
        return None
