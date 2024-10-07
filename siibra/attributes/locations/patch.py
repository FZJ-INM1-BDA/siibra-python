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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..dataproviders.volume import image

import numpy as np
import math
from nilearn.image import resample_img

from . import polyline, boundingbox


@dataclass
class Patch(polyline.PolyLine):
    """
    Construct a 2D patch sitting in 3D space, represented by its 4 corner points.
    As of now, only patches aligned in the y plane of the physical space
    are supported.
    """

    schema: str = None

    def __post_init__(self):
        assert len(self.coordinates) == 4
        assert len(np.unique(self.coordinates[:, 1])) == 1

    def extract_volume(self, imgprovider: "image.ImageRecipe", resolution_mm: float):
        """
        fetches image data in a planar patch.
        TODO The current implementation only covers patches which are strictly
        define in the y plane. A future implementation should accept arbitrary
        oriented patches.
        """
        assert imgprovider.space_id == self.space_id

        # Extend the 2D patch into a 3D structure
        # this is only valid if the patch plane lies within the image canvas.
        canvas = boundingbox.from_imageprovider(imgprovider)
        assert canvas.minpoint[1] <= self.corners.coordinates[0, 1]
        assert canvas.maxpoint[1] >= self.corners.coordinates[0, 1]
        XYZ = self.corners.coordinates
        voi = boundingbox.BoundingBox(
            minpoint=XYZ.min(0)[:3],
            maxpoint=XYZ.max(0)[:3],
            space_id=imgprovider.space_id,
        )
        # enforce the patch to have the same y dimensions
        voi.minpoint[1] = canvas.minpoint[1]
        voi.maxpoint[1] = canvas.maxpoint[1]
        patch = imgprovider.get_data(voi=voi, resolution=resolution_mm)
        assert patch is not None

        # patch rotation defined in physical space
        vx, vy, vz = XYZ[1] - XYZ[0]
        alpha = -math.atan2(-vz, -vx)
        cx, cy, cz = XYZ.mean(0)
        rot_phys = np.linalg.multi_dot(
            [
                translation_matrix(cx, cy, cz),
                y_rotation_matrix(alpha),
                translation_matrix(-cx, -cy, -cz),
            ]
        )

        # rotate the patch in physical space
        affine_rot = np.dot(rot_phys, patch.affine)

        # crop in the rotated space
        pixels = (
            np.dot(np.linalg.inv(affine_rot), self.corners.homogeneous.T)
            .astype("int")
            .T
        )
        # keep a pixel distance to avoid black border pixels
        xmin, ymin, zmin = pixels.min(0)[:3] + 1
        xmax, ymax, zmax = pixels.max(0)[:3] - 1
        h, w = xmax - xmin, zmax - zmin
        affine = np.dot(affine_rot, translation_matrix(xmin, 0, zmin))

        from ..dataproviders.volume import image

        return image.from_nifti(
            resample_img(patch, target_affine=affine, target_shape=[h, 1, w]),
            space_id=imgprovider.space_id,
            name=f"Rotated patch with corner points {self.corners} sampled from {imgprovider.name}",
        )

    def flip(self):
        """Flips the patch."""
        self.coordinates = self.coordinates[[2, 3, 0, 1]]


def translation_matrix(tx: float, ty: float, tz: float):
    """Construct a 3D homoegneous translation matrix."""
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])


def y_rotation_matrix(alpha: float):
    """Construct a 3D y axis rotation matrix."""
    return np.array(
        [
            [math.cos(alpha), 0, math.sin(alpha), 0],
            [0, 1, 0, 0],
            [-math.sin(alpha), 0, math.cos(alpha), 0],
            [0, 0, 0, 1],
        ]
    )
