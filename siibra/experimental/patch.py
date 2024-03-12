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

from ..volumes import volume
from ..locations import pointset, boundingbox
from ..commons import translation_matrix, y_rotation_matrix

import numpy as np
import math
from nilearn import image


class Patch:

    def __init__(self, corners: pointset.PointSet):
        """Construct a patch in physical coordinates.
        As of now, only patches aligned in the y plane of the physical space
        are supported."""
        # TODO: need to ensure that the points are planar, if more than 3
        assert len(corners) == 4
        assert len(np.unique(corners.coordinates[:, 1])) == 1
        self.corners = corners

    @property
    def space(self):
        return self.corners.space

    def flip(self):
        """Flips the patch. """
        self.corners._coordinates = self.corners.coordinates[[2, 3, 0, 1]]

    def extract_volume(self, image_volume: volume.Volume, resolution_mm: float):
        """
        fetches image data in a planar patch.
        TODO The current implementation only covers patches which are strictly
        define in the y plane. A future implementation should accept arbitrary
        oriented patches.accept arbitrary oriented patches.
        """
        assert image_volume.space == self.space

        # Extend the 2D patch into a 3D structure
        # this is only valid if the patch plane lies within the image canvas.
        canvas = image_volume.get_boundingbox()
        assert canvas.minpoint[1] <= self.corners.coordinates[0, 1]
        assert canvas.maxpoint[1] >= self.corners.coordinates[0, 1]
        XYZ = self.corners.coordinates
        voi = boundingbox.BoundingBox(
            XYZ.min(0)[:3], XYZ.max(0)[:3], space=image_volume.space
        )
        # enforce the patch to have the same y dimensions
        voi.minpoint[1] = canvas.minpoint[1]
        voi.maxpoint[1] = canvas.maxpoint[1]
        patch = image_volume.fetch(voi=voi, resolution_mm=resolution_mm)
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
        return volume.from_nifti(
            image.resample_img(patch, target_affine=affine, target_shape=[h, 1, w]),
            space=image_volume.space,
            name=f"Rotated patch with corner points {self.corners} sampled from {image_volume.name}",
        )
