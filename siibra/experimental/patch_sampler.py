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

from ..locations import point, boundingbox
from ..volumes import volume

import numpy as np
import math
from nilearn import image

def translation_matrix(tx: float, ty: float, tz: float):
    """Construct a 3D homoegneous translation matrix. """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def y_rotation_matrix(alpha: float):
    """Construct a 3D y axis rotation matrix. """
    return np.array([
        [math.cos(alpha), 0, math.sin(alpha), 0],
        [0, 1, 0, 0],
        [-math.sin(alpha), 0, math.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

def fetch_patch(patch_corners: np.ndarray, image_volume: volume.Volume, resolution_mm: float):
    
    # Format corners into a complete 3D bounding box in physcial space
    canvas = image_volume.get_boundingbox()
    y1, y2 = canvas.minpoint[1], canvas.maxpoint[1]
    patch_voi = boundingbox.BoundingBox(
        patch_corners.min(0)[:3], 
        patch_corners.max(0)[:3], 
        space=image_volume.space
    )
    patch_voi.minpoint[1] = y1
    patch_voi.maxpoint[1] = y2
    patch = image_volume.fetch(voi=patch_voi, resolution_mm=resolution_mm)
    assert patch is not None
    
    # patch rotation defined in physical space
    vx, vy, vz, _ = patch_corners[1] - patch_corners[0]
    alpha = -math.atan2(-vz, -vx)
    cx, cy, cz = patch_corners.mean(0)[:3]
    rot_phys = np.linalg.multi_dot([
        translation_matrix(cx, cy, cz),
        y_rotation_matrix(alpha),
        translation_matrix(-cx, -cy, -cz),
    ])

    # rotate the patch in physical space
    imgdata = image_volume.fetch(voi=patch_voi)
    affine_rot = np.dot(rot_phys, patch.affine)
    patch_rot = volume.from_nifti(
        image.resample_img(patch, target_affine=affine_rot),
        space=image_volume.space, name=""
    )

    # crop in the rotated space
    pixels = np.dot(np.linalg.inv(affine_rot), patch_corners.T).astype('int').T
    xmin, ymin, zmin = pixels.min(0)[:3] + 1  # keep a pixel distance to avoid black border pixels
    xmax, ymax, zmax = pixels.max(0)[:3] - 1 
    h, w = xmax - xmin, zmax - zmin
    affine = np.dot(affine_rot, translation_matrix(xmin, 0, zmin))
    return volume.from_nifti(
        image.resample_img(patch, target_affine=affine, target_shape=[h, 1, w]),
        space=image_volume.space, name=""
    )

