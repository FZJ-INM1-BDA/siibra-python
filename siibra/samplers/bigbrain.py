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


from ..commons import logger
from ..configuration.configuration import REGISTRY
from ..locations import Point, PointSet

from ctypes import ArgumentError
import numpy as np
from skimage import measure
from nilearn import image
from tqdm import tqdm


class BigBrainCortexSampler:
    """Sample layerwise grayvalue statistics from BigBrain cortex.

    Given a coordinate in BigBrain space, the sampler centers a fixed-size box
    near the given location in BigBrain cortex and extracts layerwise grayvalue
    statistics. The center is found by identifying the closest point in Layer IV
    which is inside a specified maximum distance (default 1mm) from the given point.
    """

    def __init__(self, boxwidth_mm=3, maxdist_mm=1):
        """Create a new cortex sample for BigBrain.

        Arguments
        ---------
        boxwidth_mm : float
            Sidelength of the boxes used for grayvalue sampling (default: 3mm)
        maxdist_mm : float
            Maximum distance allowed to shift given sampling locations towards the
            mid surface of the cortex. If the mid surface is not found within
            this distance, no data will be sampled.
        """

        self.space = REGISTRY.Space["bigbrain"]
        self._layermask = REGISTRY.Parcellation["layers"].get_map(self.space)
        self._template = self.space.get_template()
        self.boxwidth_mm = boxwidth_mm
        self.maxdist_mm = maxdist_mm

    def sample(self, location):
        """
        Given sample locations in BigBrain space, the sampler centers a fixed-size box
        near the given location in BigBrain cortex and extracts layerwise grayvalue
        statistics. The center is found by identifying the closest point in Layer IV
        which is inside a specified maximum distance (default 1mm) from the given point.
        If the location is not close enough to the cortical midsurface, the data sample
        will be empty and a warning will be printed.

        Parameters
        ----------
        location: Point, or PointSet
            Candidate location(s) for sampling

        Return
        ------
        List of dicts, one per sample point, with keys:
            - 'center': the physical coordinate of the cube used as a region of interest
            - 'boxsize': sidelenght in mm f the cube used as a region of interest
            - 'space': name of the space (bigbrain)
            - 'layers': Dict of layer-wise statistics with mean gray value, standard deviation, and volume in mm
        """
        if location.space != self.space:
            logger.info(
                f"Warping sample locations from {location.space.name} to {self.space.name}"
            )
            loc_bb = location.warp(self.space)
        else:
            loc_bb = location

        result = []
        if isinstance(loc_bb, Point):
            result.append(self._sample_single_point(loc_bb))
        elif isinstance(loc_bb, PointSet):
            for p in tqdm(
                loc_bb,
                total=len(loc_bb),
                unit="locations",
                desc=f"Sampling from {len(loc_bb)} locations",
                disable=logger.level > 20,
            ):
                result.append(self._sample_single_point(p))
        else:
            raise ArgumentError(
                f"Invalid location specification {location.__class__.__name__} BigBrain sampling."
            )

        return result

    def _sample_single_point(self, point):
        """
        Given a coordinate in BigBrain space, the sampler centers a fixed-size box
        near the given location in BigBrain cortex and extracts layerwise grayvalue
        statistics. The center is found by identifying the closest point in Layer IV
        which is inside a specified maximum distance (default 1mm) from the given point.

        Parameters
        ----------
        point: Point
            Candidate location for sampling

        Return
        ------
        dict with keys:
            - 'center': the physical coordinate of the cube used as a region of interest
            - 'boxsize': sidelenght in mm f the cube used as a region of interest
            - 'space': name of the space (bigbrain)
            - 'layers': Dict of layer-wise statistics with mean gray value, standard deviation, and number of voxels

        """
        result = {
            "center": tuple(point),
            "boxsize": self.boxwidth_mm,
            "space": self.space.name,
            "layers": {},
        }

        # find closest point in layer 4, i.e. towards the mid surface,
        # to make sure the box is well centered in the cortex.
        voi = point.get_enclosing_cube(2 * self.maxdist_mm + 1)
        voimask = self._layermask.fetch(voi=voi, resolution_mm=-1)
        L = np.asanyarray(voimask.dataobj)
        XYZ_ = np.array(np.where(L == 4)).T
        if XYZ_.shape[0] == 0:
            logger.warn(
                f"The point {tuple(point)} seems too far away from Layer IV in BigBrain. "
                f"No data sampled."
            )
            return result
        XYZ = np.dot(voimask.affine, np.c_[XYZ_, np.ones(XYZ_.shape[0])].T)[:3, :].T
        D = np.sum(np.sqrt((XYZ - tuple(point)) ** 2), axis=1)
        p_mid = Point(XYZ[np.argmin(D), :], self.space)

        # Load cube of fixed size at this cortica position
        # from BigBrain and cortical layer maps,
        # using maximum available resolution.
        voi = p_mid.get_enclosing_cube(self.boxwidth_mm)
        voidata = self._template.fetch(voi=voi, resolution_mm=-1)
        voimask = image.resample_to_img(
            self._layermask.fetch(voi=voi, resolution_mm=-1),
            voidata,
            interpolation="nearest",
        )

        # Get layer mask with possible additional segmentation
        # components at box borders suppressed (e.g. from neighboring
        # sulcal walls)
        L = np.asanyarray(voimask.dataobj)
        M = measure.label(L > 0)
        cx, cy, cz = np.array(M.shape) // 2
        L[M != M[cx, cy, cz]] = 0

        # extract layerwise grayvalues, excluding value 255
        # which we assume to belong to background.
        result["center"] = tuple(p_mid)
        arr = np.asanyarray(voidata.dataobj)
        for layer in range(1, 7):
            mask = (L == layer) & (arr != 255)
            values = arr[mask].ravel()
            result["layers"][layer] = {
                "mean": values.mean(),
                "std": values.std(),
                "num_voxels": np.count_nonzero(mask),
            }

        return result
