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


from .query import LiveQuery

from ..commons import logger
from ..locations import PointSet, Point
from ..core.space import Space
from ..core.region import Region
from ..retrieval import HttpRequest
from ..features import BigBrainIntensityFingerprint, BigBrainIntensityProfile

import numpy as np
from typing import List


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE = "data/profiles_left.npy"
    THICKNESSES_FILE = "data/thicknesses_left.npy"
    MESH_FILE = "data/gray_left_327680.surf.gii"

    profiles = None
    vertices = None
    boundary_depths = None

    def __init__(self):
        if self.profiles is None:
            self.__class__._load()

    @property
    def profile_labels(self):
        return np.arange(0, 1, 1 / self.profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last columsn which is the total thickness
        mesh_left = HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.MESH_FILE}").data
        T = HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.THICKNESSES_FILE}").data.T
        total = T[:, :-1].sum(1)
        valid = np.where(total > 0)[0]
        boundary_depths = np.c_[np.zeros_like(valid), (T[valid, :-1] / total[valid, None]).cumsum(1)]
        boundary_depths[:, -1] = 1

        # read profiles with valid thickenss
        req = HttpRequest(
            f"{cls.REPO}/raw/{cls.BRANCH}/{cls.PROFILES_FILE}",
            msg_if_not_cached="First request to BigBrain profiles. Downloading and preprocessing the data now. This may take a little."
        )

        cls.boundary_depths = boundary_depths
        cls.vertices = mesh_left.darrays[0].data[valid, :]
        cls.profiles = req.data[valid, :]
        logger.debug(f"{cls.profiles.shape[0]} BigBrain intensity profiles.")
        assert cls.vertices.shape[0] == cls.profiles.shape[0]

    def __len__(self):
        return self.vertices.shape[0]

    def match(self, region: Region):
        assert isinstance(region, Region)
        logger.debug(f"Matching locations of {len(self)} BigBrain profiles to {region}")

        for space in region.supported_spaces:
            if not space.is_surface:
                try:
                    mask = region.build_mask(space=space, maptype="labelled")
                except RuntimeError:
                    continue
                logger.info(f"Assigning {len(self)} profile locations to {region} in {space}...")
                pts = PointSet(self.vertices, space="bigbrain").warp(space)
                inside = [i for i, p in enumerate(pts) if p.contained_in(mask)]
                break
        else:
            raise RuntimeError(f"Could not filter big brain profiles by {region}")

        return (
            self.profiles[inside, :],
            self.boundary_depths[inside, :],
            self.vertices[inside, :]
        )


class BigBrainProfileQuery(LiveQuery, args=[], FeatureType=BigBrainIntensityProfile):

    def __init__(self):
        LiveQuery.__init__(self)

    def query(self, region: Region, **kwargs) -> List[BigBrainIntensityProfile]:
        assert isinstance(region, Region)
        profiles = WagstylProfileLoader()

        features = []
        for subregion in region.leaves:
            matched_profiles, boundary_depths, coords = profiles.match(subregion)
            bbspace = Space.get_instance('bigbrain')
            features.extend(
                BigBrainIntensityProfile(
                    regionname=subregion.name,
                    depths=profiles.profile_labels,
                    values=profile,
                    boundaries=boundary_depths[i, :],
                    location=Point(coords[i, :], bbspace)
                )
                for i, profile in enumerate(matched_profiles)
            )
        return features


class BigBrainIntensityFingerprintQuery(LiveQuery, args=[], FeatureType=BigBrainIntensityFingerprint):

    def __init__(self):
        LiveQuery.__init__(self)

    def query(self, region: Region, **kwargs) -> List[BigBrainIntensityFingerprint]:
        assert isinstance(region, Region)
        profiles = WagstylProfileLoader()

        result = []
        for subregion in region.leaves:
            matched_profiles, boundary_depths, coords = profiles.match(region)

            # compute array of layer labels for all coefficients in profiles_left
            N = matched_profiles.shape[1]
            prange = np.arange(N)
            region_labels = 7 - np.array([
                [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
                for b in boundary_depths
            ]).squeeze()

            fp = BigBrainIntensityFingerprint(
                regionname=subregion.name,
                means=[matched_profiles[region_labels == _].mean() for _ in range(1, 7)],
                stds=[matched_profiles[region_labels == _].std() for _ in range(1, 7)],
            )
            result.append(fp)

        return result
