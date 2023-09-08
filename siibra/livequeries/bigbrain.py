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

from . import query

from ..features.tabular import bigbrain_intensity_profile, layerwise_bigbrain_intensities
from ..features import anchor as _anchor
from ..commons import logger
from ..locations import location, point, pointset
from ..core import region
from ..retrieval import requests, cache

import numpy as np
from typing import List
from os import path


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE = "data/profiles_left.npy"
    THICKNESSES_FILE = "data/thicknesses_left.npy"
    MESH_FILE = "data/gray_left_327680.surf.gii"

    _profiles = None
    _vertices = None
    _boundary_depths = None

    def __init__(self):
        if self._profiles is None:
            self.__class__._load()

    @property
    def profile_labels(self):
        return np.arange(0, 1, 1 / self._profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last column which is the total thickness
        mesh_left = requests.HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.MESH_FILE}").data
        T = requests.HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.THICKNESSES_FILE}").data.T
        total = T[:, :-1].sum(1)
        valid = np.where(total > 0)[0]
        boundary_depths = np.c_[np.zeros_like(valid), (T[valid, :-1] / total[valid, None]).cumsum(1)]
        boundary_depths[:, -1] = 1

        # read profiles with valid thickenss
        url = f"{cls.REPO}/raw/{cls.BRANCH}/{cls.PROFILES_FILE}"
        if not path.exists(cache.CACHE.build_filename(url)):
            logger.info(
                "First request to BigBrain profiles. "
                "Downloading and preprocessing the data now. "
                "This may take a little."
            )
        req = requests.HttpRequest(url)

        cls._boundary_depths = boundary_depths
        cls._vertices = mesh_left.darrays[0].data[valid, :]
        cls._profiles = req.data[valid, :]
        logger.debug(f"{cls._profiles.shape[0]} BigBrain intensity profiles.")
        assert cls._vertices.shape[0] == cls._profiles.shape[0]

    def __len__(self):
        return self._vertices.shape[0]


class BigBrainProfileQuery(query.LiveQuery, args=[], FeatureType=bigbrain_intensity_profile.BigBrainIntensityProfile):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, concept: location.LocationFilter, **kwargs) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
        loader = WagstylProfileLoader()
        features = []
        regionname = concept.name if isinstance(concept, region.Region) else str(concept)
        matched = concept.intersection(pointset.PointSet(loader._vertices, space='bigbrain'))
        assert matched.labels is not None
        for i in matched.labels:
            prof = bigbrain_intensity_profile.BigBrainIntensityProfile(
                regionname=regionname,
                depths=loader.profile_labels,
                values=loader._profiles[i],
                boundaries=loader._boundary_depths[i],
                location=point.Point(loader._vertices[i], space='bigbrain')
            )
            prof.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
                query_structure=concept,
                assigned_structure=concept,
                qualification=_anchor.AssignmentQualification.CONTAINED,
                explanation=f"Surface vertex of BigBrain cortical profile was filtered using {concept}"
            )
            features.append(prof)

        return features


class LayerwiseBigBrainIntensityQuery(query.LiveQuery, args=[], FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, concept: location.LocationFilter, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:

        loader = WagstylProfileLoader()
        regionname = concept.name if isinstance(concept, region.Region) else str(concept)
        matched = concept.intersection(pointset.PointSet(loader._vertices, space='bigbrain'))
        indices = matched.labels
        matched_profiles = loader._profiles[indices, :]
        boundary_depths = loader._boundary_depths[indices, :]

        # compute array of layer labels for all coefficients in profiles_left
        N = matched_profiles.shape[1]
        prange = np.arange(N)
        region_labels = 7 - np.array([
            [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
            for b in boundary_depths
        ]).reshape((-1, 200))

        result = layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities(
            regionname=regionname,
            means=[matched_profiles[region_labels == _].mean() for _ in range(1, 7)],
            stds=[matched_profiles[region_labels == _].std() for _ in range(1, 7)],
        )
        result.anchor._location_cached = pointset.PointSet(loader._vertices[indices, :], space='bigbrain')
        result.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
            query_structure=concept,
            assigned_structure=concept,
            qualification=_anchor.AssignmentQualification.CONTAINED,
            explanation=f"Surface vertices of BigBrain cortical profiles were filtered using {concept}"
        )

        return [result]
