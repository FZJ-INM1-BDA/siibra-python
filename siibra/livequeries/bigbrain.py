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
from ..core import region, structure
from ..retrieval import requests, cache

import numpy as np
from typing import List
from os import path


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE_LEFT = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-26d25994-634c-40af-b88f-2a36e8e1d508/profiles/profiles_left.txt"
    PROFILES_FILE_RIGHT = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-26d25994-634c-40af-b88f-2a36e8e1d508/profiles/profiles_right.txt"
    THICKNESSES_FILE_LEFT = "data/thicknesses_left.npy"
    THICKNESSES_FILE_RIGHT = ""
    MESH_FILE_LEFT = "gray_left_327680.surf.gii"
    MESH_FILE_RIGHT = "gray_right_327680.surf.gii"
    BASEURL = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Surfaces/Apr7_2016/gii/"
    _profiles = None
    _vertices = None
    _boundary_depths = None

    def __init__(self):
        if self._profiles is None:
            self.__class__._load()

    @property
    def profile_labels(self):
        return np.arange(0., 1., 1. / self._profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last column which is the total thickness
        thickness_left = requests.HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.THICKNESSES_FILE_LEFT}").data.T
        thickness_right = np.zeros(shape=thickness_left.shape)  # TODO: replace with thickness data for te right hemisphere
        thickness = np.concatenate((thickness_left[:, :-1], thickness_right[:, :-1]))  # last column is the computed total thickness
        total_thickness = thickness.sum(1)
        valid = np.where(total_thickness > 0)[0]
        cls._boundary_depths = np.c_[np.zeros_like(valid), (thickness[valid, :] / total_thickness[valid, None]).cumsum(1)]
        cls._boundary_depths[:, -1] = 1  # account for float calculation errors

        # find profiles with valid thickness
        if not all(
            path.exists(cache.CACHE.build_filename(url))
            for url in [cls.PROFILES_FILE_LEFT, cls.PROFILES_FILE_RIGHT]
        ):
            logger.info(
                "First request to BigBrain profiles. Preprocessing the data "
                "now. This may take a little."
            )
        profiles_l = requests.HttpRequest(cls.PROFILES_FILE_LEFT).data.to_numpy()
        profiles_r = requests.HttpRequest(cls.PROFILES_FILE_RIGHT).data.to_numpy()
        cls._profiles = np.concatenate((profiles_l, profiles_r))[valid, :]

        # read mesh vertices
        mesh_left = requests.HttpRequest(f"{cls.BASEURL}/{cls.MESH_FILE_LEFT}").data
        mesh_right = requests.HttpRequest(f"{cls.BASEURL}/{cls.MESH_FILE_RIGHT}").data
        mesh_vertices = np.concatenate((mesh_left.darrays[0].data, mesh_right.darrays[0].data))
        cls._vertices = mesh_vertices[valid, :]

        logger.debug(f"{cls._profiles.shape[0]} BigBrain intensity profiles.")
        assert cls._vertices.shape[0] == cls._profiles.shape[0]

    @staticmethod
    def _choose_space(regionobj: region.Region):
        """
        Helper function to obtain a suitable space to fetch a regional mask.
        BigBrain space has priorty.

        Parameters
        ----------
        regionobj : region.Region

        Returns
        -------
        Space

        Raises
        ------
        RuntimeError
            When there is no supported space that provides image for `regionobj`.
        """
        if regionobj.mapped_in_space('bigbrain'):
            return 'bigbrain'
        supported_spaces = [s for s in regionobj.supported_spaces if s.provides_image]
        if len(supported_spaces) == 0:
            raise RuntimeError(f"Could not find a supported space for {regionobj}")
        return supported_spaces[0]

    def __len__(self):
        return self._vertices.shape[0]


class BigBrainProfileQuery(query.LiveQuery, args=[], FeatureType=bigbrain_intensity_profile.BigBrainIntensityProfile):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, concept: structure.BrainStructure, **kwargs) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
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

    def query(self, concept: structure.BrainStructure, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:

        loader = WagstylProfileLoader()
        if isinstance(concept, region.Region):
            regionname = concept.name
            space = WagstylProfileLoader._choose_space(concept)
        else:
            regionname = str(concept)
            space = 'bigbrain'
        matched = concept.intersection(pointset.PointSet(loader._vertices, space=space))
        indices = matched.labels
        if indices is None:
            return []
        matched_profiles = loader._profiles[indices, :]
        boundary_depths = loader._boundary_depths[indices, :]
        # compute array of layer labels for all coefficients in profiles_left
        N = matched_profiles.shape[1]
        prange = np.arange(N)
        layer_labels = 7 - np.array([
            [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
            for b in boundary_depths
        ]).reshape((-1, 200))

        result = layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities(
            regionname=regionname,
            means=[matched_profiles[layer_labels == layer].mean() for layer in range(1, 7)],
            stds=[matched_profiles[layer_labels == layer].std() for layer in range(1, 7)],
        )
        result.anchor._location_cached = pointset.PointSet(loader._vertices[indices, :], space='bigbrain')
        result.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
            query_structure=concept,
            assigned_structure=concept,
            qualification=_anchor.AssignmentQualification.CONTAINED,
            explanation=f"Surface vertices of BigBrain cortical profiles were filtered using {concept}"
        )

        return [result]
