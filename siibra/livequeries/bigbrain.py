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
"""Matches BigBrain intesity profiles extracted by Wagstyl et al. to volumes."""

from . import query

from ..features.tabular import bigbrain_intensity_profile, layerwise_bigbrain_intensities
from ..features import anchor as _anchor
from ..commons import logger
from ..locations import point, pointset
from ..core import structure
from ..retrieval import requests, cache
from ..retrieval.datasets import GenericDataset

import numpy as np
from typing import List
from os import path


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial/raw/main"
    PROFILES_FILE_LEFT = "data/profiles_left.npy"
    THICKNESSES_FILE_LEFT = "data/thicknesses_left.npy"
    MESH_FILE_LEFT = "data/gray_left_327680.surf.gii"
    _profiles = None
    _vertices = None
    _boundary_depths = None
    DATASET = GenericDataset(
        name="HIBALL workshop on cortical layers",
        contributors=[
            'Konrad Wagstyl',
            'Stéphanie Larocque',
            'Guillem Cucurull',
            'Claude Lepage',
            'Joseph Paul Cohen',
            'Sebastian Bludau',
            'Nicola Palomero-Gallagher',
            'Lindsay B. Lewis',
            'Thomas Funck',
            'Hannah Spitzer',
            'Timo Dickscheid',
            'Paul C. Fletcher',
            'Adriana Romero',
            'Karl Zilles',
            'Katrin Amunts',
            'Yoshua Bengio',
            'Alan C. Evans'
        ],
        url="https://github.com/kwagstyl/cortical_layers_tutorial/",
        description="Cortical profiles of BigBrain staining intensities computed by Konrad Wagstyl, "
        "as described in the publication 'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678."
        "The data is taken from the tutorial at "
        "https://github.com/kwagstyl/cortical_layers_tutorial. Each vertex is "
        "assigned to the regional map when queried."
    )

    def __init__(self):
        if self._profiles is None:
            self.__class__._load()

    @property
    def profile_labels(self):
        return np.arange(0., 1., 1. / self._profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last column which is the total thickness
        thickness = requests.HttpRequest(f"{cls.REPO}/{cls.THICKNESSES_FILE_LEFT}").data.T
        total_thickness = thickness[:, :-1].sum(1)  # last column is the computed total thickness
        valid = np.where(total_thickness > 0)[0]
        cls._boundary_depths = np.c_[np.zeros_like(valid), (thickness[valid, :] / total_thickness[valid, None]).cumsum(1)]
        cls._boundary_depths[:, -1] = 1  # account for float calculation errors

        # find profiles with valid thickness
        profile_l_url = f"{cls.REPO}/{cls.PROFILES_FILE_LEFT}"
        if not path.exists(cache.CACHE.build_filename(profile_l_url)):
            logger.info(
                "First request to BigBrain profiles. Preprocessing the data "
                "now. This may take a little."
            )
        profiles_l_all = requests.HttpRequest(profile_l_url).data
        cls._profiles = profiles_l_all[valid, :]

        # read mesh vertices
        mesh_left = requests.HttpRequest(f"{cls.REPO}/{cls.MESH_FILE_LEFT}").data
        mesh_vertices = mesh_left.darrays[0].data
        cls._vertices = mesh_vertices[valid, :]

        logger.debug(f"{cls._profiles.shape[0]} BigBrain intensity profiles.")
        assert cls._vertices.shape[0] == cls._profiles.shape[0]

    def __len__(self):
        return self._vertices.shape[0]


cache.Warmup.register_warmup_fn()(lambda: WagstylProfileLoader._load())


class BigBrainProfileQuery(query.LiveQuery, args=[], FeatureType=bigbrain_intensity_profile.BigBrainIntensityProfile):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, concept: structure.BrainStructure, **kwargs) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
        loader = WagstylProfileLoader()
        mesh_vertices = pointset.PointSet(loader._vertices, space='bigbrain')
        matched = concept.intersection(mesh_vertices)  # returns a reduced PointSet with og indices as labels
        if matched is None:
            return []
        assert isinstance(matched, pointset.PointSet)
        indices = matched.labels
        assert indices is not None
        features = []
        for i in matched.labels:
            anchor = _anchor.AnatomicalAnchor(
                location=point.Point(loader._vertices[i], space='bigbrain'),
                region=str(concept),
                species='Homo sapiens'
            )
            prof = bigbrain_intensity_profile.BigBrainIntensityProfile(
                anchor=anchor,
                depths=loader.profile_labels,
                values=loader._profiles[i],
                boundaries=loader._boundary_depths[i]
            )
            prof.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
                query_structure=concept,
                assigned_structure=concept,
                qualification=_anchor.Qualification.CONTAINED,
                explanation=f"Surface vertex of BigBrain cortical profile was filtered using {concept}"
            )
            prof.datasets = [WagstylProfileLoader.DATASET]
            features.append(prof)

        return features


class LayerwiseBigBrainIntensityQuery(query.LiveQuery, args=[], FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, concept: structure.BrainStructure, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:

        loader = WagstylProfileLoader()
        mesh_vertices = pointset.PointSet(loader._vertices, space='bigbrain')
        matched = concept.intersection(mesh_vertices)  # returns a reduced PointSet with og indices as labels
        if matched is None:
            return []
        assert isinstance(matched, pointset.PointSet)
        indices = matched.labels
        assert indices is not None
        matched_profiles = loader._profiles[indices, :]
        boundary_depths = loader._boundary_depths[indices, :]
        # compute array of layer labels for all coefficients in profiles_left
        N = matched_profiles.shape[1]
        prange = np.arange(N)
        layer_labels = 7 - np.array([
            [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
            for b in boundary_depths
        ]).reshape((-1, 200))

        anchor = _anchor.AnatomicalAnchor(
            location=pointset.PointSet(loader._vertices[indices, :], space='bigbrain'),
            region=str(concept),
            species='Homo sapiens'
        )
        result = layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities(
            anchor=anchor,
            means=[matched_profiles[layer_labels == layer].mean() for layer in range(1, 7)],
            stds=[matched_profiles[layer_labels == layer].std() for layer in range(1, 7)],
        )
        result.anchor._assignments[concept] = _anchor.AnatomicalAssignment(
            query_structure=concept,
            assigned_structure=concept,
            qualification=_anchor.Qualification.CONTAINED,
            explanation=f"Surface vertices of BigBrain cortical profiles were filtered using {concept}"
        )
        result.datasets = [WagstylProfileLoader.DATASET]

        return [result]
