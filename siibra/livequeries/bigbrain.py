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
from ..commons import logger
from ..locations import point, pointset
from ..core import region
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

    def match(self, regionobj: region.Region, space: str = None):
        """
        Retrieve BigBrainIntesityProfiles .

        Parameters
        ----------
        regionobj : region.Region
            Region or parcellation
        space : str, optional
            The space in which the region masks will be calculated. By default,
            siibra tries to fetch in BigBrain first and then the other spaces.

        Returns
        -------
        tuple
            tuple of profiles, boundary depths, and vertices
        """
        assert isinstance(regionobj, region.Region)
        logger.debug(f"Matching locations of {len(self)} BigBrain profiles to {regionobj}")

        if space is None:
            space = self._choose_space(regionobj)

        mask = regionobj.fetch_regional_map(space=space, maptype="labelled")
        logger.info(f"Assigning {len(self)} profile locations to {regionobj} in {space}...")
        voxels = (
            pointset.PointSet(self._vertices, space="bigbrain")
            .warp(space)
            .transform(np.linalg.inv(mask.affine), space=None)
        )
        arr = mask.get_fdata()
        XYZ = np.array(voxels.as_list()).astype('int')
        X, Y, Z = np.split(
            XYZ[np.all((XYZ < arr.shape) & (XYZ > 0), axis=1), :],
            3, axis=1
        )
        inside = np.where(arr[X, Y, Z] > 0)[0]

        return (
            self._profiles[inside, :],
            self._boundary_depths[inside, :],
            self._vertices[inside, :]
        )


class BigBrainProfileQuery(query.LiveQuery, args=[], FeatureType=bigbrain_intensity_profile.BigBrainIntensityProfile):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, regionobj: region.Region, **kwargs) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
        assert isinstance(regionobj, region.Region)
        loader = WagstylProfileLoader()

        space = WagstylProfileLoader._choose_space(regionobj)
        if not regionobj.is_leaf:
            leaves_defined_on_space = [
                r for r in regionobj.leaves if r.mapped_in_space(space)
            ]
        else:
            leaves_defined_on_space = [regionobj]

        result = []
        for subregion in leaves_defined_on_space:
            matched_profiles, boundary_depths, coords = loader.match(subregion, space)
            for i, profile in enumerate(matched_profiles):
                prof = bigbrain_intensity_profile.BigBrainIntensityProfile(
                    regionname=subregion.name,
                    depths=loader.profile_labels,
                    values=profile,
                    boundaries=boundary_depths[i, :],
                    location=point.Point(coords[i, :], 'bigbrain')  # points are warped into BigBrain
                )
                prof.datasets = [WagstylProfileLoader.DATASET]
                result.append(prof)

        return result


class LayerwiseBigBrainIntensityQuery(query.LiveQuery, args=[], FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, regionobj: region.Region, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:
        assert isinstance(regionobj, region.Region)
        loader = WagstylProfileLoader()

        space = WagstylProfileLoader._choose_space(regionobj)
        if not regionobj.is_leaf:
            leaves_defined_on_space = [
                r for r in regionobj.leaves if r.mapped_in_space(space)
            ]
        else:
            leaves_defined_on_space = [regionobj]

        result = []
        for subregion in leaves_defined_on_space:
            matched_profiles, boundary_depths, coords = loader.match(subregion, space)
            if matched_profiles.shape[0] == 0:
                continue

            # compute array of layer labels for all coefficients in profiles_left
            N = matched_profiles.shape[1]
            prange = np.arange(N)
            layer_labels = 7 - np.array([
                [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
                for b in boundary_depths
            ]).reshape((-1, 200))

            fp = layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities(
                regionname=subregion.name,
                means=[matched_profiles[layer_labels == layer].mean() for layer in range(1, 7)],
                stds=[matched_profiles[layer_labels == layer].std() for layer in range(1, 7)],
            )
            assert fp.matches(subregion)  # to create an assignment result
            fp.datasets = [WagstylProfileLoader.DATASET]
            result.append(fp)

        return result
