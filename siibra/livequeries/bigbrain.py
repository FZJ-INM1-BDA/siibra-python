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
from ..commons import logger
from ..locations import point, pointset
from ..core import region
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

    def match(self, regionobj: region.Region, space: str = None):
        """
        Retrieve BigBrainIntesityProfiles .

        Parameters
        ----------
        regionobj : region.Region
            Region or parcellation
        space_spec : str, optional
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

    def __init__(self, **kwargs):
        query.LiveQuery.__init__(self, **kwargs)
        self.space_spec = kwargs.pop('space', None)

    def query(self, regionobj: region.Region, **kwargs) -> List[bigbrain_intensity_profile.BigBrainIntensityProfile]:
        assert isinstance(regionobj, region.Region)
        loader = WagstylProfileLoader()

        space = self.space_spec or WagstylProfileLoader._choose_space(regionobj)
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
                result.append(prof)

        return result


class LayerwiseBigBrainIntensityQuery(query.LiveQuery, args=[], FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities):

    def __init__(self, **kwargs):
        query.LiveQuery.__init__(self, **kwargs)
        self.space_spec = kwargs.pop('space', None)

    def query(self, regionobj: region.Region, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:
        assert isinstance(regionobj, region.Region)
        loader = WagstylProfileLoader()

        space = self.space_spec or WagstylProfileLoader._choose_space(regionobj)
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
            result.append(fp)

        return result
