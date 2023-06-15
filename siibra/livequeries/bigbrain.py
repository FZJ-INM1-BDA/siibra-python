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
from ..core import space, region
from ..retrieval import requests, cache

import numpy as np
from typing import List
from os import path


class WagstylProfileLoader:

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE_LEFT = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-26d25994-634c-40af-b88f-2a36e8e1d508/profiles/profiles_left.txt"
    PROFILES_FILE_RIGHT = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-26d25994-634c-40af-b88f-2a36e8e1d508/profiles/profiles_right.txt"
    THICKNESSES_FILE = "data/thicknesses_left.npy"
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
        return np.arange(0, 1, 1 / self._profiles.shape[1])

    @classmethod
    def _load(cls):
        # read thicknesses, in mm, and normalize by their last column which is the total thickness
        thickness_left = requests.HttpRequest(f"{cls.REPO}/raw/{cls.BRANCH}/{cls.THICKNESSES_FILE}").data.T
        thickness_right = np.zeros(shape=thickness_left.shape)  # TODO: replace with thickness data for te right hemisphere
        thickness = np.concatenate((thickness_left, thickness_right))
        total_thickness = thickness[:, :-1].sum(1)
        valid = np.where(total_thickness > 0)[0]
        cls._boundary_depths = np.c_[np.zeros_like(valid), (thickness[valid, :-1] / total_thickness[valid, None]).cumsum(1)]
        cls._boundary_depths[:, -1] = 1  # account for float calculation errors

        # read profiles with valid thickenss
        profile_left_url = f"{cls.PROFILES_FILE_LEFT}"
        profile_right_url = f"{cls.PROFILES_FILE_RIGHT}"
        if not all(
            path.exists(cache.CACHE.build_filename(url))
            for url in [profile_left_url, profile_right_url]
        ):
            logger.info(
                "First request to BigBrain profiles. "
                "Downloading and preprocessing the data now. "
                "This may take a little."
            )
        profiles_l = requests.HttpRequest(profile_left_url).data.to_numpy()
        profiles_r = requests.HttpRequest(profile_right_url).data.to_numpy()
        cls._profiles = np.concatenate((profiles_l, profiles_r))[valid, :]

        # read mesh vertices
        mesh_left = requests.HttpRequest(f"{cls.BASEURL}/{cls.MESH_FILE_LEFT}").data
        mesh_right = requests.HttpRequest(f"{cls.BASEURL}/{cls.MESH_FILE_RIGHT}").data
        mesh_vertices = np.concatenate((mesh_left.darrays[0].data, mesh_right.darrays[0].data))
        cls._vertices = mesh_vertices[valid, :]

        logger.debug(f"{cls._profiles.shape[0]} BigBrain intensity profiles.")
        assert cls._vertices.shape[0] == cls._profiles.shape[0]

    def __len__(self):
        return self._vertices.shape[0]

    def match(self, regionobj: region.Region):
        assert isinstance(regionobj, region.Region)
        logger.debug(f"Matching locations of {len(self)} BigBrain profiles to {regionobj}")

        supported_spaces = [s for s in regionobj.supported_spaces if s.matches('bigbrain')] or regionobj.supported_spaces
        for spaceobj in supported_spaces:
            if spaceobj.provides_image:
                try:
                    mask = regionobj.fetch_regional_map(space=spaceobj, maptype="labelled")
                except RuntimeError:
                    continue
                logger.info(f"Assigning {len(self)} profile locations to {regionobj} in {spaceobj}...")
                voxels = (
                    pointset.PointSet(self._vertices, space="bigbrain")
                    .warp(spaceobj)
                    .transform(np.linalg.inv(mask.affine), space=None)
                )
                arr = np.asanyarray(mask.dataobj)
                XYZ = np.array(voxels.as_list()).astype('int')
                X, Y, Z = np.split(
                    XYZ[np.all((XYZ < arr.shape) & (XYZ > 0), axis=1), :],
                    3, axis=1
                )
                inside = np.where(arr[X, Y, Z] > 0)[0]
                break
        else:
            raise RuntimeError(f"Could not filter big brain profiles by {regionobj}")

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

        features = []
        for subregion in regionobj.leaves:
            matched_profiles, boundary_depths, coords = loader.match(subregion)
            if len(matched_profiles) == 0:
                continue
            region_profile = bigbrain_intensity_profile.BigBrainIntensityProfile(
                regionname=subregion.name,
                coords=coords,
                depths=loader.profile_labels,
                values=matched_profiles,
                boundary_depths=boundary_depths,
            )
            # assert prof.matches(subregion)  # disabled, this is too slow for the many featuresvim
            features.append(region_profile)

        return features


class LayerwiseBigBrainIntensityQuery(query.LiveQuery, args=[], FeatureType=layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities):

    def __init__(self):
        query.LiveQuery.__init__(self)

    def query(self, regionobj: region.Region, **kwargs) -> List[layerwise_bigbrain_intensities.LayerwiseBigBrainIntensities]:
        assert isinstance(regionobj, region.Region)
        loader = WagstylProfileLoader()

        result = []
        for subregion in regionobj.leaves:
            matched_profiles, boundary_depths, coords = loader.match(subregion)
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
