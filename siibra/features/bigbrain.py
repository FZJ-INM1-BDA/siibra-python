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

from .feature import CorticalProfile, RegionalFingerprint
from .query import FeatureQuery

from ..registry import REGISTRY
from ..commons import logger, QUIET
from ..core.space import PointSet, Point
from ..core.atlas import Atlas
from ..retrieval import HttpRequest

import numpy as np
import os
import pandas as pd


def load_wagstyl_profiles():

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE = "data/profiles_left.npy"
    THICKNESSES_FILE = "data/thicknesses_left.npy"
    MESH_FILE = "data/gray_left_327680.surf.gii"

    # read thicknesses, in mm, and normalize by their last columsn which is the total thickness
    mesh_left = HttpRequest(f"{REPO}/raw/{BRANCH}/{MESH_FILE}").data
    T = HttpRequest(f"{REPO}/raw/{BRANCH}/{THICKNESSES_FILE}").data.T
    total = T[:, :-1].sum(1)
    valid = np.where(total > 0)[0]
    boundary_depths = np.c_[np.zeros_like(valid), (T[valid, :-1] / total[valid, None]).cumsum(1)]
    boundary_depths[:, -1] = 1

    # read profiles with valid thickenss
    req = HttpRequest(
        f"{REPO}/raw/{BRANCH}/{PROFILES_FILE}",
        msg_if_not_cached="First request to BigBrain profiles. Downloading and preprocessing the data now. This may take a little."
    )
    profiles_left = req.data[valid, :]

    vertices = mesh_left.darrays[0].data[valid, :]
    assert vertices.shape[0] == profiles_left.shape[0]

    # we cache the assignments of profiles to regions as well
    with QUIET:
        jubrain = REGISTRY.Parcellation["julich"]
    assfile = f"{req.cachefile}_{jubrain.key}_assignments.csv"
    ptsfile = assfile.replace('assignments.csv', 'bbcoords.txt')

    if not os.path.isfile(assfile):

        logger.info(f"Warping locations of {len(vertices)} BigBrain profiles to MNI space...")
        pts_bb = PointSet(vertices, space="bigbrain")
        pts_mni = pts_bb.warp("mni152")

        logger.info(f"Assigning {len(vertices)} profile locations to Julich-Brain regions...")
        pmaps = jubrain.get_map(space="mni152", maptype="continuous")
        # keep only the unique matches with maximum probability
        with QUIET:
            ass = (
                pmaps.assign(pts_mni)
                .sort_values("MaxValue", ascending=False)
                .drop_duplicates(["Component"])
            )

        ass.to_csv(assfile)
        np.savetxt(ptsfile, pts_bb.as_list())

    assignments = pd.read_csv(assfile)
    bbcoords = np.loadtxt(ptsfile)

    return profiles_left, boundary_depths, assignments, bbcoords


class BigBrainIntensityProfile(CorticalProfile):

    DESCRIPTION = (
        "Cortical profiles of BigBrain staining intensities computed by Konrad Wagstyl, "
        "as described in the publication 'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678'."
        "Taken from the tutorial at https://github.com/kwagstyl/cortical_layers_tutorial "
        "and assigned to cytoarchitectonic regions of Julich-Brain."
    )

    def __init__(
        self,
        regionname: str,
        depths: list,
        values: list,
        boundaries: list,
        location: Point
    ):
        CorticalProfile.__init__(
            self,
            measuretype="BigBrain cortical intensity profile",
            species=Atlas.get_species_id('human'),
            regionname=regionname,
            description=self.DESCRIPTION,
            unit="staining intensity",
            depths=depths,
            values=values,
            boundary_positions={
                b: boundaries[b[0]]
                for b in CorticalProfile.BOUNDARIES
            }
        )
        self.location = location


class WagstylBigBrainProfileQuery(FeatureQuery, parameters=[]):

    _FEATURETYPE = BigBrainIntensityProfile

    def __init__(self):

        FeatureQuery.__init__(self)
        profiles_left, boundary_depths, assignments, bbcoords = load_wagstyl_profiles()
        logger.debug(f"{profiles_left.shape[0]} BigBrain intensity profiles...")
        depths = np.arange(0, 1, 1 / profiles_left.shape[1])
        for assignment in assignments.itertuples():
            idx = assignment.Component
            p = BigBrainIntensityProfile(
                regionname=assignment.Region,
                depths=depths,
                values=profiles_left[idx, :],
                boundaries=boundary_depths[idx, :],
                location=Point(bbcoords[idx, :], REGISTRY.Space['bigbrain'])
            )
            self.add_feature(p)


class BigBrainIntensityFingerprint(RegionalFingerprint):

    DESCRIPTION = (
        "Layerwise averages and standard deviations of of BigBrain staining intensities "
        "computed by Konrad Wagstyl, as described in the publication "
        "'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678'."
        "Taken from the tutorial at https://github.com/kwagstyl/cortical_layers_tutorial "
        "and assigned to cytoarchitectonic regions of Julich-Brain."
    )

    def __init__(
        self,
        regionname: str,
        means: list,
        stds: list,
    ):
        RegionalFingerprint.__init__(
            self,
            measuretype="Layerwise BigBrain intensities",
            species=Atlas.get_species_id('human'),
            regionname=regionname,
            description=self.DESCRIPTION,
            unit="staining intensity",
            means=means,
            stds=stds,
            labels=list(CorticalProfile.LAYERS.values())[1: -1],
        )


class WagstylBigBrainIntensityFingerprintQuery(FeatureQuery, parameters=[]):

    _FEATURETYPE = BigBrainIntensityFingerprint

    def __init__(self):

        FeatureQuery.__init__(self)
        profiles_left, boundary_positions, assignments, bbcoords = load_wagstyl_profiles()

        # compute array of layer labels for all coefficients in profiles_left
        N = profiles_left.shape[1]
        prange = np.arange(N)
        labels = 7 - np.array([
            [np.array([[(prange < T) * 1] for i, T in enumerate((b * N).astype('int'))]).squeeze().sum(0)]
            for b in boundary_positions
        ]).squeeze()

        for region in assignments.Region.unique():
            indices = list(assignments[assignments.Region == region].Component)
            region_profiles = profiles_left[indices, :]
            region_labels = labels[indices, :]
            p = BigBrainIntensityFingerprint(
                regionname=region,
                means=[region_profiles[region_labels == _].mean() for _ in range(1, 7)],
                stds=[region_profiles[region_labels == _].std() for _ in range(1, 7)],
            )
            self.add_feature(p)
