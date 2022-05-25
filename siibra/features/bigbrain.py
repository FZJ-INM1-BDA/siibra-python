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

from .feature import CorticalProfile
from .query import FeatureQuery

from ..commons import logger, QUIET
from ..core.space import PointSet
from ..core.atlas import Atlas
from ..core.parcellation import Parcellation
from ..retrieval import HttpRequest

import numpy as np
import os
import pandas as pd


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
        boundaries: list
    ):
        CorticalProfile.__init__(
            self,
            measuretype="BigBrain cortical intensity profile",
            species=Atlas.get_species_data('human').dict(),
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


class WagstylBigBrainProfileQuery(FeatureQuery):

    _FEATURETYPE = BigBrainIntensityProfile

    REPO = "https://github.com/kwagstyl/cortical_layers_tutorial"
    BRANCH = "main"
    PROFILES_FILE = "data/profiles_left.npy"
    THICKNESSES_FILE = "data/thicknesses_left.npy"
    MESH_FILE = "data/gray_left_327680.surf.gii"

    def __init__(self):

        FeatureQuery.__init__(self)

        # read thicknesses, in mm, and normalize by their last columsn which is the total thickness
        mesh_left = HttpRequest(f"{self.REPO}/raw/{self.BRANCH}/{self.MESH_FILE}").data
        T = HttpRequest(f"{self.REPO}/raw/{self.BRANCH}/{self.THICKNESSES_FILE}").data.T
        total = T[:, :-1].sum(1)
        valid = np.where(total > 0)[0]
        boundary_depths = np.c_[np.zeros_like(valid), (T[valid, :-1] / total[valid, None]).cumsum(1)]
        boundary_depths[:, -1] = 1

        # read profiles with valid thickenss
        req = HttpRequest(
            f"{self.REPO}/raw/{self.BRANCH}/{self.PROFILES_FILE}",
            msg_if_not_cached="First request to BigBrain profiles. Downloading and preprocessing the data now. This may take a little."
        )
        profiles_left = req.data[valid, :]

        vertices = mesh_left.darrays[0].data[valid, :]
        assert vertices.shape[0] == profiles_left.shape[0]

        # we cache the assignments of profiles to regions as well
        with QUIET:
            jubrain = Parcellation.REGISTRY["julich"]
        cachefile = f"{req.cachefile}_{jubrain.key}_assignments.csv"

        if not os.path.isfile(cachefile):

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

            ass.to_csv(cachefile)

        assignments = pd.read_csv(cachefile)

        logger.debug(f"{len(vertices)} BigBrain intensity profiles...")
        depths = np.arange(0, 1, 1 / profiles_left.shape[1])
        for assignment in assignments.itertuples():
            p = BigBrainIntensityProfile(
                regionname=assignment.Region,
                depths=depths,
                values=profiles_left[assignment.Component, :],
                boundaries=boundary_depths[assignment.Component, :]
            )
            self.register(p)
