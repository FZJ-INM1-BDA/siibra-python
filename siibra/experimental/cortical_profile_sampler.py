# Copyright 2018-2024
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

from . import contour
from ..locations import point
from ..core import parcellation

import numpy as np


class CorticalProfileSampler:
    """Samples cortical profiles from the cortical layer maps."""

    def __init__(self):
        self.layermap = parcellation.Parcellation.get_instance(
            "cortical layers"
        ).get_map(space="bigbrain", maptype="labelled")

    def query(self, query_point: point.Point):
        q = query_point.warp(self.layermap.space)
        smallest_dist = np.inf
        best_match = None
        for layername in self.layermap.regions:
            vertices = self.layermap.fetch(region=layername, format="mesh")["verts"]
            dists = np.sqrt(((vertices - q.coordinate) ** 2).sum(1))
            best = np.argmin(dists)
            if dists[best] < smallest_dist:
                best_match = (layername, best)
                smallest_dist = dists[best]

        best_vertex = best_match[1]
        hemisphere = "left" if "left" in best_match[0] else "right"
        print(f"Best match is vertex #{best_match[1]} in {best_match[0]}.")

        profile = [
            (_, self.layermap.fetch(region=_, format="mesh")["verts"][best_vertex])
            for _ in self.layermap.regions
            if hemisphere in _
        ]

        return contour.Contour(
            [p[1] for p in profile],
            space=self.layermap.space,
            labels=[p[0] for p in profile],
        )
