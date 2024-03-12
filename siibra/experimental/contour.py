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

from ..locations import point, pointset, boundingbox

import numpy as np


class Contour(pointset.PointSet):
    """
    A PointSet that represents a contour line.
    The only difference is that the point order is relevant,
    and consecutive points are thought as being connected by an edge.

    In fact, PointSet assumes order as well, but no connections between points.
    """

    def __init__(self, coordinates, space=None, sigma_mm=0, labels: list = None):
        pointset.PointSet.__init__(self, coordinates, space, sigma_mm, labels)

    def crop(self, voi: boundingbox.BoundingBox):
        """
        Crop the contour with a volume of interest.
        Since the contour might be split from the cropping,
        returns a set of contour segments.
        """
        segments = []

        # set the contour point labels to a linear numbering
        # so we can use them after the intersection to detect splits.
        old_labels = self.labels
        self.labels = list(range(len(self)))
        cropped = self.intersection(voi)

        if cropped is not None and not isinstance(cropped, point.Point):
            assert isinstance(cropped, pointset.PointSet)
            # Identifiy contour splits are by discontinuouities ("jumps")
            # of their labels, which denote positions in the original contour
            jumps = np.diff([self.labels.index(lb) for lb in cropped.labels])
            splits = [0] + list(np.where(jumps > 1)[0] + 1) + [len(cropped)]
            for i, j in zip(splits[:-1], splits[1:]):
                segments.append(
                    self.__class__(cropped.coordinates[i:j, :], space=cropped.space)
                )

        # reset labels of the input contour points.
        self.labels = old_labels

        return segments
