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

from dataclasses import dataclass
import numpy as np

from . import point, pointcloud, boundingbox


@dataclass
class PolyLine(pointcloud.PointCloud):
    """
    A polyline in 3D, ie. a sequence of connected straight line segments.
    The polyline is represented as an ordered set of points and thus
    based on a PointCloud with explicit interpretation of the coordinate lists's order.
    """
    schema: str = "siibra/attr/loc/polyline"
    closed: bool = False

    def crop(self, voi: boundingbox.BoundingBox):
        """
        Crop the contour with a volume of interest.
        Since the contour might be split from the cropping,
        returns a set of contour segments.
        """
        segments = []

        # set the contour point labels to a linear numbering
        # so we can use them after the intersection to detect splits.
        labelled = pointcloud.LabelledPointCloud(
            coordinates=self.coordinates,
            labels=list(range(len(self))),
            space=self.space
        )
        cropped = labelled.intersection(voi)

        if cropped is not None and not isinstance(cropped, point.Point):
            assert isinstance(cropped, pointcloud.PointCloud)
            # Identifiy contour splits are by discontinuouities ("jumps")
            # of their labels, which denote positions in the original contour
            jumps = np.diff([labelled.labels.index(lb) for lb in cropped.labels])
            splits = [0] + list(np.where(jumps > 1)[0] + 1) + [len(cropped)]
            for i, j in zip(splits[:-1], splits[1:]):
                segments.append(
                    PolyLine(
                        space=cropped.space,
                        coordinates=cropped.coordinates[i:j, :],
                    )
                )

        return segments
