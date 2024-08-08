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

"""A set of coordinates on a reference space."""

from typing import List, Union, Iterator
import numpy as np
from dataclasses import dataclass, field

from .base import Location
from . import point, boundingbox as _boundingbox


@dataclass
class PointCloud(Location):
    # TODO: reconsider how to implement `labels`.
    # TODO: coordinates should be a numpy array
    schema = "siibra/attr/loc/pointcloud/v0.1"
    coordinates: List[List[float]] = field(default_factory=list, repr=False)
    sigma: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.sigma:
            assert len(self.sigma) == len(self.coordinates)
        else:
            self.sigma = np.zeros(len(self.coordinates), 0).to_list()

    def __iter__(self) -> Iterator[point.Point]:
        """Return an iterator over the coordinate locations."""
        yield self.to_pts()

    def __len__(self):
        """The number of points in this PointVloud."""
        return len(self.coordinates)

    def __eq__(self, other: "PointCloud"):
        if isinstance(other, point.Point):
            return len(self) == 1 and self[0] == other
        if not isinstance(other, PointCloud):
            return False
        return list(self) == list(other)

    def __getitem__(self, index: int):
        if index >= self.__len__():
            raise IndexError(
                f"Index out of range; PointCloud has {self.__len__()} points."
            )
        return point.Point(
            coordinate=self.coordinates[index, :],
            space_id=self.space_id,
            sigma=self.sigma[index],
        )

    @property
    def homogeneous(self):
        return np.c_[self.coordinates, np.ones(len(self.coordinates))]

    def to_pts(self):
        return [
            point.Point(space_id=self.space_id, coordinate=coord)
            for coord in self.coordinates
        ]

    def append(self, pt: "point.Point"):
        self.coordinates.append(pt.coordinate)
        self.sigma.append(pt.sigma)

    def extend(self, points: Union[List["point.Point"], "PointCloud"]):
        coords, sigmas = zip(*((p.coordinate, p.sigma) for p in points))
        self.coordinates.extend(coords)
        self.sigma.extend(sigmas)

    @property
    def boundingbox(self):
        """Return the bounding box of these points."""
        minpoint = np.min(self.coordinates, axis=0)
        maxpoint = np.max(self.coordinates, axis=0)

        # TODO pointcloud sigma is currently broken
        # sigma length does not necessary equal to length of points
        sigma_min = [0]  # max(self.sigma[i] for i in XYZ.argmin(0))
        sigma_max = [0]  # max(self.sigma[i] for i in XYZ.argmax(0))
        return _boundingbox.BoundingBox(
            minpoint=minpoint - max(sigma_min),
            maxpoint=maxpoint + max(sigma_max),
            space_id=self.space_id,
        )


@dataclass
class Contour(PointCloud):
    """
    A PointCloud that represents a contour line. The points making up a Contour
    are assumed to be in an order such that consecutive points are treated to be
    being connected by an edge.
    """

    schema: str = "siibra/attr/loc/contour"
    closed: bool = False
    coordinates: List[float] = field(default_factory=list, repr=False)

    # TODO: crop requires labels and labels needs to be reimplemented or an alternative is required.
    # def crop(self, voi: boundingbox.BoundingBox):
    #     """
    #     Crop the contour with a volume of interest.
    #     Since the contour might be split from the cropping,
    #     returns a set of contour segments.
    #     """
    #     segments = []

    #     # set the contour point labels to a linear numbering
    #     # so we can use them after the intersection to detect splits.
    #     old_labels = self.labels
    #     self.labels = list(range(len(self)))
    #     cropped = self.intersection(voi)

    #     if cropped is not None and not isinstance(cropped, point.Point):
    #         assert isinstance(cropped, pointset.PointSet)
    #         # Identifiy contour splits are by discontinuouities ("jumps")
    #         # of their labels, which denote positions in the original contour
    #         jumps = np.diff([self.labels.index(lb) for lb in cropped.labels])
    #         splits = [0] + list(np.where(jumps > 1)[0] + 1) + [len(cropped)]
    #         for i, j in zip(splits[:-1], splits[1:]):
    #             segments.append(
    #                 self.__class__(cropped.coordinates[i:j, :], space=cropped.space)
    #             )

    #     # reset labels of the input contour points.
    #     self.labels = old_labels

    #     return segments


def from_points(points: List["point.Point"]) -> "PointCloud":
    if len(points) == 0:
        return PointCloud([])
    spaces = {p.space for p in points}
    assert (
        len(spaces) == 1
    ), f"PointCloud can only be constructed with points from the same space.\n{spaces}"

    coords, sigmas = zip(*((p.coordinate, p.sigma) for p in points))
    return PointCloud(coordinates=coords, space=next(iter(spaces)), sigma_mm=sigmas)


# TODO: requires labels
# def find_clusters(self, min_fraction=1 / 200, max_fraction=1 / 8):
    # try:
    #     from sklearn.cluster import HDBSCAN

    #     _HAS_HDBSCAN = True
    # except ImportError:
    #     import sklearn

    #     _HAS_HDBSCAN = False
    #     logger.warning(
    #         f"HDBSCAN is not available with your version {sklearn.__version__} of sckit-learn."
    #         "`PointSet.find_clusters()` will not be avaiable."
    #     )
#     if not _HAS_HDBSCAN:
#         raise RuntimeError(
#             f"HDBSCAN is not available with your version {sklearn.__version__} "
#             "of sckit-learn. `PointSet.find_clusters()` will not be avaiable."
#         )
#     points = np.array(self.as_list())
#     N = points.shape[0]
#     clustering = HDBSCAN(
#         min_cluster_size=int(N * min_fraction),
#         max_cluster_size=int(N * max_fraction),
#     )
#     if self.labels is not None:
#         logger.warning(
#             "Existing labels of PointSet will be overwritten with cluster labels."
#         )
#     self.labels = clustering.fit_predict(points)
#     return self.labels
