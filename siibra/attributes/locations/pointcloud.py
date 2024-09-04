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

from typing import List, Union, Iterator, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, replace
if TYPE_CHECKING:
    from ..dataproviders.volume.image import ImageProvider

from .base import Location
from . import point, boundingbox as _boundingbox
from ...commons.logger import logger

import numpy as np
from skimage import feature as skimage_feature
try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except ImportError:
    import sklearn
    _HAS_HDBSCAN = False
    logger.warning(
        f"HDBSCAN is not available with your version {sklearn.__version__} of sckit-learn."
        "`PointSet.find_clusters()` will not be avaiable."
    )


@dataclass
class PointCloud(Location):
    # TODO: reconsider how to implement `labels`.
    schema = "siibra/attr/loc/pointcloud/v0.1"
    coordinates: List[Tuple[float]] = field(default_factory=list, repr=False)
    sigma: List[float] = field(default_factory=list, repr=False)
    labels: List[Union[int, float, str]] = field(default_factory=list, repr=False)

    @staticmethod
    def _parse_values(
        coordinates: Union[List[Tuple[float]], np.ndarray],
        sigma: Union[List, np.ndarray] = None,
        labels: List[Union[int, float, str]] = None
    ) -> Tuple[List[Tuple[float]], List[float], List[Union[int, float, str]]]:
        if sigma:
            assert len(sigma) == len(coordinates)
        else:
            sigma = list(np.zeros(len(coordinates)))

        if labels:
            assert len(labels) == len(coordinates)
        else:
            labels = [None for _ in coordinates]

        if isinstance(sigma, (tuple, np.ndarray)):
            sigma = list(sigma)
        if isinstance(coordinates, np.ndarray):
            coordinates = coordinates.tolist()
        if isinstance(labels, (tuple, np.ndarray)):
            labels = list(labels)
        if isinstance(coordinates, (list, tuple)):
            coordinates = list(tuple(float(c) for c in coord) for coord in coordinates)

        return coordinates, sigma, labels

    def __post_init__(self):
        self.coordinates, self.sigma, self.labels = self._parse_values(self.coordinates, self.sigma, self.labels)

    def __iter__(self) -> Iterator[point.Point]:
        """Return an iterator over the coordinate locations."""
        yield from self.to_points()

    def __len__(self):
        """The number of points in this PointCloud."""
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
            coordinate=self.coordinates[index],
            space_id=self.space_id,
            sigma=self.sigma[index],
            label=self.labels[index]
        )

    @property
    def homogeneous(self) -> np.ndarray:
        return np.c_[self.coordinates, np.ones(len(self.coordinates))]

    def to_points(self) -> List[point.Point]:
        return [
            point.Point(space_id=self.space_id, coordinate=coord, sigma=sigma)
            for coord, sigma in zip(self.coordinates, self.sigma)
        ]

    def to_ndarray(self) -> np.ndarray:
        """Return the coordinates as an numpy array"""
        return np.asarray(self.coordinates)

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
            minpoint=(minpoint - max(sigma_min)).tolist(),
            maxpoint=(maxpoint + max(sigma_max)).tolist(),
            space_id=self.space_id,
        )

    def find_clusters(self, min_fraction=1 / 200, max_fraction=1 / 8):
        if not _HAS_HDBSCAN:
            raise RuntimeError(
                f"HDBSCAN is not available with your version {sklearn.__version__} "
                "of sckit-learn. `PointSet.find_clusters()` will not be avaiable."
            )
        points = np.array(self.coordinates)
        N = points.shape[0]
        clustering = HDBSCAN(
            min_cluster_size=int(N * min_fraction),
            max_cluster_size=int(N * max_fraction),
        )
        result = clustering.fit_predict(points)
        return replace(self, labels=result.tolist())

    @staticmethod
    def from_points(points: List["point.Point"]) -> "PointCloud":
        if len(points) == 0:
            return PointCloud(coordinates=(), sigma=(), space_id=None)
        spaces = {p.space for p in points}
        assert (
            len(spaces) == 1
        ), f"PointCloud can only be constructed with points from the same space.\n{spaces}"

        coords, sigmas = zip(*((p.coordinate, p.sigma) for p in points))
        return PointCloud(coordinates=coords, space_id=next(iter(spaces)).ID, sigma=sigmas)


def sample_from_image(provider: "ImageProvider", num_points: int, sample_size: int = 100, e: float = 1, sigma=None, invert=False, **kwargs):
    """
    Draw samples from the volume, by interpreting its values as an
    unnormalized empirical probability distribtution.
    Any keyword arguments are passed over to fetch()
    """
    image = provider.get_data(**kwargs)
    array = np.asanyarray(image.dataobj)
    samples = []
    P = (array - array.min()) / (array.max() - array.min())
    if invert:
        P = 1 - P
    P = P**e
    while True:
        pts = (np.random.rand(sample_size, 3) * max(P.shape))
        inside = np.all(pts < P.shape, axis=1)
        Y, X, Z = np.split(pts[inside, :].astype('int'), 3, axis=1)
        T = np.random.rand(1)
        choice = np.where(P[Y, X, Z] >= T)[0]
        samples.extend(list(pts[inside, :][choice, :]))
        if len(samples) > num_points:
            break
    voxels = PointCloud(
        coordinates=np.random.permutation(samples)[:num_points, :],
        space_id=None
    )
    result = voxels.transform(image.affine, space_id=provider.space_id)
    result.sigma = [sigma for _ in result]
    return result


def peaks_from_image(provider: "ImageProvider", mindist=5, sigma=0, **kwargs):
    """
    Find local peaks in the volume.
    Additional keyword arguments are passed over to fetch()
    """
    img = provider.get_data(**kwargs)
    array = np.asanyarray(img.dataobj)
    voxels = PointCloud(
        coordinates=skimage_feature.peak_local_max(array, min_distance=mindist),
        space_id=None
    )
    result = voxels.transform(img.affine, space_id=provider.space_id)
    result.sigma = [sigma for _ in result]
    return result


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
