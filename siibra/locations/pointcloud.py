# Copyright 2018-2025
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

from . import location, point, boundingbox as _boundingbox

from ..retrieval.requests import HttpRequest
from ..commons import logger
from ..exceptions import SpaceWarpingFailedError, EmptyPointCloudError

from typing import List, Union, Tuple
import numbers
import json
import numpy as np
try:
    from sklearn.cluster import HDBSCAN
    _HAS_HDBSCAN = True
except ImportError:
    import sklearn
    _HAS_HDBSCAN = False
    logger.warning(
        f"HDBSCAN is not available with your version {sklearn.__version__} of sckit-learn."
        "`PointCloud.find_clusters()` will not be available."
    )


def from_points(points: List["point.Point"], newlabels: List[Union[int, float, tuple]] = None) -> "PointCloud":
    """
    Create a PointCloud from an iterable of Points.

    Parameters
    ----------
    points : Iterable[point.Point]
    newlabels: List[int], optional
        Use these labels instead of the original labels of the points.

    Returns
    -------
    PointCloud
    """
    if len(points) == 0:
        raise EmptyPointCloudError("Cannot create a PointCloud without any points.")

    spaces = {p.space for p in points}
    assert len(spaces) == 1, f"PointCloud can only be constructed with points from the same space.\n{spaces}"
    coords, sigmas, labels = zip(*((p.coordinate, p.sigma, p.label) for p in points))
    if all(lb is None for lb in set(labels)):
        labels = None
    return PointCloud(
        coordinates=coords,
        space=next(iter(spaces)),
        sigma_mm=sigmas,
        labels=newlabels or labels
    )


class PointCloud(location.Location):
    """A set of 3D points in the same reference space,
    defined by a list of coordinates."""

    def __init__(
        self,
        coordinates: Union[List[Tuple], np.ndarray],
        space=None,
        sigma_mm: Union[int, float, List[Union[int, float]]] = 0,
        labels: List[Union[int, float, tuple]] = None
    ):
        """
        Construct a 3D point set in the given reference space.

        Parameters
        ----------
        coordinates : array-like, Nx3
            Coordinates in mm of the given space
        space : reference space (id, name, or Space object)
            The reference space
        sigma_mm : float, or list of float
            Optional standard deviation of point locations.
        labels: list of point labels (optional)
        """
        location.Location.__init__(self, space)

        if len(coordinates) == 0:
            raise EmptyPointCloudError(f"Cannot create a {self.__class__.__name__} without any coordinates.")

        self._coordinates = coordinates
        if not isinstance(coordinates, np.ndarray):
            self._coordinates = np.array(self._coordinates).reshape((-1, 3))
        assert len(self._coordinates.shape) == 2
        assert self._coordinates.shape[1] == 3

        if isinstance(sigma_mm, numbers.Number):
            self.sigma_mm = [sigma_mm for _ in range(len(self))]
        else:
            assert len(sigma_mm) == len(self), "The number of coordinate must be equal to the number of sigmas."
            self.sigma_mm = sigma_mm

        if labels is not None:
            assert len(labels) == self._coordinates.shape[0]
        self.labels = labels

    def intersection(self, other: location.Location):
        """Return the subset of points that are inside the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        if not isinstance(other, (point.Point, PointCloud, _boundingbox.BoundingBox)):
            return other.intersection(self)

        if isinstance(other, PointCloud):
            intersecting_points = [p for p in self if p.coordinate in other.coordinates]
        else:
            intersecting_points = [p for p in self if p.intersects(other)]
        if len(intersecting_points) == 0:
            return None
        intersection = from_points(intersecting_points)
        return intersection[0] if len(intersection) == 1 else intersection

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def sigma(self) -> List[Union[int, float]]:
        """The list of sigmas corresponding to the points."""
        return self.sigma_mm

    @property
    def has_constant_sigma(self) -> bool:
        return len(set(self.sigma)) == 1

    def warp(self, space, chunksize=1000):
        """Creates a new point set by warping its points to another space"""
        from ..core.space import Space
        spaceobj = space if isinstance(space, Space) else Space.get_instance(space)
        if spaceobj == self.space:
            return self
        if any(_ not in location.Location.SPACEWARP_IDS for _ in [self.space.id, spaceobj.id]):
            raise SpaceWarpingFailedError(
                f"Cannot convert coordinates between {self.space.id} and {spaceobj.id}"
            )

        src_points = self.as_list()
        tgt_points = []
        N = len(src_points)
        if N > 10e5:
            logger.info(f"Warping {N} points from {self.space.name} to {spaceobj.name} space")
        for i0 in range(0, N, chunksize):

            i1 = min(i0 + chunksize, N)
            data = json.dumps({
                "source_space": location.Location.SPACEWARP_IDS[self.space.id],
                "target_space": location.Location.SPACEWARP_IDS[spaceobj.id],
                "source_points": src_points[i0:i1]
            })
            response = HttpRequest(
                url=f"{location.Location.SPACEWARP_SERVER}/transform-points",
                post=True,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                data=data,
                func=lambda b: json.loads(b.decode()),
            ).data
            tgt_points.extend(list(response["target_points"]))

        # TODO: consider using np.isnan(np.dot(arr, arr)). see https://stackoverflow.com/a/45011547
        if np.any(np.isnan(response['target_points'])):
            raise SpaceWarpingFailedError(f'Warping {str(self)} to {spaceobj.name} resulted in NaN')

        return self.__class__(coordinates=tuple(tgt_points), space=spaceobj, labels=self.labels)

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new PointCloud obtained by transforming the
        coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space (id, name, or Space)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        return self.__class__(
            np.dot(affine, self.homogeneous.T)[:3, :].T,
            space,
            labels=self.labels
        )

    def __getitem__(self, index: int):
        if (abs(index) >= self.__len__()):
            raise IndexError(
                f"pointcloud with {self.__len__()} points "
                f"cannot be accessed with index {index}."
            )
        return point.Point(
            self.coordinates[index, :],
            space=self.space,
            sigma_mm=self.sigma_mm[index],
            label=None if self.labels is None else self.labels[index]
        )

    def __iter__(self):
        """Return an iterator over the coordinate locations."""
        return (
            point.Point(
                self.coordinates[i, :],
                space=self.space,
                sigma_mm=self.sigma_mm[i],
                label=None if self.labels is None else self.labels[i]
            )
            for i in range(len(self))
        )

    def __eq__(self, other: 'PointCloud'):
        if isinstance(other, point.Point):
            return len(self) == 1 and self[0] == other
        if not isinstance(other, PointCloud):
            return False
        return list(self) == list(other)

    def __hash__(self):
        return super().__hash__()

    def __len__(self):
        """The number of points in this PointCloud."""
        return self.coordinates.shape[0]

    def __str__(self):
        return f"Set of {len(self)} points in the {self.boundingbox}"

    @property
    def boundingbox(self):
        """
        Return the bounding box of these points, or None in the
        special case of an empty PointCloud.
        """
        if len(self.coordinates) == 0:
            return None
        coords = self.coordinates
        # TODO this needs a more precise treatment of the sigmas
        sigma_min = max(self.sigma[i] for i in coords.argmin(0))
        sigma_max = max(self.sigma[i] for i in coords.argmax(0))
        return _boundingbox.BoundingBox(
            point1=coords.min(0),
            point2=coords.max(0),
            space=self.space,
            sigma_mm=[sigma_min, sigma_max]
        )

    @property
    def centroid(self):
        return point.Point(self.coordinates.mean(0), space=self.space)

    @property
    def volume(self):
        if len(self) < 2:
            return 0
        else:
            return self.boundingbox.volume

    def as_list(self):
        """Return the point set as a list of 3D tuples."""
        return list(zip(*self.coordinates.T.tolist()))

    @property
    def homogeneous(self):
        """Access the list of 3D point as an Nx4 array of homogeneous coordinates."""
        return np.c_[self.coordinates, np.ones(len(self))]

    def find_clusters(
        self,
        min_fraction: float = 1 / 200,
        max_fraction: float = 1 / 8
    ) -> List[int]:
        """
        Find clusters using HDBSCAN (https://dl.acm.org/doi/10.1145/2733381)
        implementation of scikit-learn (https://dl.acm.org/doi/10.5555/1953048.2078195).

        Parameters
        ----------
        min_fraction: min cluster size as a fraction of total points in the PointCloud
        max_fraction: max cluster size as a fraction of total points in the PointCloud

        Returns
        -------
        List[int]
            Returns the cluster labels found by skilearn.cluster.HDBSCAN.

            Note
            ----
            Replaces the labels of the PointCloud instance with these labels.

        Raises
        ------
        RuntimeError
            If a sklearn version without HDBSCAN is installed.
        """
        if not _HAS_HDBSCAN:
            raise RuntimeError(
                f"HDBSCAN is not available with your version {sklearn.__version__} "
                "of sckit-learn. `PointCloud.find_clusters()` will not be available."
            )
        points = np.array(self.as_list())
        N = points.shape[0]
        clustering = HDBSCAN(
            min_cluster_size=int(N * min_fraction),
            max_cluster_size=int(N * max_fraction),
        )
        if self.labels is not None:
            logger.warning(
                "Existing labels of PointCloud will be overwritten with cluster labels."
            )
        self.labels = clustering.fit_predict(points)
        return self.labels

    @property
    def label_colors(self):
        """ return a color for the given label. """
        if self.labels is None:
            return None
        else:
            try:
                from matplotlib.pyplot import cm as colormaps
            except Exception:
                logger.error("Matplotlib is not available. Label colors is disabled.")
                return None
            return colormaps.rainbow(np.linspace(0, 1, max(self.labels) + 1))
