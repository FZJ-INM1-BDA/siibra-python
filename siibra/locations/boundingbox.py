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
"""A box defined by two farthest corner coordinates on a specific space."""

from . import point, pointset, location, boundingbox

from ..commons import logger

import hashlib
import numpy as np
from typing import Union
from nibabel import Nifti1Image


class BoundingBox(location.Location):
    """
    A 3D axis-aligned bounding box spanned by two 3D corner points.
    The box does not necessarily store the given points,
    instead it computes the real minimum and maximum points
    from the two corner points.
    """

    def __init__(self, point1, point2, space=None, minsize: float = None, sigma_mm=None):
        """
        Construct a new bounding box spanned by two 3D coordinates
        in the given reference space.

        TODO allow to pass sigma for the points, if tuples

        Parameters
        ----------
        point1 : Point or 3-tuple
            Startpoint given in mm of the given space
        point2 : Point or 3-tuple
            Endpoint given in mm of the given space
        space : reference space (id, name, or Space)
            The reference space
        minsize : float
            Minimum size along each dimension. If not None, the maxpoint will
            be adjusted to match the minimum size, if needed.
        sigma_mm : float, or list of float
            Optional standard deviation of the spanning point locations.
        """
        location.Location.__init__(self, space)
        xyz1 = point.Point.parse(point1)
        xyz2 = point.Point.parse(point2)
        if sigma_mm is None:
            s1, s2 = 0, 0
        elif isinstance(sigma_mm, float):
            s1, s2 = sigma_mm, sigma_mm
        elif isinstance(sigma_mm, list):
            assert len(sigma_mm) == 2
            s1, s2 = sigma_mm
        else:
            raise ValueError(f"Cannot interpret sigma_mm parameter value {sigma_mm} for bounding box")
        self.minpoint = point.Point([min(xyz1[i], xyz2[i]) for i in range(3)], space, sigma_mm=s1)
        self.maxpoint = point.Point([max(xyz1[i], xyz2[i]) for i in range(3)], space, sigma_mm=s2)
        if minsize is not None:
            for d in range(3):
                if self.shape[d] < minsize:
                    self.maxpoint[d] = self.minpoint[d] + minsize

    @property
    def id(self) -> str:
        return hashlib.md5(str(self).encode("utf-8")).hexdigest()

    @property
    def volume(self):
        return np.prod(self.shape)

    @property
    def center(self):
        return self.minpoint + (self.maxpoint - self.minpoint) / 2

    @property
    def shape(self):
        return tuple(
            (self.maxpoint + self.maxpoint.sigma)
            - (self.minpoint - self.minpoint.sigma)
        )

    @property
    def is_planar(self):
        return any(d == 0 for d in self.shape)

    @staticmethod
    def _determine_bounds(A, threshold=0):
        """
        Bounding box of nonzero values in a 3D array.
        https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        """
        x = np.any(A > threshold, axis=(1, 2))
        y = np.any(A > threshold, axis=(0, 2))
        z = np.any(A > threshold, axis=(0, 1))
        nzx, nzy, nzz = [np.where(v) for v in (x, y, z)]
        if any(len(nz[0]) == 0 for nz in [nzx, nzy, nzz]):
            # empty array
            return None
        xmin, xmax = nzx[0][[0, -1]]
        ymin, ymax = nzy[0][[0, -1]]
        zmin, zmax = nzz[0][[0, -1]]
        return np.array([[xmin, xmax + 1], [ymin, ymax + 1], [zmin, zmax + 1], [1, 1]])

    @classmethod
    def from_image(cls, image: Nifti1Image, space, ignore_affine=False, threshold=0):
        """Construct a bounding box from a nifti image"""
        bounds = cls._determine_bounds(image.get_fdata(), threshold=threshold)
        if bounds is None:
            return None
        if ignore_affine:
            target_space = None
        else:
            bounds = np.dot(image.affine, bounds)
            target_space = space
        return BoundingBox(point1=bounds[:3, 0], point2=bounds[:3, 1], space=target_space)

    def __str__(self):
        if self.space is None:
            return (
                f"Bounding box from ({','.join(f'{v:.2f}' for v in self.minpoint)}) mm "
                f"to ({','.join(f'{v:.2f}' for v in self.maxpoint)}) mm"
            )
        else:
            return (
                f"Bounding box from ({','.join(f'{v:.2f}' for v in self.minpoint)}) mm "
                f"to ({','.join(f'{v:.2f}' for v in self.maxpoint)})mm in {self.space.name} space"
            )

    def contains(self, other: location.Location):
        """Returns true if the bounding box contains the given location."""
        if isinstance(other, point.Point):
            return (other >= self.minpoint) and (other <= self.maxpoint)
        elif isinstance(other, pointset.PointSet):
            return all(self.contains(p) for p in other)
        elif isinstance(other, boundingbox.BoundingBox):
            return all([
                other.minpoint >= self.minpoint,
                other.maxpoint <= self.maxpoint
            ])
        elif isinstance(other, Nifti1Image):
            return self.contains(BoundingBox.from_image(other, space=self.space))
        else:
            raise NotImplementedError(
                f"Cannot test containedness of {type(other)} in {self.__class__.__name__}"
            )

    def contained_in(self, other: Union[location.Location, Nifti1Image]):
        if isinstance(other, location.Location):
            return other.contains(self)
        elif isinstance(other, Nifti1Image):
            return self.contained_in(BoundingBox.from_image(other, space=self.space))
        else:
            raise RuntimeError(f"Cannot test containedness of {self} in type {other.__class__}")

    def intersects(self, other: Union[location.Location, Nifti1Image]):
        intersection = self.intersection(other)
        if intersection is None:
            return False
        else:
            return intersection.volume > 0

    def intersection(self, other, dims=[0, 1, 2], threshold=0):
        """Computes the intersection of this bounding box with another one.

        TODO process the sigma values o the points

        Args:
            other (BoundingBox): Another bounding box
            dims (list of int): Dimensions where the intersection should be computed (applies only to bounding boxes)
            Default: all three. Along dimensions not listed, the union is applied instead.
            threshold: optional intensity threshold for intersecting with image mask
        """
        if isinstance(other, Nifti1Image):
            return self._intersect_mask(other, threshold=threshold)
        elif isinstance(other, BoundingBox):
            return self._intersect_bbox(other, dims)
        else:
            raise NotImplementedError(
                f"Intersection of bounding box with {type(other)} not implemented."
            )

    def _intersect_bbox(self, other, dims=[0, 1, 2]):
        warped = other.warp(self.space)

        # Determine the intersecting bounding box by sorting
        # the coordinates of both bounding boxes for each dimension,
        # and fetching the second and third coordinate after sorting.
        # If those belong to a minimum and maximum point,
        # no matter of which bounding box,
        # we have a nonzero intersection in that dimension.
        minpoints = [b.minpoint for b in (self, warped)]
        maxpoints = [b.maxpoint for b in (self, warped)]
        allpoints = minpoints + maxpoints
        result_minpt = []
        result_maxpt = []

        for dim in range(3):

            if dim not in dims:
                # do not intersect in this dimension, so take the union instead
                result_minpt.append(min(p[dim] for p in allpoints))
                result_maxpt.append(max(p[dim] for p in allpoints))
                continue

            A, B = sorted(allpoints, key=lambda P: P[dim])[1:3]
            if (A in maxpoints) or (B in minpoints):
                # no intersection in this dimension
                return None
            else:
                result_minpt.append(A[dim])
                result_maxpt.append(B[dim])

        bbox = BoundingBox(
            point1=point.Point(result_minpt, self.space),
            point2=point.Point(result_maxpt, self.space),
            space=self.space,
        )
        return bbox if bbox.volume > 0 else None

    def _intersect_mask(self, mask, threshold=0):
        """Intersect this bounding box with an image mask. Returns None if they do not intersect.

        TODO process the sigma values o the points

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # nonzero voxel coordinates
        X, Y, Z = np.where(mask.get_fdata() > threshold)
        h = np.ones(len(X))

        # array of homogenous physical nonzero voxel coordinates
        coords = np.dot(mask.affine, np.vstack((X, Y, Z, h)))[:3, :].T
        minpoint = [min(self.minpoint[i], self.maxpoint[i]) for i in range(3)]
        maxpoint = [max(self.minpoint[i], self.maxpoint[i]) for i in range(3)]
        minpt_voxel = np.dot(np.linalg.inv(mask.affine), np.r_[minpoint, 1])[:3]
        voxel_size = np.diff(
            np.dot(
                mask.affine,
                np.vstack((np.r_[minpt_voxel, 1], np.r_[minpt_voxel + 1, 1])).T
            )[:3, :]
        ).squeeze()
        # use voxel size for the test to intersect voxels properly with verythin bounding boxes
        inside = np.logical_and.reduce([
            coords >= minpoint - voxel_size / 2,
            coords <= maxpoint + voxel_size / 2
        ]).min(1)
        XYZ = coords[inside, :3]
        if XYZ.shape[0] == 0:
            return None
        elif XYZ.shape[0] == 1:  # reflect voxel size in resulting point set
            return self._intersect_bbox(
                point
                .Point(XYZ.flatten(), space=self.space, sigma_mm=voxel_size.max())
                .boundingbox
            )
        else:
            return self._intersect_bbox(
                pointset
                .PointSet(XYZ, space=self.space, sigma_mm=voxel_size.max())
                .boundingbox
            )

    def union(self, other):
        """
        Computes the union of this boudning box with another one.

        Args:
            other (BoundingBox): Another bounding box
        """
        warped = other.warp(self.space)
        points = [self.minpoint, self.maxpoint, warped.minpoint, warped.maxpoint]
        return BoundingBox(
            point1=[min(p[i] for p in points) for i in range(3)],
            point2=[max(p[i] for p in points) for i in range(3)],
            space=self.space,
            sigma_mm=[self.minpoint.sigma, self.maxpoint.sigma]
        )

    def clip(self, xyzmax, xyzmin=(0, 0, 0)):
        """
        Returns a new bounding box obtained by clipping at the given maximum coordinate.
        """
        return self.intersection(
            BoundingBox(
                point.Point(xyzmin, self.space), point.Point(xyzmax, self.space), self.space
            ),
            sigma_mm=[self.minpoint.sigma, self.maxpoint.sigma]
        )

    def warp(self, space):
        """Returns a new bounding box obtained by warping the
        min- and maxpoint of this one into the new target space.

        TODO process the sigma values o the points
        """
        from ..core.space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        else:
            try:
                return self.__class__(
                    point1=self.minpoint.warp(spaceobj),
                    point2=self.maxpoint.warp(spaceobj),
                    space=spaceobj,
                )
            except ValueError:
                logger.debug(f"Warping {str(self)} to {spaceobj.name} not successful.")
                return None

    def fetch_regional_map(self):
        """Generate a volumetric binary mask of this
        bounding box in the reference template space."""
        tpl = self.space.get_template().fetch()
        arr = np.zeros(tpl.shape, dtype="uint8")
        bbvox = self.transform(np.linalg.inv(tpl.affine))
        arr[
            int(bbvox.minpoint[0]): int(bbvox.maxpoint[0]),
            int(bbvox.minpoint[1]): int(bbvox.maxpoint[2]),
            int(bbvox.minpoint[2]): int(bbvox.maxpoint[2]),
        ] = 1
        return Nifti1Image(arr, tpl.affine)

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new bounding box obtained by transforming the
        min- and maxpoint of this one with the given affine matrix.

        TODO process the sigma values o the points

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space (str, Space, or None)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        from ..core.space import Space
        spaceobj = Space.get_instance(space)
        return self.__class__(
            point1=self.minpoint.transform(affine, spaceobj),
            point2=self.maxpoint.transform(affine, spaceobj),
            space=space,
            sigma_mm=[self.minpoint.sigma, self.maxpoint.sigma]  # TODO: error propagation
        )

    def shift(self, offset):
        return self.__class__(
            point1=self.minpoint + offset,
            point2=self.maxpoint + offset,
            space=self.space,
            sigma_mm=[self.minpoint.sigma, self.maxpoint.sigma]
        )

    def zoom(self, ratio: float):
        """
        Create a new bounding box by zooming this one around its center.
        """
        c = self.center
        self.maxpoint - c
        return self.__class__(
            point1=(self.minpoint - c) * ratio + c,
            point2=(self.maxpoint - c) * ratio + c,
            space=self.space,
            sigma_mm=[self.minpoint.sigma * ratio, self.maxpoint.sigma * ratio]
        )

    def estimate_affine(self, space):
        """
        Computes an affine transform which approximates
        the nonlinear warping of the eight corner points
        to the desired target space.
        The transform is estimated using a least squares
        solution to A*x = b, where A is the matrix of
        point coefficients in the space of this bounding box,
        and b are the target coefficients in the given space
        after calling the nonlinear warping.
        """

        x0, y0, z0 = self.minpoint
        x1, y1, z1 = self.maxpoint

        # set of 8 corner points in source space
        corners1 = pointset.PointSet(
            [
                (x0, y0, z0),
                (x0, y0, z1),
                (x0, y1, z0),
                (x0, y1, z1),
                (x1, y0, z0),
                (x1, y0, z1),
                (x1, y1, z0),
                (x1, y1, z1)
            ], self.space
        )

        # coefficient matrix from original points
        A = np.hstack(
            [
                [np.kron(np.eye(3), np.r_[tuple(c), 1])]
                for c in corners1
            ]).squeeze()

        # righthand side from warped points
        corners2 = corners1.warp(space)
        b = np.hstack(corners2.as_list())

        # least squares solution
        x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        affine = np.vstack([x.reshape((3, 4)), np.array([0, 0, 0, 1])])

        # test
        errors = []
        for c1, c2 in zip(list(corners1), list(corners2)):
            errors.append(
                np.linalg.norm(
                    np.dot(affine, np.r_[tuple(c1), 1])
                    - np.r_[tuple(c2), 1]
                )
            )
        logger.debug(
            f"Average projection error under linear approximation "
            f"was {np.mean(errors):.2f} pixel"
        )

        return affine

    def __iter__(self):
        """Iterate the min- and maxpoint of this bounding box."""
        return iter((self.minpoint, self.maxpoint))
