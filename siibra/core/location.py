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


from ..commons import logger
from ..retrieval import HttpRequest

import hashlib
import re
import numpy as np
from abc import ABC, abstractmethod
from nibabel import Nifti1Image
from nibabel.affines import apply_affine
import json
from urllib.parse import quote
import numbers


# The id of BigBrain reference space
BIGBRAIN_ID = "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588"


# backend for transforming coordinates between spaces
SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"


# lookup of space identifiers to be used by SPACEWARP_SERVER
SPACEWARP_IDS = {
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2": "MNI 152 ICBM 2009c Nonlinear Asymmetric",
    "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992": "MNI Colin 27",
    "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588": "Big Brain (Histology)",
}


class Location(ABC):
    """
    Abstract base class for locations in a given reference space.
    """

    def __init__(self, space):
        from .space import Space
        self.space = Space.get_instance(space)

    @abstractmethod
    def intersection(self, mask: Nifti1Image) -> bool:
        """All subclasses of Location must implement intersection, as it is required by SpatialFeature._test_mask()
        """
        pass

    @abstractmethod
    def intersects(self, mask: Nifti1Image):
        """
        Verifies wether this 3D location intersects the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        pass

    @abstractmethod
    def warp(self, space):
        """Generates a new location by warping the
        current one into another reference space."""
        pass

    @abstractmethod
    def transform(self, affine: np.ndarray, space=None):
        """Returns a new location obtained by transforming the
        reference coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space (id, name, or Space)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """To be implemented in derived classes to return an iterator
        over the coordinates associated with the location."""
        pass

    def __str__(self):
        if self.space is None:
            return (
                f"{self.__class__.__name__} "
                f"[{','.join(str(l) for l in iter(self))}]"
            )
        else:
            return (
                f"{self.__class__.__name__} in {self.space.name} "
                f"[{','.join(str(l) for l in iter(self))}]"
            )


class WholeBrain(Location):
    """
    Trivial location class for formally representing
    location in a particular reference space, which
    is not further specified.
    """

    def intersection(self, mask: Nifti1Image) -> bool:
        """
        Required for abstract class Location
        """
        return True

    def __init__(self, space=None):
        Location.__init__(self, space)

    def intersects(self, mask: Nifti1Image):
        """Always true for whole brain features"""
        return True

    def warp(self, space):
        """Generates a new whole brain location
        in another reference space."""
        return self.__class__(space)

    def transform(self, affine: np.ndarray, space=None):
        """Does nothing."""
        pass

    def __iter__(self):
        """To be implemented in derived classes to return an iterator
        over the coordinates associated with the location."""
        yield from ()

    def __str__(self):
        return f"{self.__class__.__name__} in {self.space.name}"


class Point(Location):
    """A single 3D point in reference space."""

    @staticmethod
    def parse(spec, unit="mm"):
        """Converts a 3D coordinate specification into a 3D tuple of floats.

        Parameters
        ----------
        spec : Any of str, tuple(float,float,float)
            For string specifications, comma separation with decimal points are expected.
        unit : str
            specification of the unit (only 'mm' supported so far)

        Returns
        -------
        tuple(float,float,float)
        """
        if unit != "mm":
            raise NotImplementedError(
                "Coordinate parsing from strings is only supported for mm specifications so far."
            )
        if isinstance(spec, str):
            pat = r"([-\d\.]*)" + unit
            digits = re.findall(pat, spec)
            if len(digits) == 3:
                return tuple(float(d) for d in digits)
        elif isinstance(spec, (tuple, list)) and len(spec) in [3, 4]:
            if len(spec) == 4:
                assert spec[3] == 1
            return tuple(float(v) for v in spec[:3])
        elif isinstance(spec, np.ndarray) and spec.size == 3:
            return tuple(float(v) for v in spec[:3])
        elif isinstance(spec, Point):
            return spec.coordinate

        raise ValueError(
            f"Cannot decode the specification {spec} (type {type(spec)}) to create a point."
        )

    def __hash__(self):
        return sum(map(hash, (self.coordinate, self.sigma, self.space.key)))

    def __init__(self, coordinatespec, space=None, sigma_mm: float = 0.0):
        """
        Construct a new 3D point set in the given reference space.

        Parameters
        ----------
        coordinate : 3-tuple of int/float, or string specification
            Coordinate in mm of the given space
        space : reference space specification (id, object, or name)
            The reference space
        sigma_mm : float
            Optional location uncertainy of the point
            (will be intrepreded as the isotropic standard deviation of location)
        """
        Location.__init__(self, space)
        self.coordinate = Point.parse(coordinatespec)
        self.sigma = sigma_mm
        if isinstance(coordinatespec, Point):
            assert coordinatespec.sigma == sigma_mm
            assert coordinatespec.space == space

    @property
    def homogeneous(self):
        """The homogenous coordinate of this point as a 4-tuple,
        obtained by appending '1' to the original 3-tuple."""
        return self.coordinate + (1,)

    def intersection(self, mask: Nifti1Image):
        if self.intersects(mask):
            return self
        else:
            return None

    def intersects(self, mask: Nifti1Image):
        """Returns true if this point lies in the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # transform physical coordinates to voxel coordinates for the query
        def check(mask, c):
            voxel = (apply_affine(np.linalg.inv(mask.affine), c) + 0.5).astype(int)
            if np.any(voxel >= mask.dataobj.shape):
                return False
            if np.any(voxel < 0):
                return False
            if mask.dataobj[voxel[0], voxel[1], voxel[2]] == 0:
                return False
            return True

        if mask.ndim == 4:
            return any(
                check(mask.slicer[:, :, :, i], self.coordinate)
                for i in range(mask.shape[3])
            )
        else:
            return check(mask, self.coordinate)

    def contained_in(self, mask: Nifti1Image):
        return self.intersects(mask)

    def contains(self, mask: Nifti1Image):
        return False

    def warp(self, space):
        """Creates a new point by warping this point to another space"""
        from .space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        if any(_ not in SPACEWARP_IDS for _ in [self.space.id, spaceobj.id]):
            raise ValueError(
                f"Cannot convert coordinates between {self.space.id} and {spaceobj.id}"
            )
        url = "{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}".format(
            server=SPACEWARP_SERVER,
            src=quote(SPACEWARP_IDS[self.space.id]),
            tgt=quote(SPACEWARP_IDS[space.id]),
            x=self.coordinate[0],
            y=self.coordinate[1],
            z=self.coordinate[2],
        )
        response = HttpRequest(url, lambda b: json.loads(b.decode())).get()
        return self.__class__(
            coordinatespec=tuple(response["target_point"]), space=spaceobj.id
        )

    def __sub__(self, other):
        """Substract the coordinates of two points to get
        a new point representing the offset vector. Alternatively,
        subtract an integer from the all coordinates of this point
        to create a new one."""
        if isinstance(other, numbers.Number):
            return Point([c - other for c in self.coordinate], self.space)

        assert self.space == other.space
        return Point(
            [self.coordinate[i] - other.coordinate[i] for i in range(3)], self.space
        )

    def __lt__(self, other):
        o = other if self.space is None else other.warp(self.space)
        return all(self[i] < o[i] for i in range(3))

    def __gt__(self, other):
        o = other if self.space_id is None else other.warp(self.space)
        return all(self[i] > o[i] for i in range(3))

    def __eq__(self, other):
        o = other if self.space is None else other.warp(self.space)
        return all(self[i] == o[i] for i in range(3))

    def __le__(self, other):
        return not self > other

    def __ge__(self, other):
        return not self < other

    def __add__(self, other):
        """Add the coordinates of two points to get
        a new point representing."""
        if isinstance(other, numbers.Number):
            return Point([c + other for c in self.coordinate], self.space)
        assert self.space == other.space
        return Point(
            [self.coordinate[i] + other.coordinate[i] for i in range(3)], self.space
        )

    def __truediv__(self, number):
        """Return a new point with divided
        coordinates in the same space."""
        return Point(np.array(self.coordinate) / float(number), self.space)

    def __mult__(self, number):
        """Return a new point with multiplied
        coordinates in the same space."""

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new Point obtained by transforming the
        coordinate of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space, (str, Space, or None))
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        from .space import Space
        spaceobj = Space.get_instance(space)
        x, y, z, h = np.dot(affine, self.homogeneous)
        if h != 1:
            logger.warning(f"Homogeneous coordinate is not one: {h}")
        return self.__class__((x / h, y / h, z / h), spaceobj)

    def get_enclosing_cube(self, width_mm):
        """
        Create a bounding box centered around this point with the given width.
        """
        offset = width_mm / 2
        return BoundingBox(
            point1=self - offset,
            point2=self + offset,
            space=self.space,
        )

    def __iter__(self):
        """Return an iterator over the location,
        so the Point can be easily cast to list or tuple."""
        return iter(self.coordinate)

    def __setitem__(self, index, value):
        """Write access to the coefficients of this point."""
        assert 0 <= index < 3
        values = list(self.coordinate)
        values[index] = value
        self.coordinate = tuple(values)

    def __getitem__(self, index):
        """Index access to the coefficients of this point."""
        assert 0 <= index < 3
        return self.coordinate[index]

    def bigbrain_section(self):
        """
        Estimate the histological section number of BigBrain
        which corresponds to this point. If the point is given
        in another space, a warping to BigBrain space will be tried.
        """
        if self.space.id == BIGBRAIN_ID:
            coronal_position = self[1]
        else:
            try:
                bigbrain_point = self.warp("bigbrain")
                coronal_position = bigbrain_point[1]
            except Exception:
                raise RuntimeError(
                    "BigBrain section numbers can only be determined "
                    "for points in BigBrain space, but the given point "
                    f"is given in '{self.space.name}' and could not "
                    "be converted."
                )
        return int((coronal_position + 70.0) / 0.02 + 1.5)

    @property
    def id(self) -> str:
        return hashlib.md5(
            f"{self.space.id}{','.join(str(val) for val in self)}".encode("utf-8")
        ).hexdigest()


class PointSet(Location):
    """A set of 3D points in the same reference space,
    defined by a list of coordinates."""

    def __init__(self, coordinates, space=None, sigma_mm=0):
        """
        Construct a 3D point set in the given reference space.

        Parameters
        ----------
        coordinates : list of Point, 3-tuples or string specs
            Coordinates in mm of the given space
        space : reference space (id, name, or Space object)
            The reference space
        sigma_mm : float, or list of float
            Optional standard deviation of point locations.
        """
        Location.__init__(self, space)
        if isinstance(sigma_mm, numbers.Number):
            self.points = [Point(c, self.space, sigma_mm) for c in coordinates]
        else:
            self.points = [
                Point(c, self.space, s) for c, s in zip(coordinates, sigma_mm)
            ]

    def intersection(self, mask: Nifti1Image):
        """Return the subset of points that are inside the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        inside = [p for p in self if p.intersects(mask)]
        if len(inside) == 0:
            return None
        elif len(inside) == 1:
            return inside[0]
        else:
            return PointSet(
                [p.coordinate for p in inside],
                space=self.space,
                sigma_mm=[p.sigma for p in inside],
            )

    def intersects(self, mask: Nifti1Image):
        return len(self.intersection(mask)) > 0

    @property
    def sigma(self):
        return [p.sigma for p in self]

    @property
    def has_constant_sigma(self):
        return len(set(self.sigma)) == 1

    def warp(self, space):
        """Creates a new point set by warping its points to another space"""
        from .space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        if any(_ not in SPACEWARP_IDS for _ in [self.space.id, spaceobj.id]):
            raise ValueError(
                f"Cannot convert coordinates between {self.space.id} and {spaceobj.id}"
            )

        data = json.dumps({
            "source_space": SPACEWARP_IDS[self.space.id],
            "target_space": SPACEWARP_IDS[spaceobj.id],
            "source_points": self.as_list()
        })

        response = HttpRequest(
            url=f"{SPACEWARP_SERVER}/transform-points",
            post=True,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            data=data,
            func=lambda b: json.loads(b.decode()),
        ).data

        return self.__class__(
            coordinates=tuple(response["target_points"]), space=spaceobj
        )

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new PointSet obtained by transforming the
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
            [c.transform(affine, space) for c in self.points], space
        )

    def __getitem__(self, index: int):
        if (index >= self.__len__()) or (index < 0):
            raise IndexError(
                f"Pointset has only {self.__len__()} points, "
                f"but index of {index} was requested."
            )
        else:
            return self.points[index]

    def __iter__(self):
        """Return an iterator over the coordinate locations."""
        return iter(self.points)

    def __len__(self):
        """The number of points in this PointSet."""
        return len(self.points)

    def __str__(self):
        return f"Set of points {self.space.name}: " + ", ".join(
            f"({','.join(str(v) for v in p)})" for p in self
        )

    @property
    def boundingbox(self):
        """Return the bounding box of these points.

        TODO inherit sigma of the min and max points
        """
        XYZ = self.homogeneous[:, :3]
        return BoundingBox(
            point1=XYZ.min(0),
            point2=XYZ.max(0),
            space=self.space,
        )

    @property
    def centroid(self):
        return Point(self.homogeneous[:, :3].mean(0), space=self.space)

    @property
    def volume(self):
        if len(self) < 2:
            return 0
        else:
            return self.boundingbox.volume

    def as_list(self):
        """Return the point set as a list of 3D tuples."""
        return [tuple(p) for p in self]

    @property
    def homogeneous(self):
        """Access the list of 3D point as an Nx4 array of homogeneous coorindates."""
        return np.array([c.homogeneous for c in self.points]).reshape((-1, 4))


class BoundingBox(Location):
    """
    A 3D axis-aligned bounding box spanned by two 3D corner points.
    The box does not necessarily store the given points,
    instead it computes the real minimum and maximum points
    from the two corner points.
    """

    def __init__(self, point1, point2, space=None, minsize: float = None):
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
        """
        Location.__init__(self, space)
        xyz1 = Point.parse(point1)
        xyz2 = Point.parse(point2)
        self.minpoint = Point([min(xyz1[i], xyz2[i]) for i in range(3)], space)
        self.maxpoint = Point([max(xyz1[i], xyz2[i]) for i in range(3)], space)
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
        return tuple(self.maxpoint - self.minpoint)

    @property
    def is_planar(self):
        return any(d == 0 for d in self.shape)

    @staticmethod
    def _determine_bounds(A):
        """
        Bounding box of nonzero values in a 3D array.
        https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        """
        x = np.any(A, axis=(1, 2))
        y = np.any(A, axis=(0, 2))
        z = np.any(A, axis=(0, 1))
        nzx, nzy, nzz = [np.where(v) for v in (x, y, z)]
        if any(len(nz[0]) == 0 for nz in [nzx, nzy, nzz]):
            # empty array
            return None
        xmin, xmax = nzx[0][[0, -1]]
        ymin, ymax = nzy[0][[0, -1]]
        zmin, zmax = nzz[0][[0, -1]]
        return np.array([[xmin, xmax + 1], [ymin, ymax + 1], [zmin, zmax + 1], [1, 1]])

    @classmethod
    def from_image(cls, image: Nifti1Image, space, ignore_affine=False):
        """Construct a bounding box from a nifti image"""
        bounds = cls._determine_bounds(image.get_fdata())
        if bounds is None:
            return None
        if ignore_affine:
            target_space = None
        else:
            bounds = np.dot(image.affine, bounds)
            target_space = space
        return cls(point1=bounds[:3, 0], point2=bounds[:3, 1], space=target_space)

    def __str__(self):
        if self.space is None:
            return f"Bounding box {tuple(self.minpoint)} -> {tuple(self.maxpoint)}"
        else:
            return f"Bounding box from {tuple(self.minpoint)}mm to {tuple(self.maxpoint)}mm in {self.space.name} space"

    def contains(self, other: Location):
        """Returns true if the bounding box contains the given location."""
        if isinstance(other, Point):
            return (other >= self.minpoint) and (other <= self.maxpoint)
        elif isinstance(other, PointSet):
            return all(self.contains(p) for p in other)
        elif isinstance(other, BoundingBox):
            return (other.minpoint >= self.minpoint) and (
                other.maxpoint <= self.maxpoint
            )
        else:
            raise NotImplementedError(
                f"Cannot test containedness of {type(other)} in {self.__class__.__name__}"
            )

    def contained_in(self, other):
        return other.contains(self)

    def intersects(self, other):
        return self.intersection(other).volume > 0

    def intersection(self, other, dims=[0, 1, 2]):
        """Computes the intersection of this bounding box with another one.

        TODO process the sigma values o the points

        Args:
            other (BoundingBox): Another bounding box
            dims (list of int): Dimensions where the intersection should be computed (applies only to bounding boxes)
            Default: all three. Along dimensions not listed, the union is applied instead.
        """
        if isinstance(other, Nifti1Image):
            return self._intersect_mask(other)
        elif isinstance(other, BoundingBox):
            return self._intersect_bbox(other, dims)
        else:
            raise NotImplementedError(
                f"Intersection of bounding box with {type(other)} not implemented."
            )

    def _intersect_bbox(self, other, dims):
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
            point1=Point(result_minpt, self.space),
            point2=Point(result_maxpt, self.space),
            space=self.space,
        )
        return bbox if bbox.volume > 0 else None

    def _intersect_mask(self, mask):
        """Intersect this bounding box with an image mask.

        TODO process the sigma values o the points

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # nonzero voxel coordinates
        X, Y, Z = np.where(mask.get_fdata() > 0)
        h = np.ones(len(X))

        # array of homogenous physical nonzero voxel coordinates
        coords = np.dot(mask.affine, np.vstack((X, Y, Z, h)))[:3, :].T
        minpoint = [min(self.minpoint[i], self.maxpoint[i]) for i in range(3)]
        maxpoint = [max(self.minpoint[i], self.maxpoint[i]) for i in range(3)]
        inside = np.logical_and.reduce([coords > minpoint, coords <= maxpoint]).min(1)
        XYZ = coords[inside, :3]
        if XYZ.shape[0] == 0:
            return None
        elif XYZ.shape[0] == 1:
            return Point(XYZ.flatten(), space=self.space)
        else:
            return PointSet(XYZ, space=self.space)

    def union(self, other):
        """Computes the union of this boudning box with another one.

        TODO process the sigma values o the points

        Args:
            other (BoundingBox): Another bounding box
        """
        warped = other.warp(self.space)
        points = [self.minpoint, self.maxpoint, warped.minpoint, warped.maxpoint]
        return BoundingBox(
            point1=[min(p[i] for p in points) for i in range(3)],
            point2=[max(p[i] for p in points) for i in range(3)],
            space=self.space,
        )

    def clip(self, xyzmax, xyzmin=(0, 0, 0)):
        """Returns a new bounding box obtained by clippin at the given maximum coordinate.

        TODO process the sigma values o the points
        """
        return self.intersection(
            BoundingBox(
                Point(xyzmin, self.space), Point(xyzmax, self.space), self.space
            )
        )

    def warp(self, space):
        """Returns a new bounding box obtained by warping the
        min- and maxpoint of this one into the new target space.

        TODO process the sigma values o the points
        """
        from .space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        else:
            return self.__class__(
                point1=self.minpoint.warp(spaceobj),
                point2=self.maxpoint.warp(spaceobj),
                space=spaceobj,
            )

    def build_mask(self):
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
        from .space import Space
        spaceobj = Space.get_instance(space)
        return self.__class__(
            point1=self.minpoint.transform(affine, spaceobj),
            point2=self.maxpoint.transform(affine, spaceobj),
            space=space,
        )

    def __iter__(self):
        """Iterate the min- and maxpoint of this bounding box."""
        return iter((self.minpoint, self.maxpoint))
