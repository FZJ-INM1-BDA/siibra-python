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

from .concept import AtlasConcept, provide_registry

from ..commons import logger
from ..retrieval import HttpRequest

from cloudvolume import Bbox
import re
import numpy as np
from abc import ABC, abstractmethod
from nibabel import Nifti1Image
from nibabel.affines import apply_affine
import json
from urllib.parse import quote
from os import path
import numbers


@provide_registry
class Space(
    AtlasConcept,
    bootstrap_folder="spaces",
    type_id="minds/core/referencespace/v1.0.0",
):
    """
    A particular brain reference space.
    """

    def __init__(
        self,
        identifier,
        name,
        template_type=None,
        src_volume_type=None,
        dataset_specs=[],
    ):
        AtlasConcept.__init__(self, identifier, name, dataset_specs)
        self.src_volume_type = src_volume_type
        self.type = template_type
        self.atlases = set()

    def get_template(self):
        """
        Get the volumetric reference template image for this space.

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side.
        """
        candidates = [vsrc for vsrc in self.volumes if vsrc.volume_type == self.type]
        if not len(candidates) == 1:
            raise RuntimeError(
                f"Could not resolve template image for {self.name}. This is most probably due to a misconfiguration of the volume src."
            )
        return candidates[0]

    def __getitem__(self, slices):
        """
        Get a volume of interest specification from this space.

        Arguments
        ---------
        slices: triple of slice
            defines the x, y and z range
        """
        if len(slices) != 3:
            raise TypeError(
                "Slice access to spaces needs to define x,y and z ranges (e.g. Space[10:30,0:10,200:300])"
            )
        point1 = [0 if s.start is None else s.start for s in slices]
        point2 = [s.stop for s in slices]
        if None in point2:
            # fill upper bounds with maximum physical coordinates
            T = self.get_template()
            shape = Point(T.get_shape(-1), None).transform(T.build_affine(-1))
            point2 = [
                shape[i] if v is None else v
                for i, v in enumerate(point2)]
        return self.get_bounding_box(point1, point2)

    def __lt__(self, other):
        return self.type < other.type

    def get_bounding_box(self, point1, point2):
        """
        Get a volume of interest specification from this space.

        Arguments
        ---------
        point1: 3D tuple defined in physical coordinates of this reference space
        point2: 3D tuple defined in physical coordinates of this reference space
        """
        return BoundingBox(point1, point2, self)

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a Space
        object from a json stream.
        """
        required_keys = ["@id", "name", "shortName", "templateType"]
        if any([k not in obj for k in required_keys]):
            return obj
        if "minds/core/referencespace/v1.0.0" not in obj["@id"]:
            return obj

        result = cls(
            identifier=obj["@id"],
            name=obj["shortName"],
            template_type=obj["templateType"],
            src_volume_type=obj.get("srcVolumeType"),
            dataset_specs=obj.get("datasets", []),
        )

        return result


# backend for transforming coordinates between spaces
SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"


# lookup of space identifiers to be used by SPACEWARP_SERVER
SPACEWARP_IDS = {
    Space.REGISTRY.MNI152_2009C_NONL_ASYM: "MNI 152 ICBM 2009c Nonlinear Asymmetric",
    Space.REGISTRY.MNI_COLIN_27: "MNI Colin 27",
    Space.REGISTRY.BIG_BRAIN: "Big Brain (Histology)",
}


class Location(ABC):
    """
    Abstract base class for locations in a given reference space.
    """

    def __init__(self, space: Space):
        if space is None:
            # typically only used for temporary entities, e.g. in individual voxel spaces.
            self.space = None
        else:
            self.space = Space.REGISTRY[space]

    @abstractmethod
    def intersects_mask(self, mask: Nifti1Image):
        """
        Verifies wether this 3D location intersects the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        pass

    @abstractmethod
    def warp(self, targetspace: Space):
        """Generates a new location by warping the
        current one into another reference space."""
        pass

    @abstractmethod
    def transform(self, affine: np.ndarray, space: Space = None):
        """Returns a new location obtained by transforming the
        reference coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : Space, or None (optional)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """ To be implemented in derived classes to return an iterator
        over the coordinates associated with the location. """
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

    @classmethod
    def from_sands(cls, spec):
        """ Try to build a location object from an openMINDS/SANDS specification. """
        if isinstance(spec, str):
            if path.isfile(spec):
                with open(spec, 'r') as f:
                    obj = json.load(f)
            else:
                obj = json.loads(spec)
        elif isinstance(spec, dict):
            obj = spec
        else:
            raise NotImplementedError(f'Cannot read openMINDS/SANDS info from {type(spec)} types.')

        if obj['@type'] == "https://openminds.ebrains.eu/sands/CoordinatePoint":
            return Point.from_sands(obj)
        elif obj['@type'] == "tmp/poly":
            return PointSet.from_sands(obj)
        else:
            raise NotImplementedError(
                "Building location objects from openMINDS/Sands "
                f"type {obj['@type']} is not yet supported."
            )


class WholeBrain(Location):
    """
    Trivial location class for formally representing
    location in a particular reference space. Which
    is not further specified.
    """

    def __init__(self, space: Space):
        if space is None:
            # typically only used for temporary entities, e.g. in individual voxel spaces.
            self.space = None
        else:
            self.space = Space.REGISTRY[space]

    def intersects_mask(self, mask: Nifti1Image):
        """ Always true for whole brain features """
        return True

    def warp(self, targetspace: Space):
        """ Generates a new whole brain location
        in another reference space. """
        return self.__class__(targetspace)

    def transform(self, affine: np.ndarray, space: Space = None):
        """ Does nothing. """
        pass

    def __iter__(self):
        """ To be implemented in derived classes to return an iterator
        over the coordinates associated with the location. """
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
        elif (
            isinstance(spec, (tuple, list))
            and len(spec) in [3, 4]
        ):
            if len(spec) == 4:
                assert(spec[3] == 1)
            return tuple(float(v) for v in spec[:3])
        elif isinstance(spec, np.ndarray) and spec.size == 3:
            return tuple(spec)
        elif isinstance(spec, Point):
            return spec.coordinate
        else:
            raise ValueError(
                f"Cannot decode the specification {spec} (type {type(spec)}) to create a point."
            )

    def __init__(self, coordinatespec, space: Space, sigma_mm:float=0.):
        """
        Construct a new 3D point set in the given reference space.

        Parameters
        ----------
        coordinate : 3-tuple of int/float, or string specification
            Coordinate in mm of the given space
        space : Space
            The reference space
        sigma_mm : float
            Optional location uncertainy of the point 
            (will be intrepreded as the isotropic standard deviation of the location)
        """
        Location.__init__(self, space)
        self.coordinate = Point.parse(coordinatespec)
        self.sigma = sigma_mm
        if isinstance(coordinatespec, Point):
            assert coordinatespec.sigma == sigma_mm
            assert coordinatespec.space == space

    @property
    def homogeneous(self):
        """ The homogenous coordinate of this point as a 4-tuple,
        obtained by appending '1' to the original 3-tuple. """
        return self.coordinate + (1,)

    def intersects_mask(self, mask: Nifti1Image):
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

    def warp(self, targetspace: Space):
        """ Creates a new point by warping this point to another space """
        if not isinstance(targetspace, Space):
            targetspace = Space.REGISTRY[targetspace]
        if targetspace == self.space:
            return self
        if any(s not in SPACEWARP_IDS for s in [self.space, targetspace]):
            raise ValueError(
                f"Cannot convert coordinates between {self.space} and {targetspace}"
            )
        url = "{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}".format(
            server=SPACEWARP_SERVER,
            src=quote(SPACEWARP_IDS[Space.REGISTRY[self.space]]),
            tgt=quote(SPACEWARP_IDS[Space.REGISTRY[targetspace]]),
            x=self.coordinate[0],
            y=self.coordinate[1],
            z=self.coordinate[2],
        )
        response = HttpRequest(url, lambda b: json.loads(b.decode())).get()
        return self.__class__(
            coordinatespec=tuple(response["target_point"]),
            space=targetspace
        )

    def __sub__(self, other):
        """ Substract the coordinates of two points to get
        a new point representing the offset vector. Alternatively,
        subtract an integer from the all coordinates of this point
        to create a new one."""
        if isinstance(other, numbers.Number):
            return Point(
                [c - other for c in self.coordinate],
                self.space
            )

        assert(self.space == other.space)
        return Point(
            [
                self.coordinate[i] - other.coordinate[i]
                for i in range(3)],
            self.space)

    def __add__(self, other):
        """ Add the coordinates of two points to get
        a new point representing. """
        if isinstance(other, numbers.Number):
            return Point(
                [c + other for c in self.coordinate],
                self.space
            )
        assert(self.space == other.space)
        return Point(
            [
                self.coordinate[i] + other.coordinate[i]
                for i in range(3)],
            self.space)

    def __truediv__(self, number):
        """ Return a new point with divided
        coordinates in the same space. """
        return Point(
            np.array(self.coordinate) / float(number),
            self.space
        )

    def __mult__(self, number):
        """ Return a new point with multiplied
        coordinates in the same space. """

    def transform(self, affine: np.ndarray, space: Space = None):
        """Returns a new Point obtained by transforming the
        coordinate of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : Space, or None (optional)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        x, y, z, h = np.dot(affine, self.homogeneous)
        if h != 1:
            logger.warning(f"Homogeneous coordinate is not one: {h}")
        return self.__class__((x / h, y / h, z / h), space)

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
        """ Return an iterator over the location,
        so the Point can be easily cast to list or tuple. """
        return iter(self.coordinate)

    def __getitem__(self, index):
        """ Index access to the coefficients of this point. """
        assert(0 <= index < 3)
        return self.coordinate[index]

    @classmethod
    def from_sands(cls, spec):
        """ Generate a point from an openMINDS/SANDS specification,
        given as a dictionary, json string, or json filename. """

        if isinstance(spec, str):
            if path.isfile(spec):
                with open(spec, 'r') as f:
                    obj = json.load(f)
            else:
                obj = json.loads(spec)
        elif isinstance(spec, dict):
            obj = spec
        else:
            raise NotImplementedError(f'Cannot read openMINDS/SANDS info from {type(spec)} types.')

        assert(obj['@type'] == "https://openminds.ebrains.eu/sands/CoordinatePoint")

        # require space spec
        space_id = obj['coordinateSpace']['@id']
        assert(Space.REGISTRY.provides(space_id))

        # require a 3D point spec for the coordinates
        assert(all(
            c["unit"]["@id"] == "id.link/mm"
            for c in obj['coordinates']))

        # build the Point
        return cls(
            list(np.float16(c["value"]) for c in obj['coordinates']),
            space=Space.REGISTRY[space_id]
        )

    def bigbrain_section(self):
        """
        Estimate the histological section number of BigBraing
        which corresponds to this point. If the point is given
        in another space, a warping to BigBrain space will be tried.
        """
        if self.space == Space.REGISTRY['bigbrain']:
            coronal_position = self[1]
        else:
            try:
                bigbrain_point = self.warp('bigbrain')
                coronal_position = bigbrain_point[1]
            except Exception:
                raise RuntimeError(
                    "BigBrain section numbers can only be determined "
                    "for points in BigBrain space, but the given point "
                    f"is given in '{self.space.name}' and could not "
                    "be converted.")
        return int((coronal_position + 70.) / 0.02 + 1.5)


class PointSet(Location):
    """A set of 3D points in the same reference space,
    defined by a list of coordinates. """

    def __init__(self, coordinates, space: Space, sigma_mm = 0):
        """
        Construct a 3D point set in the given reference space.

        Parameters
        ----------
        coordinates : list of Point, 3-tuples or string specs
            Coordinates in mm of the given space
        space : Space
            The reference space
        sigma_mm : float, or list of float
            Optional standard deviation of point locations.
        """
        Location.__init__(self, space)
        if isinstance(sigma_mm, numbers.Number):
            self.coordinates = [Point(c, space, sigma_mm) for c in coordinates]
        else:
            self.coordinates = [Point(c, space, s) for c, s in zip(coordinates, sigma_mm)]

    def intersects_mask(self, mask):
        """Returns true if any of the polyline points lies in the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        return any(c.intersects_mask(mask) for c in self.coordinates)

    def warp(self, targetspace):
        spaceobj = Space.REGISTRY[targetspace]
        if spaceobj == self.space:
            return self
        return self.__class__(
            [c.warp(spaceobj) for c in self.coordinates], spaceobj
        )

    def transform(self, affine: np.ndarray, space: Space = None):
        """Returns a new PointSet obtained by transforming the
        coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : Space, or None (optional)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        return self.__class__(
            [c.transform(affine, space) for c in self.coordinates], space
        )

    def __getitem__(self, index: int):
        if (index >= self.__len__()) or (index < 0):
            raise IndexError(
                f"Pointset has only {self.__len__()} points, "
                f"but index of {index} was requested.")
        else:
            return self.coordinates[index]

    def __iter__(self):
        """ Return an iterator over the coordinate locations. """
        return iter(self.coordinates)

    def __len__(self):
        """ The number of points in this PointSet. """
        return len(self.coordinates)

    @classmethod
    def from_sands(cls, spec):
        """ Generate a point set from an openMINDS/SANDS specification,
        given as a dictionary, json string, or json filename. """

        if isinstance(spec, str):
            if path.isfile(spec):
                with open(spec, 'r') as f:
                    obj = json.load(f)
            else:
                obj = json.loads(spec)
        elif isinstance(spec, dict):
            obj = spec
        else:
            raise NotImplementedError(f'Cannot read openMINDS/SANDS info from {type(spec)} types.')

        assert(obj['@type'] == "tmp/poly")

        # require space spec
        space_id = obj['coordinateSpace']['@id']
        assert(Space.REGISTRY.provides(space_id))

        # require mulitple 3D point specs
        coords = []
        for coord in obj['coordinates']:
            assert(all(c["unit"]["@id"] == "id.link/mm" for c in coord))
            coords.append(list(np.float16(c["value"]) for c in coord))

        # build the Point
        return cls(coords, space=Space.REGISTRY[space_id])

    @property
    def boundingbox(self):
        """ Return the bounding box of these points.

        TODO inherit sigma of the min and max points
        """
        XYZ = self.homogeneous[:, :3]
        return BoundingBox(
            point1 = XYZ.min(0),
            point2 = XYZ.max(0),
            space = self.space,
        )

    @property
    def centroid(self):
        return Point(
            self.homogeneous[:, :3].mean(0),
            space=self.space)

    def as_list(self):
        """ Return the point set as a list of 3D tuples. """
        return [tuple(p) for p in self]

    @property
    def homogeneous(self):
        """ Access the list of 3D point as an Nx4 array of homogeneous coorindates."""
        return np.array([c.homogeneous for c in self.coordinates])


class BoundingBox(Location):
    """
    A 3D axis-aligned bounding box spanned by two 3D corner points.
    The box does not necessarily store the given points,
    instead it computes the real minimum and maximum points
    from the two corner points.
    """

    def __init__(self, point1, point2, space: Space):
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
        space : Space
            The reference space
        """
        Location.__init__(self, space)
        xyz1 = Point.parse(point1)
        xyz2 = Point.parse(point2)
        self.minpoint = Point([min(xyz1[i], xyz2[i]) for i in range(3)], space)
        self.maxpoint = Point([max(xyz1[i], xyz2[i]) for i in range(3)], space)

    @property
    def _Bbox(self):
        """
        Return the bounding box as a cloudvolume Bbox object,
        with rounded integer coordinates. """
        return Bbox(
            [int(v) for v in self.minpoint.coordinate],
            [int(v+.5) for v in self.maxpoint.coordinate]
        )

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
    def from_image(cls, image: Nifti1Image, space: Space, ignore_affine=False):
        """ Construct a bounding box from a nifti image """
        bounds = cls._determine_bounds(image.get_fdata())
        if bounds is None:
            return None
        if ignore_affine:
            target_space = None
        else:
            bounds = np.dot(image.affine, bounds)
            targetspace = space
        return cls(point1=bounds[:3, 0], point2=bounds[:3, 1], space=space)

    def __str__(self):
        if self.space is None:
            return f"Bounding box {self.minpoint} -> {self.maxpoint} in unknown coordinate system"
        else:
            return f"Bounding box {self.minpoint}mm -> {self.maxpoint}mm defined in {self.space.name}"

    def intersection(self, other, dims=[0, 1, 2]):
        """Computes the intersection of this boudning box with another one.

        TODO process the sigma values o the points

        Args:
            other (BoundingBox): Another bounding box
            dims (list of int): Dimensions where the intersection should be computed.
            Default: all three. Along dimensions not listed, the union is applied instead.
        """
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

        return BoundingBox(
            point1=Point(result_minpt, self.space),
            point2=Point(result_maxpt, self.space),
            space=self.space,
        )

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
        """ Returns a new bounding box obtained by clippin at the given maximum coordinate. 

        TODO process the sigma values o the points
        """
        return self.intersection(
            BoundingBox(
                Point(xyzmin, self.space),
                Point(xyzmax, self.space),
                self.space
            )
        )

    def intersects_mask(self, mask):
        """Returns true if at least one nonzero voxel
        of the given mask is inside the boundding box.

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
        return any(inside)

    def warp(self, targetspace):
        """Returns a new bounding box obtained by warping the
        min- and maxpoint of this one into the new targetspace.

        TODO process the sigma values o the points
        """
        if targetspace == self.space:
            return self
        else:
            return self.__class__(
                point1=self.minpoint.warp(targetspace),
                point2=self.maxpoint.warp(targetspace),
                space=targetspace,
            )

    def build_mask(self):
        """Generate a volumetric binary mask of this
        bounding box in the reference template space."""
        tpl = self.space.get_template().fetch()
        arr = np.zeros(tpl.shape, dtype='uint8')
        bbvox = self.transform(np.linalg.inv(tpl.affine))
        arr[
            int(bbvox.minpoint[0]):int(bbvox.maxpoint[0]),
            int(bbvox.minpoint[1]):int(bbvox.maxpoint[2]),
            int(bbvox.minpoint[2]):int(bbvox.maxpoint[2]),
        ] = 1
        return Nifti1Image(arr, tpl.affine)

    def transform(self, affine: np.ndarray, space: Space = None):
        """Returns a new bounding box obtained by transforming the
        min- and maxpoint of this one with the given affine matrix.

        TODO process the sigma values o the points

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : Space, or None (optional)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        return self.__class__(
            point1=self.minpoint.transform(affine, space),
            point2=self.maxpoint.transform(affine, space),
            space=space,
        )

    def __iter__(self):
        """ Iterate the min- and maxpoint of this bounding box. """
        return iter((self.minpoint, self.maxpoint))
