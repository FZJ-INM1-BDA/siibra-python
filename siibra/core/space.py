# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .core import SemanticConcept

from ..retrieval import HttpRequest

from cloudvolume import Bbox
import re
import numpy as np
from abc import ABC, abstractmethod
from nibabel import Nifti1Image
from nibabel.affines import apply_affine
import json
from urllib.parse import quote


@SemanticConcept.provide_registry
class Space(SemanticConcept, bootstrap_folder="spaces", type_id="minds/core/referencespace/v1.0.0"):
    """
    A particular brain reference space.
    """

    def __init__(self, identifier, name, template_type=None, src_volume_type=None, dataset_specs=[]):
        SemanticConcept.__init__(self, identifier, name, dataset_specs)
        self.src_volume_type = src_volume_type
        self.type = template_type

    def get_template(self, resolution_mm=None):
        """
        Get the volumetric reference template image for this space.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side.
        """
        candidates = [vsrc for vsrc in self.volume_src if vsrc.volume_type == self.type]
        if not len(candidates) == 1:
            raise RuntimeError(f"Could not resolve template image for {self.name}. This is most probably due to a misconfiguration of the volume src.")
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
            raise TypeError("Slice access to spaces needs to define x,y and z ranges (e.g. Space[10:30,0:10,200:300])")
        startpoint = [s.start for s in slices]
        endpoint = [s.stop for s in slices]
        return self.get_bounding_box(startpoint, endpoint)

    def get_bounding_box(self, startpoint, endpoint):
        """
        Get a volume of interest specification from this space.

        Arguments
        ---------
        startpoint: 3D tuple defined in physical coordinates of this reference space
        endpoint: 3D tuple defined in physical coordinates of this reference space
        """
        return BoundingBox(startpoint, endpoint, self)

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a Space
        object from a json stream.
        """
        required_keys = ['@id', 'name', 'shortName', 'templateType']
        if any([k not in obj for k in required_keys]):
            return obj
        if "minds/core/referencespace/v1.0.0" not in obj['@id']:
            return obj

        result = cls(
            identifier=obj['@id'],
            name=obj['shortName'],
            template_type=obj['templateType'],
            src_volume_type=obj.get('srcVolumeType'),
            dataset_specs=obj.get('datasets', []))

        return result


# backend for transforming coordinates between spaces
SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"


# lookup of space identifiers to be used by SPACEWARP_SERVER
SPACEWARP_IDS = {
    Space.REGISTRY.MNI152_2009C_NONL_ASYM: "MNI 152 ICBM 2009c Nonlinear Asymmetric",
    Space.REGISTRY.MNI_COLIN_27: "MNI Colin 27",
    Space.REGISTRY.BIG_BRAIN: "Big Brain (Histology)"
}


class Location(ABC):
    """
    Defines a location in the given reference space.
    """

    def __init__(self, space: Space):
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
        """ Generates a new location by warping the
        current one into another reference space. """
        pass


class Point(Location):
    """ A single 3D point in reference space. """

    @staticmethod
    def parse(spec, unit='mm'):
        """ Converts a 3D coordinate specification into a 3D tuple of floats.

        Parameters
        ----------
        spec : Any of str, tuple(float,float,float)
            For string specifications, comma separation with decimal points are expected.

        Returns
        -------
        tuple(float,float,float)
        """
        if unit != 'mm':
            raise NotImplementedError("Coordinate parsing from strings is only supported for mm specifications so far.")
        if isinstance(spec, str):
            pat = r'([-\d\.]*)' + unit
            digits = re.findall(pat, spec)
            if len(digits) == 3:
                return (float(d) for d in digits)
        elif isinstance(spec, tuple) and len(spec) == 3 and all(v.isnumeric() for v in spec):
            return tuple(float(v) for v in spec)
        elif isinstance(spec, np.ndarray) and spec.size() == 3:
            return tuple(spec)

        raise ValueError("Cannot decode the specification 'spec' into a 3D coordinate tuple")

    def __init__(self, coordinatespec, space: Space):
        Location.__init__(self, space)
        self.coordinate = Point.parse(coordinatespec)

    def intersects_mask(self, mask: Nifti1Image):
        """ Returns true if this point lies in the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # transform physical coordinates to voxel coordinates for the query
        def check(mask, c):
            voxel = (apply_affine(np.linalg.inv(mask.affine), c) + .5).astype(int)
            if np.any(voxel >= mask.dataobj.shape):
                return False
            if mask.dataobj[voxel[0], voxel[1], voxel[2]] == 0:
                return False
            return True

        if mask.ndim == 4:
            return any(check(mask.slicer[:, :, :, i], self.coordinate) for i in range(mask.shape[3]))
        else:
            return check(mask, self.coordinate)

    def warp(self, targetspace: Space):
        """ Creates a new location by warping this location to another space """
        if any(s not in SPACEWARP_IDS for s in [self.space, targetspace]):
            raise ValueError(f"Cannot convert coordinates between {self.space} and {targetspace}")
        url = '{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}'.format(
            server=SPACEWARP_SERVER,
            src=quote(SPACEWARP_IDS[Space.REGISTRY[self.space]]),
            tgt=quote(SPACEWARP_IDS[Space.REGISTRY[targetspace]]),
            x=self.coordinate[0], y=self.coordinate[1], z=self.coordinate[2])
        response = HttpRequest(url, lambda b: json.loads(b.decode())).get()
        return self.__class__(
            coordinate=tuple(response["target_point"]),
            space=targetspace)


class PointSet(Location):
    """ A 3D polyline in reference space, defined by a list of coordinates. """

    def __init__(self, coordinates, space: Space):
        Location.__init__(self, space)
        self.coordinates = [Point(c, space) for c in coordinates]

    def intersects_mask(self, mask):
        """ Returns true if any of the polyline points lies in the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        return any(c.intersects_mask(mask) for c in self.coordinates)

    def warp(self, targetspace):
        return self.__class__(
            [c.warp(targetspace) for c in self.coordinates],
            targetspace
        )


class BoundingBox(Location, Bbox):
    """ A 3D axis-aligned bounding box, spanned by a 3D start- and endpoint """

    def __init__(self, startpoint, endpoint, space: Space):
        Location.__init__(self, space)
        Bbox.__init__(self, startpoint, endpoint)
        self.startpoint = Point(startpoint)
        self.endpoint = Point(endpoint)

    @classmethod
    def _bounding_box(A):
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
        return np.array([
            [xmin, xmax + 1],
            [ymin, ymax + 1],
            [zmin, zmax + 1],
            [1, 1]])

    @classmethod
    def from_image(cls, image: Nifti1Image, space: Space):
        """ Construct a bounding box from a nifti image """
        coords = np.dot(image.affine, cls._bounding_box(image.get_fdata()))
        return cls(
            startpoint=coords[:3, 0],
            endpoint=coords[:3, 1],
            space=space)

    def __str__(self):
        return f"Bounding box {self.minpt}mm -> {self.maxpt}mm defined in {self.space.name}"

    def intersects_mask(self, mask):
        """ Returns true if at least one nonzero voxel
        of the given mask is inside the boundding box.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # nonzero voxel coordinates
        X, Y, Z = np.where(mask.get_fdata() > 0)
        h = np.ones(len(X))

        # array of homogenous physical nonzero voxel coordinates
        coords = np.dot(mask.affine, np.vstack((X, Y, Z, h)))[:3, :].T
        minpt = [min(self.minpt[i], self.maxpt[i]) for i in range(3)]
        maxpt = [max(self.minpt[i], self.maxpt[i]) for i in range(3)]
        inside = np.logical_and.reduce([coords > minpt, coords <= maxpt]).min(1)
        return any(inside)

    def warp(self, targetspace):
        """ Returns a new bounding box obtanied by warping the
        start- and endpoint of this one into the new targetspace. """
        return self.__class__(
            startpoint=self.startpoint.warp(targetspace),
            endpoint=self.endpoint.warp(targetspace),
            space=targetspace
        )
