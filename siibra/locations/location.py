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
from __future__ import annotations

from . import assignment

from ..commons import logger
from ..core import region as _region

import numpy as np
from abc import ABC, abstractmethod
from nibabel import Nifti1Image
from typing import Union


class Location(ABC):
    """
    Abstract base class for locations in a given reference space.
    """

    # backend for transforming coordinates between spaces
    SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"

    # lookup of space identifiers to be used by SPACEWARP_SERVER
    SPACEWARP_IDS = {
        "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2": "MNI 152 ICBM 2009c Nonlinear Asymmetric",
        "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992": "MNI Colin 27",
        "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588": "Big Brain (Histology)",
    }

    # The id of BigBrain reference space
    BIGBRAIN_ID = "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588"
    _ASSIGNMENT_CACHE = {}  # cache assignment results, see Location.assign()
    _MASK_MEMO = {}  # cache region masks for Location._assign_region()

    def __init__(self, space):
        from ..core.space import Space
        self.space = Space.get_instance(space)

    @abstractmethod
    def intersection(self, other: Union[Location, Nifti1Image]) -> Location:
        """
        Subclasses of Location must implement intersection with other locations.
        """
        pass

    def intersects(self, other: Union[Nifti1Image, Location]) -> bool:
        """
        Verifies wether this 3D location intersects the given mask.
        """
        return self.intersection(other) is not None

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

    def __len__(self):
        """The number of coordinates or sublocations in a location."""
        return 0

    @abstractmethod
    def __iter__(self):
        """To be implemented in derived classes to return an iterator
        over the coordinates associated with the location."""
        pass

    @property
    def species(self):
        return None if self.space is None else self.space.species

    def __str__(self):
        space_str = "" if self.space is None else f" in {self.space.name}"
        coord_str = "" if len(self) == 0 else f" [{','.join(str(l) for l in iter(self))}]"
        return f"{self.__class__.__name__}{space_str}{coord_str}"

    def assign(self, other: Union[_region.Region, Location]):
        """ Assign this location to another location. """
        if (self, other) not in self._ASSIGNMENT_CACHE:
            if isinstance(other, Location):
                self._ASSIGNMENT_CACHE[self, other] = self._assign_location(other)

            elif isinstance(other, _region.Region):
                self._ASSIGNMENT_CACHE[self, other] = self._assign_region(other)
            else:
                raise ValueError(f"Cannot assign {self.__class__.__name__} to {other.__class__.__name__} objects.")
        return self._ASSIGNMENT_CACHE[self, other]

    def _assign_location(self, other: Location):
        if self == other:
            qualification = assignment.AssignmentQualification.EXACT
        elif self.contained_in(other):
            qualification = assignment.AssignmentQualification.CONTAINED
        elif self.contains(other):
            qualification = assignment.AssignmentQualification.CONTAINS
        elif self.intersects(other):
            qualification = assignment.AssignmentQualification.OVERLAPS
        else:
            qualification = None
        return None if qualification is None \
            else assignment.AnatomicalAssignment(self, other, qualification)

    def _assign_region(self, region: _region.Region):
        assert isinstance(region, _region.Region)

        if self.species != region.species:
            return None

        for subregion in region.children:
            res = self._assign_region(subregion)
            if res is not None:
                return res

        # Retrieve a mask of the region, if possible in the same space
        mask = None
        if region.mapped_in_space(self.space, recurse=True):
            # The region is mapped in the space of this location,
            # so we fetch and memoize the voxel mask.
            if (region, self.space) not in self._MASK_MEMO:
                self._MASK_MEMO[region, self.space] = \
                    region.fetch_regional_map(space=self.space, maptype='labelled')
            mask_space = self.space
        else:
            # the region not mapped in the space of this location.
            for space in region.supported_spaces:
                if not space.provides_image:
                    # siibra does not yet match locations to surface spaces
                    continue
                if (region, space) not in self._MASK_MEMO:
                    self._MASK_MEMO[region, space] = region.fetch_regional_map(space=space, maptype='labelled')
                mask_space = space
                if self._MASK_MEMO[region, space] is not None:
                    break

        if len(self._MASK_MEMO) > 3:
            # Limit the mask cache to the three masks that were fetched most recently.
            self._MASK_MEMO.pop(next(iter(self._MASK_MEMO)))

        mask = self._MASK_MEMO[region, mask_space]
        if mask is None:
            logger.debug(
                f"'{region.name}' provides no mask in a space to which {self} can be warped."
            )
            return None

        # compare mask to location
        loc_warped = self.warp(mask_space)
        if loc_warped is None:
            logger.warn(
                f"Cannot warp {self} to {mask_space} for testing against region mask. "
                "Will try to test against warped bounding box of the region."
            )
            region_bbox = region.get_bounding_box(self.space)
            if region_bbox is None:
                raise RuntimeError(f"No bounding box obtained from region mask of {region} in {mask_space}")
            else:
                # got a bounding box, so assign this location to the bounding box location.
                return self.assign(region_bbox)
        elif loc_warped.contained_in(mask):
            qualification = assignment.AssignmentQualification.CONTAINED
        elif loc_warped.contains(mask):
            qualification = assignment.AssignmentQualification.CONTAINS
        elif loc_warped.intersects(mask):
            qualification = assignment.AssignmentQualification.OVERLAPS
        else:
            qualification = None

        # put together an explanation for the assignment
        if self.space == mask_space:
            expl = (
                f"{self} was compared with the mask of query region '{region.name}' "
                f"in {mask_space}."
            )
        else:
            expl = (
                f"{self} was warped from {self.space.name} and then compared "
                f"in this space with the mask of query region '{region.name}'."
            )
        return None if qualification is None \
            else assignment.AnatomicalAssignment(self, region, qualification, expl)


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

    def intersects(self, *_args, **_kwargs):
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
