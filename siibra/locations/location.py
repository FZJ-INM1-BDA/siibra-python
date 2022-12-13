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


import numpy as np
from abc import ABC, abstractmethod
from nibabel import Nifti1Image


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

    def __init__(self, space):
        from ..core.space import Space
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
