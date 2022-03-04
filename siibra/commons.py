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

import os
from enum import Enum
from typing import Any, Generic, Iterable, Iterator, List, TypeVar
from nibabel import Nifti1Image
import logging
import numpy as np

logger = logging.getLogger(__name__.split(os.path.extsep)[0])
ch = logging.StreamHandler()
formatter = logging.Formatter("[{name}:{levelname}] {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


class LoggingContext:
    def __init__(self, level):
        self.level = level

    def __enter__(self):
        self.old_level = logger.level
        logger.setLevel(self.level)

    def __exit__(self, et, ev, tb):
        logger.setLevel(self.old_level)


def set_log_level(level):
    logger.setLevel(level)


set_log_level(os.getenv("SIIBRA_LOG_LEVEL", "INFO"))
QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")

T = TypeVar('T')

class TypedRegistry(Generic[T], Iterable):
    """
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """

    def __init__(self, matchfunc=lambda a, b: a == b, elements=None):
        """
        Build a glossary from a dictionary with string keys, for easy
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary
        (ie. an object that you put into this glossary), and as second argument
        the index/specification that should match one of the objects, and returning a boolean.
        """

        assert hasattr(matchfunc, "__call__")
        if elements is None:
            self._elements = {}
        else:
            assert isinstance(elements, dict)
            assert all(isinstance(k, str) for k in elements.keys())
            self._elements = elements
        self._matchfunc = matchfunc

    def add(self, key: str, value: T) -> None:
        """Add a key/value pair to the registry.

        Args:
            key (string): Unique name or key of the object
            value (object): The registered object
        """
        assert isinstance(key, str)
        if key in self._elements:
            logger.warning(
                f"Key {key} already in {__class__.__name__}, existing value will be replaced."
            )
        self._elements[key] = value

    def __dir__(self) -> Iterable[str]:
        """List of all object keys in the registry"""
        return self._elements.keys()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: " + ",".join(self._elements.keys())

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __contains__(self, key) -> bool:
        """Test wether the given key is defined by the registry."""
        return (
            key in self._elements
        )  # or any([self._matchfunc(v,spec) for v in self._elements.values()])

    def __len__(self) -> int:
        """Return the number of elements in the registry"""
        return len(self._elements)

    def __getitem__(self, spec) -> T:
        """Give access to objects in the registry by sequential index,
        exact key, or keyword matching. If the keywords match multiple objects,
        the first in sorted order is returned. If the specification does not match,
        a RuntimeError is raised.

        Args:
            spec [int or str]: Index or string specification of an object

        Returns:
            Matched object
        """
        if spec is None:
            raise RuntimeError(f"{__class__.__name__} indexed with None")
        matches = self.find(spec)
        if len(matches) == 0:
            print(str(self))
            raise IndexError(
                f"{__class__.__name__} has no entry matching the specification '{spec}'.\n"
                f"Possible values are:\n - " + "\n - ".join(self._elements.keys())
            )
        elif len(matches) == 1:
            return matches[0]
        else:
            S = sorted(matches, reverse=True)
            largest = S[0]
            logger.info(
                f"Multiple elements matched the specification '{spec}' - the first in order was chosen: {largest}"
            )
            logger.debug(f"(Other candidates were: {', '.join(m.name for m in S[1:])})")
            return largest

    def __sub__(self, obj) -> 'TypedRegistry[T]':
        """
        remove an object from the registry
        """
        if obj in self._elements.values():
            return TypedRegistry[T](
                self._matchfunc, {k: v for k, v in self._elements.items() if v != obj}
            )
        else:
            return self

    def provides(self, spec) -> bool:
        """
        Returns True if an element that matches the given specification can be found
        (using find(), thus going beyond the matching of names only as __contains__ does)
        """
        matches = self.find(spec)
        return len(matches) > 0

    def find(self, spec) -> List[T]:
        """
        Return a list of items matching the given specification,
        which could be either the name or a specification that
        works with the matchfunc of the Glossary.
        """
        if isinstance(spec, str) and (spec in self._elements):
            return [self._elements[spec]]
        elif isinstance(spec, int) and (spec < len(self._elements)):
            return [list(self._elements.values())[spec]]
        else:
            # string matching on values
            matches = [v for v in self._elements.values() if self._matchfunc(v, spec)]
            if len(matches) == 0:
                # string matching on keys
                matches = [
                    self._elements[k]
                    for k in self._elements.keys()
                    if all(w.lower() in k.lower() for w in spec.split())
                ]
            return matches

    def __getattr__(self, index) -> T:
        """Access elements by using their keys as attributes.
        Keys are auto-generated from the provided names to be uppercase,
        with words delimited using underscores.
        """
        if index in self._elements:
            return self._elements[index]
        else:
            hint = ""
            if isinstance(index, str):
                import difflib

                closest = difflib.get_close_matches(
                    index, list(self._elements.keys()), n=3
                )
                if len(closest) > 0:
                    hint = f"Did you mean {' or '.join(closest)}?"
            raise AttributeError(f"Term '{index}' not in {__class__.__name__}. " + hint)


class Registry(TypedRegistry[Any]): pass

class ParcellationIndex:
    """
    Identifies a unique region in a ParcellationMap, combining its labelindex (the "color") and mapindex (the number of the 3Dd map, in case multiple are provided).
    """

    def __init__(self, map, label):
        self.map = map
        self.label = label

    def __str__(self):
        return f"({self.map}/{self.label})"

    def __repr__(self):
        return f"{self.__class__.__name__} " + str(self)

    def __eq__(self, other):
        return all([self.map == other.map, self.label == other.label])

    def __hash__(self):
        return hash((self.map, self.label))


class MapType(Enum):
    LABELLED = 1
    CONTINUOUS = 2


def snake2camel(s: str):
    """Converts a string in snake_case into CamelCase.
    For example: JULICH_BRAIN -> JulichBrain"""
    return "".join([w[0].upper() + w[1:].lower() for w in s.split("_")])


# getting nonzero pixels of pmaps is one of the most time consuming tasks when computing metrics,
# so we cache the nonzero coordinates of array objects at runtime.
NZCACHE = {}


def nonzero_coordinates(arr):
    obj_id = id(arr)
    if obj_id not in NZCACHE:
        NZCACHE[obj_id] = np.c_[np.nonzero(arr > 0)]
    return NZCACHE[obj_id]


def affine_scaling(affine):
    """Estimate approximate isotropic scaling factor of an affine matrix. """
    orig = np.dot(affine, [0, 0, 0, 1])
    unit_lengths = []
    for vec in np.identity(3):
        vec_phys = np.dot(affine, np.r_[vec, 1])
        unit_lengths.append(np.linalg.norm(orig - vec_phys))
    return np.prod(unit_lengths)


def compare_maps(map1: Nifti1Image, map2: Nifti1Image):
    """
    Compare two maps, given as Nifti1Image objects.
    This function exploits that nibabel's get_fdata() caches the numerical arrays,
    so we can use the object id to cache extraction of the nonzero coordinates.
    Repeated calls involving the same map will therefore be much faster as they
    will only access the image array if overlapping pixels are detected.

    It is recommended to install the indexed-gzip package,
    which will further speed this up.
    """

    a1, a2 = [m.get_fdata().squeeze() for m in (map1, map2)]

    def homog(XYZ):
        return np.c_[XYZ, np.ones(XYZ.shape[0])]

    def colsplit(XYZ):
        return np.split(XYZ, 3, axis=1)

    # Compute the nonzero voxels in map2 and their correspondences in map1
    XYZnz2 = nonzero_coordinates(a2)
    N2 = XYZnz2.shape[0]
    warp2on1 = np.dot(np.linalg.inv(map1.affine), map2.affine)
    XYZnz2on1 = (np.dot(warp2on1, homog(XYZnz2).T).T[:, :3] + 0.5).astype("int")

    # valid voxel pairs
    valid = np.all(
        np.logical_and.reduce(
            [
                XYZnz2on1 >= 0,
                XYZnz2on1 < map1.shape[:3],
                XYZnz2 >= 0,
                XYZnz2 < map2.shape[:3],
            ]
        ),
        1,
    )
    X1, Y1, Z1 = colsplit(XYZnz2on1[valid, :])
    X2, Y2, Z2 = colsplit(XYZnz2[valid, :])

    # intersection
    v1, v2 = a1[X1, Y1, Z1].squeeze(), a2[X2, Y2, Z2].squeeze()
    m1, m2 = ((_ > 0).astype("uint8") for _ in [v1, v2])
    intersection = np.minimum(m1, m2).sum()
    if intersection == 0:
        return {"overlap": 0, "contained": 0, "contains": 0, "correlation": 0}

    # Compute the nonzero voxels in map1 with their correspondences in map2
    XYZnz1 = nonzero_coordinates(a1)
    N1 = XYZnz1.shape[0]
    warp1on2 = np.dot(np.linalg.inv(map2.affine), map1.affine)

    # Voxels referring to the union of the nonzero pixels in both maps
    XYZa1 = np.unique(np.concatenate((XYZnz1, XYZnz2on1)), axis=0)
    XYZa2 = (np.dot(warp1on2, homog(XYZa1).T).T[:, :3] + 0.5).astype("int")
    valid = np.all(
        np.logical_and.reduce(
            [XYZa1 >= 0, XYZa1 < map1.shape[:3], XYZa2 >= 0, XYZa2 < map2.shape[:3]]
        ),
        1,
    )
    Xa1, Ya1, Za1 = colsplit(XYZa1[valid, :])
    Xa2, Ya2, Za2 = colsplit(XYZa2[valid, :])

    # pearson's r wrt to full size image
    x = a1[Xa1, Ya1, Za1].squeeze()
    y = a2[Xa2, Ya2, Za2].squeeze()
    mu_x = x.sum() / a1.size
    mu_y = y.sum() / a2.size
    x0 = x - mu_x
    y0 = y - mu_y
    dem = np.sqrt(np.sum(x0 ** 2) * np.sum(y0 ** 2))
    if dem==0:
        r = 0
    else:
        r = np.sum(np.multiply(x0, y0)) / dem

    bx = (x > 0).astype("uint8")
    by = (y > 0).astype("uint8")

    return {
        "overlap": intersection / np.maximum(bx, by).sum(),
        "contained": intersection / N1,
        "contains": intersection / N2,
        "correlation": r,
    }
