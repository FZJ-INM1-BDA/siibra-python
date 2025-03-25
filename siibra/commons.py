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
"""Constants, functions, and classes used commonly across siibra."""

import os
import re
from enum import Enum
from nibabel import Nifti1Image
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Generic, Iterable, Iterator, List, TypeVar, Union, Dict
from skimage.filters import gaussian
from dataclasses import dataclass
try:
    from typing import TypedDict
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict

logger = logging.getLogger(__name__.split(os.path.extsep)[0])
ch = logging.StreamHandler()
formatter = logging.Formatter("[{name}:{levelname}] {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HBP_AUTH_TOKEN = os.getenv("HBP_AUTH_TOKEN")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")
SIIBRA_CACHEDIR = os.getenv("SIIBRA_CACHEDIR")
SIIBRA_LOG_LEVEL = os.getenv("SIIBRA_LOG_LEVEL", "INFO")
SIIBRA_USE_CONFIGURATION = os.getenv("SIIBRA_USE_CONFIGURATION")
SIIBRA_USE_LOCAL_SNAPSPOT = os.getenv("SIIBRA_USE_LOCAL_SNAPSPOT")
SKIP_CACHEINIT_MAINTENANCE = os.getenv("SKIP_CACHEINIT_MAINTENANCE")
NEUROGLANCER_MAX_GIB = os.getenv("NEUROGLANCER_MAX_GIB", 0.2)

with open(os.path.join(ROOT_DIR, "VERSION"), "r") as fp:
    __version__ = fp.read().strip()


@dataclass
class CompareMapsResult:
    intersection_over_union: float
    intersection_over_first: float
    intersection_over_second: float
    correlation: float
    weighted_mean_of_first: float
    weighted_mean_of_second: float


class TypePublication(TypedDict):
    citation: str
    url: str


T = TypeVar("T")


class InstanceTable(Generic[T], Iterable):
    """
    Lookup table for instances of a given class by name/id.
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """

    def __init__(self, matchfunc=lambda a, b: a == b, elements=None):
        """
        Build an object lookup table from a dictionary with string keys, for easy
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary
        (ie. an object that you put into this glossary), and as second argument
        the index/specification that should match one of the objects, and returning a boolean.
        """

        assert hasattr(matchfunc, "__call__")
        if elements is None:
            self._elements: Dict[str, T] = {}
        else:
            assert isinstance(elements, dict)
            assert all(isinstance(k, str) for k in elements.keys())
            self._elements: Dict[str, T] = elements
        self._matchfunc = matchfunc
        self._dataframe_cached = None

    def add(self, key: str, value: T) -> None:
        """Add a key/value pair to the registry.

        Args:
            key (string): Unique name or key of the object
            value (object): The registered object
        """
        if key in self._elements:
            logger.error(
                f"Key {key} already in {__class__.__name__}, existing value will be replaced."
            )
        self._elements[key] = value

    def __dir__(self) -> Iterable[str]:
        """List of all object keys in the registry"""
        if isinstance(self[0], type):
            return list(self._elements.keys())
        else:
            return ["dataframe"] + list(self._elements.keys())

    def __str__(self) -> str:
        if len(self) > 0:
            return f"{self.__class__.__name__}:\n - " + "\n - ".join(self._elements.keys())
        else:
            return f"Empty {self.__class__.__name__}"

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __contains__(self, key: Union[str, T]) -> bool:
        """Test wether the given key or element is defined by the registry."""
        if isinstance(key, str):
            return key in self._elements
        return key in [item for _, item in self._elements.values()]

    def __len__(self) -> int:
        """Return the number of elements in the registry"""
        return len(self._elements)

    def __getitem__(self, spec) -> T:
        return self.get(spec)

    def get(self, spec) -> T:
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
            return None
        elif spec == "":
            raise IndexError(f"{__class__.__name__} indexed with empty string")
        matches = self.find(spec)
        if len(matches) == 0:
            print(str(self))
            raise IndexError(
                f"{__class__.__name__} has no entry matching the specification '{spec}'.\n"
                f"Possible values are: " + ", ".join(self._elements.keys())
            )
        elif len(matches) == 1:
            return matches[0]
        else:
            try:
                S = sorted(matches, reverse=True)
            except TypeError:
                # not all object types support sorting, accept this
                S = matches
            largest = S[0]
            logger.info(
                f"Multiple elements matched the specification '{spec}' - the first in order was chosen: {largest}"
            )
            return largest

    def __sub__(self, obj) -> "InstanceTable[T]":
        """
        remove an object from the registry
        """
        if obj in self._elements.values():
            return InstanceTable[T](
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

    def values(self):
        return self._elements.values()

    def __getattr__(self, index) -> T:
        """Access elements by using their keys as attributes.
        Keys are auto-generated from the provided names to be uppercase,
        with words delimited using underscores.
        """
        if index in ["keys", "names"]:
            return list(self._elements.keys())
        elif index in self._elements:
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

    @property
    def dataframe(self):
        if self._dataframe_cached is None:
            values = self._elements.values()
            attrs = []
            for i, val in enumerate(values):
                attrs.append({'name': val.name, 'species': str(val.species)})
                if hasattr(val, 'maptype'):
                    attrs[i].update(
                        {
                            attribute: val.__getattribute__(attribute).name
                            for attribute in ['parcellation', 'space', 'maptype']
                        }
                    )
            self._dataframe_cached = pd.DataFrame(index=list(self._elements.keys()), data=attrs)
        return self._dataframe_cached


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


set_log_level(SIIBRA_LOG_LEVEL)
QUIET = LoggingContext("ERROR")
VERBOSE = LoggingContext("DEBUG")


def siibra_tqdm(iterable: Iterable[T] = None, *args, **kwargs):
    return tqdm(
        iterable,
        *args,
        disable=kwargs.pop("disable", False) or (logger.level > 20),
        **kwargs
    )


def create_key(name: str):
    """
    Creates an uppercase identifier string that includes only alphanumeric
    characters and underscore from a natural language name.
    """
    return re.sub(
        r" +",
        "_",
        "".join([e if e.isalnum() else " " for e in name]).upper().strip(),
    )


class MapIndex:
    """
    Identifies a unique region in a ParcellationMap, combining its labelindex (the "color") and mapindex (the number of the 3Dd map, in case multiple are provided).
    """

    def __init__(self, volume: int = None, label: int = None, fragment: str = None):
        if volume is None and label is None:
            raise ValueError(
                "At least volume or label need to be specified to build a valid map index."
            )
        if volume is not None:
            assert isinstance(volume, int)
        if label is not None:
            assert isinstance(label, int)
        self.volume = volume
        self.label = label
        self.fragment = fragment

    @classmethod
    def from_dict(cls, spec: dict):
        assert all(k in spec for k in ['volume', 'label'])
        return cls(
            volume=spec['volume'],
            label=spec['label'],
            fragment=spec.get('fragment')
        )

    def __str__(self):
        return f"(volume:{self.volume}, label:{self.label}, fragment:{self.fragment})"

    def __repr__(self):
        return f"{self.__class__.__name__}{str(self)}"

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        return all([
            self.volume == other.volume,
            self.label == other.label,
            self.fragment == other.fragment
        ])

    def __hash__(self):
        return hash((self.volume, self.label, self.fragment))


class MapType(Enum):
    LABELLED = 1
    STATISTICAL = 2


SIIBRA_DEFAULT_MAPTYPE = MapType.LABELLED
SIIBRA_DEFAULT_MAP_THRESHOLD = None

REMOVE_FROM_NAME = [
    "hemisphere",
    " -",
    "-brain",
    "both",
    "Both",
]

REPLACE_IN_NAME = {
    "ctx-lh-": "left ",
    "ctx-rh-": "right ",
}


def clear_name(name):
    """ clean up a region name to the for matching"""
    result = name
    for word in REMOVE_FROM_NAME:
        result = result.replace(word, "")
    for search, repl in REPLACE_IN_NAME.items():
        result = result.replace(search, repl)
    return " ".join(w for w in result.split(" ") if len(w))


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


def iterate_connected_components(img: Nifti1Image):
    """
    Provide an iterator over masks of connected components in the given image.
    """
    from skimage import measure
    imgdata = np.asanyarray(img.dataobj).squeeze()
    components = measure.label(imgdata > 0)
    component_labels = np.unique(components)
    assert component_labels[0] == 0
    return (
        (label, Nifti1Image((components == label).astype('uint8'), img.affine))
        for label in component_labels[1:]
    )


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
        return CompareMapsResult(
            intersection_over_union=0,
            intersection_over_first=0,
            intersection_over_second=0,
            correlation=0,
            weighted_mean_of_first=0,
            weighted_mean_of_second=0,
        )

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
    if dem == 0:
        r = 0
    else:
        r = np.sum(np.multiply(x0, y0)) / dem

    bx = (x > 0).astype("uint8")
    by = (y > 0).astype("uint8")
    return CompareMapsResult(
        intersection_over_union=intersection / np.maximum(bx, by).sum(),
        intersection_over_first=intersection / N1,
        intersection_over_second=intersection / N2,
        correlation=r,
        weighted_mean_of_first=np.sum(x * y) / np.sum(y),
        weighted_mean_of_second=np.sum(x * y) / np.sum(x),
    )


class PolyLine:
    """Simple polyline representation which allows equidistant sampling.."""

    def __init__(self, pts):
        self.pts = pts
        self.lengths = [
            np.sqrt(np.sum((pts[i, :] - pts[i - 1, :]) ** 2))
            for i in range(1, pts.shape[0])
        ]

    def length(self):
        return sum(self.lengths)

    def sample(self, d):

        # if d is interable, we assume a list of sample positions
        try:
            iter(d)
        except TypeError:
            positions = [d]
        else:
            positions = d

        samples = []
        for s_ in positions:
            s = min(max(s_, 0), 1)
            target_distance = s * self.length()
            current_distance = 0
            for i, length in enumerate(self.lengths):
                current_distance += length
                if current_distance >= target_distance:
                    p1 = self.pts[i, :]
                    p2 = self.pts[i + 1, :]
                    r = (target_distance - current_distance + length) / length
                    samples.append(p1 + (p2 - p1) * r)
                    break

        if len(samples) == 1:
            return samples[0]
        else:
            return np.array(samples)


def unify_stringlist(L: list):
    """Adds asterisks to strings that appear multiple times, so the resulting
    list has only unique strings but still the same length, order, and meaning.
    For example:
        unify_stringlist(['a','a','b','a','c']) -> ['a','a*','b','a**','c']
    """
    assert all([isinstance(_, str) for _ in L])
    return [L[i] + "*" * L[:i].count(L[i]) for i in range(len(L))]


def create_gaussian_kernel(sigma=1, sigma_point=3):
    """
    Compute a 3D Gaussian kernel of the given bandwidth.
    """
    r = int(sigma_point * sigma)
    k_size = 2 * r + 1
    impulse = np.zeros((k_size, k_size, k_size))
    impulse[r, r, r] = 1
    kernel = gaussian(impulse, sigma)
    kernel /= kernel.sum()
    return kernel


def argmax_dim4(img, dim=-1):
    """
    Given a nifti image object with four dimensions, returns a modified object
    with 3 dimensions that is obtained by taking the argmax along one of the
    four dimensions (default: the last one). To distinguish the pure background
    voxels from the foreground voxels of channel 0, the argmax indices are
    incremented by 1 and label index 0 is kept to represent the background.
    """
    assert len(img.shape) == 4
    assert dim >= -1 and dim < 4
    newarr = np.asarray(img.dataobj).argmax(dim) + 1
    # reset the true background voxels to zero
    newarr[np.asarray(img.dataobj).max(dim) == 0] = 0
    return Nifti1Image(dataobj=newarr, header=img.header, affine=img.affine)


def MI(arr1, arr2, nbins=100, normalized=True):
    """
    Compute the mutual information between two 3D arrays, which need to have the same shape.

    Parameters:
    arr1 : First 3D array
    arr2 : Second 3D array
    nbins : number of bins to use for computing the joint histogram (applies to intensity range)
    normalized : Boolean, default:True
        if True, the normalized MI of arrays X and Y will be returned,
        leading to a range of values between 0 and 1. Normalization is
        achieved by NMI = 2*MI(X,Y) / (H(X) + H(Y)), where  H(x) is the entropy of X
    """

    assert all(len(arr.shape) == 3 for arr in [arr1, arr2])
    assert (all(arr.size > 0) for arr in [arr1, arr2])

    # compute the normalized joint 2D histogram as an
    # empirical measure of the joint probabily of arr1 and arr2
    pxy, _, _ = np.histogram2d(arr1.ravel(), arr2.ravel(), bins=nbins)
    pxy /= pxy.sum()

    # extract the empirical propabilities of intensities
    # from the joint histogram
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x

    # compute the mutual information
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0  # nonzero value indices
    MI = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    if not normalized:
        return MI

    # normalize, using the sum of their individual entropies H
    def entropy(p):
        nz = p > 0
        assert np.count_nonzero(nz) > 0
        return -np.sum(p[nz] * np.log(p[nz]))

    Hx, Hy = [entropy(p) for p in [px, py]]
    assert (Hx + Hy) > 0
    NMI = 2 * MI / (Hx + Hy)
    return NMI


def is_mesh(structure: Union[list, dict]):
    if isinstance(structure, dict):
        return all(k in structure for k in ["verts", "faces"])
    elif isinstance(structure, list):
        return all(map(is_mesh, structure))
    else:
        return False


def merge_meshes(meshes: list, labels: list = None):
    # merge a list of meshes into one
    # if meshes have no labels, a list of labels of the
    # same length as the number of meshes can
    # be supplied to add a labeling per sub mesh.

    assert len(meshes) > 0
    if len(meshes) == 1:
        return meshes[0]

    assert all('verts' in m for m in meshes)
    assert all('faces' in m for m in meshes)
    has_labels = all('labels' in m for m in meshes)
    if has_labels:
        assert labels is None

    nverts = [0] + [m['verts'].shape[0] for m in meshes[:-1]]
    verts = np.concatenate([m['verts'] for m in meshes])
    faces = np.concatenate([m['faces'] + N for m, N in zip(meshes, nverts)])
    if has_labels:
        labels = np.array([_ for m in meshes for _ in m['labels']])
        return {'verts': verts, 'faces': faces, 'labels': labels}
    elif labels is not None:
        assert len(labels) == len(meshes)
        labels = np.array(
            [labels[i] for i, m in enumerate(meshes) for v in m['verts']]
        )
        return {'verts': verts, 'faces': faces, 'labels': labels}
    else:
        return {'verts': verts, 'faces': faces}


class Species(Enum):

    HOMO_SAPIENS = 1
    RATTUS_NORVEGICUS = 2
    MUS_MUSCULUS = 3
    MACACA_FASCICULARIS = 4
    MACACA_MULATTA = 5
    MACACA_FUSCATA = 6
    CHLOROCEBUS_AETHIOPS_SABAEUS = 7
    CALLITHRIX_JACCHUS = 8

    UNSPECIFIED_SPECIES = 999

    @classmethod
    def decode(cls, spec: Union[str, 'Species', dict], fail_if_not_successful=True):

        MINDS_IDS = {
            "0ea4e6ba-2681-4f7d-9fa9-49b915caaac9": 1,
            "f3490d7f-8f7f-4b40-b238-963dcac84412": 2,
            "cfc1656c-67d1-4d2c-a17e-efd7ce0df88c": 3,
            "c541401b-69f4-4809-b6eb-82594fc90551": 4,
            "745712aa-fad1-47c4-8ab6-088063f78f64": 5,
            "ed8254b1-519c-4356-b1c9-7ead5aa1e3e1": 6,
            "e578d886-c55d-4174-976b-3cf43b142203": 7
        }

        OPENMINDS_IDS = {
            "97c070c6-8e1f-4ee8-9d28-18c7945921dd": 1,
            "ab532423-1fd7-4255-8c6f-f99dc6df814f": 2,
            "d9875ebd-260e-4337-a637-b62fed4aa91d:": 3,
            "0b6df2b3-5297-40cf-adde-9443d3d8214a": 4,
            "3ad33ec1-5152-497d-9352-1cf4497e0edd": 5,
            "2ab3ecf5-76cc-46fa-98ab-309e3fd50f57": 6,
            "b8bf99e7-0914-4b65-a386-d785249725f1": 7
        }

        if isinstance(spec, Species):
            return spec
        elif isinstance(spec, str):
            # split it in case it is an actual uuid from KG
            if spec.split('/')[-1] in MINDS_IDS:
                return cls(MINDS_IDS[spec])
            if spec.split('/')[-1] in OPENMINDS_IDS:
                return cls(OPENMINDS_IDS[spec])
            key = cls.name_to_key(spec)
            if key in cls.__members__.keys():
                return getattr(cls, key)
        else:
            if isinstance(spec, list):
                next_specs = spec
            elif isinstance(spec, dict):
                next_specs = spec.values()
            else:
                raise ValueError(f"Species specification cannot be decoded: {spec}")
            for s in next_specs:
                result = cls.decode(s, fail_if_not_successful=False)
                if result is not None:
                    return result

        # if we get here, spec was not decoded into a species
        if fail_if_not_successful:
            raise ValueError(f"Species specification cannot be decoded: {spec}")
        else:
            return None

    @staticmethod
    def name_to_key(name: str):
        return re.sub(r'\s+', '_', name.strip()).upper()

    @staticmethod
    def key_to_name(key: str):
        return re.sub(r'_', ' ', key.strip()).lower()

    def __str__(self):
        return f"{self.name.lower().replace('_', ' ')}".capitalize()

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self)}"
