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
"""A specific mesh or 3D array."""
from .. import logger
from ..retrieval import requests
from ..locations import boundingbox as _boundingbox
from ..core import space

import nibabel as nib
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Set, TYPE_CHECKING
import json
from time import sleep

if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsDataset
    TypeDataset = EbrainsDataset


class ColorVolumeNotSupported(NotImplementedError):
    pass


class Volume:
    """
    A volume is a specific mesh or 3D array,
    which can be accessible via multiple providers in different formats.
    """

    IMAGE_FORMATS = [
        "nii",
        "zip/nii",
        "neuroglancer/precomputed"
    ]

    MESH_FORMATS = [
        "neuroglancer/precompmesh",
        "neuroglancer/precompmesh/surface",
        "gii-mesh",
        "gii-label",
        "freesurfer-annot",
        "zip/freesurfer-annot",
    ]

    SUPPORTED_FORMATS = IMAGE_FORMATS + MESH_FORMATS

    _FORMAT_LOOKUP = {
        "image": IMAGE_FORMATS,
        "mesh": MESH_FORMATS,
        "surface": MESH_FORMATS,
        "nifti": ["nii", "zip/nii"],
        "nii": ["nii", "zip/nii"]
    }

    def __init__(
        self,
        space_spec: dict,
        providers: List['VolumeProvider'],
        name: str = "",
        variant: str = None,
        datasets: List['TypeDataset'] = [],
    ):
        self._name_cached = name  # see lazy implementation below
        self._space_spec = space_spec
        self.variant = variant
        self._providers: Dict[str, 'VolumeProvider'] = {}
        self.datasets = datasets
        for provider in providers:
            srctype = provider.srctype
            assert srctype not in self._providers
            self._providers[srctype] = provider
        if len(self._providers) == 0:
            logger.debug(f"No provider for volume {self}")

    @property
    def name(self):
        """
        Allows derived classes to implement a lazy name specification.
        """
        return self._name_cached

    @property
    def providers(self):
        def concat(url: Union[str, Dict[str, str]], concat: str):
            if isinstance(url, str):
                return url + concat
            return {key: url[key] + concat for key in url}
        return {
            srctype: concat(prov._url, f" {prov.label}" if hasattr(prov, "label") else "")
            for srctype, prov in self._providers.items()
        }

    @property
    def boundingbox(self):
        for provider in self._providers.values():
            try:
                bbox = provider.boundingbox
                if bbox.space is None:  # provider does usually not know the space!
                    bbox.space = self.space
                    bbox.minpoint.space = self.space
                    bbox.maxpoint.space = self.space
            except NotImplementedError as e:
                print(str(e))
                continue
            return bbox
        raise RuntimeError(f"No bounding box specified by any volume provider of {str(self)}")

    @property
    def formats(self) -> Set[str]:
        result = set()
        for fmt in self._providers:
            result.add(fmt)
            result.add('mesh' if fmt in self.MESH_FORMATS else 'image')
        return result

    @property
    def provides_mesh(self):
        return any(f in self.MESH_FORMATS for f in self.formats)

    @property
    def provides_image(self):
        return any(f in self.IMAGE_FORMATS for f in self.formats)

    @property
    def fragments(self) -> Dict[str, List[str]]:
        result = {}
        for srctype, p in self._providers.items():
            t = 'mesh' if srctype in self.MESH_FORMATS else 'image'
            for fragment_name in p.fragments:
                if t in result:
                    result[t].append(fragment_name)
                else:
                    result[t] = [fragment_name]
        return result

    @property
    def space(self):
        for key in ["@id", "name"]:
            if key in self._space_spec:
                return space.Space.get_instance(self._space_spec[key])
        return space.Space(None, "Unspecified space", species=space.Species.UNSPECIFIED_SPECIES)

    def __str__(self):
        if self.space is None:
            return f"{self.__class__.__name__} '{self.name}'"
        else:
            return f"{self.__class__.__name__} '{self.name}' in space '{self.space.name}'"

    def __repr__(self):
        return self.__str__()

    def fetch(
        self,
        format: str = None,
        **kwargs
    ):
        """
        Fetch a volumetric or surface representation from one of the providers.

        Parameters
        ----------
        format: str, default=None
            Requested format. If `None`, the first supported format matching in
            `self.formats` is tried, starting with volumetric formats.
            It can be explicitly specified as:
                - 'surface' or 'mesh' to fetch a surface format
                - 'volumetric' or 'voxel' to fetch a volumetric format
                - supported format types, see SUPPORTED_FORMATS. This includes
                'nii', 'zip/nii', 'neuroglancer/precomputed', 'gii-mesh',
                'neuroglancer/precompmesh', 'gii-label'

        Returns
        -------
        An image or mesh
        """

        if format is None:
            requested_formats = self.SUPPORTED_FORMATS
        elif format in self._FORMAT_LOOKUP:  # allow use of aliases
            requested_formats = self._FORMAT_LOOKUP[format]
        elif format in self.SUPPORTED_FORMATS:
            requested_formats = [format]
        else:
            raise ValueError(f"Invalid format requested: {format}")

        # select the first source unless the user specifically requests a format
        for fmt in requested_formats:
            if fmt in self.formats:
                selected_format = fmt
                logger.debug(f"Requested format was '{format}', selected format is '{selected_format}'")
                break
        else:
            raise ValueError(f"Invalid format requested: {format}")

        # ensure the voi is inside the template
        voi = kwargs.get("voi", None)
        if isinstance(voi, _boundingbox.BoundingBox) and voi.space is not None:
            tmplt_bbox = voi.space.get_template().boundingbox
            intersection_bbox = voi.intersection(tmplt_bbox)
            if intersection_bbox is None:
                raise RuntimeError(f"voi provided ({voi}) lies out side the voxel space of the {voi.space.name} template.")
            if intersection_bbox.minpoint != voi.minpoint or intersection_bbox.maxpoint != voi.maxpoint:
                logger.info(
                    f"Since provided voi lies outside the template ({voi.space}) it is clipped as: {intersection_bbox}"
                )
                kwargs["voi"] = intersection_bbox

        # try the selected format only
        for try_count in range(6):
            try:
                if selected_format in ["gii-label", "freesurfer-annot", "zip/freesurfer-annot"]:
                    tpl = self.space.get_template(variant=kwargs.get('variant'))
                    mesh = tpl.fetch(**kwargs)
                    labels = self._providers[selected_format].fetch(**kwargs)
                    return dict(**mesh, **labels)
                else:
                    return self._providers[selected_format].fetch(**kwargs)
            except requests.SiibraHttpRequestError as e:
                if e.status_code == 429:  # too many requests
                    sleep(0.1)
                logger.error(f"Cannot access {self._providers[selected_format]}", exc_info=(try_count == 5))
        if format is None and len(self.formats) > 1:
            logger.info(
                f"No format was specified and auto-selected format '{selected_format}' "
                "was unsuccesful. You can specify another format from "
                f"{set(self.formats) - set(selected_format)} to try.")
        return None


class Subvolume(Volume):
    """
    Wrapper class for exposing a z level of a 4D volume to be used like a 3D volume.
    """

    def __init__(self, parent_volume: Volume, z: int):
        Volume.__init__(
            self,
            space_spec=parent_volume._space_spec,
            providers=[
                SubvolumeProvider(p, z=z)
                for p in parent_volume._providers.values()
            ]
        )


# TODO add mesh primitive. Check nibabel implementation? Use trimesh? Do we want to add yet another dependency?
VolumeData = Union[nib.Nifti1Image, Dict]


class VolumeProvider(ABC):

    def __init_subclass__(cls, srctype: str) -> None:
        cls.srctype = srctype
        return super().__init_subclass__()

    @property
    @abstractmethod
    def boundingbox(self) -> _boundingbox.BoundingBox:
        raise NotImplementedError

    @property
    def fragments(self) -> List[str]:
        return []

    @abstractmethod
    def fetch(self, *args, **kwargs) -> VolumeData:
        raise NotImplementedError

    @property
    @abstractmethod
    def _url(self) -> Union[str, Dict[str, str]]:
        """
        This is needed to provide urls to applications that can utilise such resources directly.
        e.g. siibra-api
        """
        return {}


class SubvolumeProvider(VolumeProvider, srctype="subvolume"):
    """
    This provider wraps around an existing volume provider,
    but is preconfigured to always fetch a fixed subvolume.
    The primary use is to provide a fixed z coordinate
    of a 4D volume provider as a 3D volume under the
    interface of a normal volume provider.
    """

    _USE_CACHING = False
    _FETCHED_VOLUMES = {}

    class UseCaching:
        def __enter__(self):
            SubvolumeProvider._USE_CACHING = True

        def __exit__(self, et, ev, tb):
            SubvolumeProvider._USE_CACHING = False
            SubvolumeProvider._FETCHED_VOLUMES = {}

    def __init__(self, parent_provider: VolumeProvider, z: int):
        VolumeProvider.__init__(self)
        self.provider = parent_provider
        self.srctype = parent_provider.srctype
        self.z = z

    @property
    def boundingbox(self) -> _boundingbox.BoundingBox:
        return self.provider.boundingbox

    def fetch(self, **kwargs):
        # activate caching at the caller using "with SubvolumeProvider.UseCaching():""
        if self.__class__._USE_CACHING:
            data_key = json.dumps(self.provider._url, sort_keys=True) \
                + json.dumps(kwargs, sort_keys=True)
            if data_key not in self.__class__._FETCHED_VOLUMES:
                vol = self.provider.fetch(**kwargs)
                self.__class__._FETCHED_VOLUMES[data_key] = vol
            vol = self.__class__._FETCHED_VOLUMES[data_key]
        else:
            vol = self.provider.fetch(**kwargs)
        return vol.slicer[:, :, :, self.z]

    def __getattr__(self, attr):
        return self.provider.__getattribute__(attr)

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return super()._url
