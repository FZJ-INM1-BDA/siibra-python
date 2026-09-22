# Copyright 2018-2026
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

import os as _os

from .commons import (
    logger,
    QUIET,
    VERBOSE,
    MapType,
    set_log_level,
    __version__
)
from . import configuration, features, livequeries
from .configuration import factory
from .core import (
    atlas as _atlas,
    parcellation as _parcellation,
    space as _space
)
from .volumes import parcellationmap as _parcellationmap
from .retrieval.requests import CACHE as cache
from .retrieval.cache import Warmup, WarmupLevel
from .locations import Point, PointCloud, Plane, BoundingBox


logger.info(f"Version: {__version__}\nPlease file issues at https://github.com/FZJ-INM1-BDA/siibra-python.")

# forward access to some functions
find_regions = _parcellation.find_regions
from_json = factory.Factory.from_json


def __getattr__(attr: str):
    # lazy loading of some classes for package-level functions.
    if attr == 'atlases':
        return _atlas.Atlas.registry()
    elif attr == 'spaces':
        return _space.Space.registry()
    elif attr == 'parcellations':
        return _parcellation.Parcellation.registry()
    elif attr == 'maps':
        return _parcellationmap.Map.registry()
    elif attr == 'use_configuration':
        return configuration.Configuration.use_configuration
    elif attr == 'extend_configuration':
        return configuration.Configuration.extend_configuration
    else:
        raise AttributeError(f"No such attribute: {__name__}.{attr}")


# convenient access to reference space templates
def get_template(space_spec: str, variant: str = None):
    """
    Get the reference template for a brain reference space.

    Parameters
    ----------
    space_spec : str
        Specification of the reference space.
    variant: str, optional
        Some templates are provided in different variants, e.g.
        freesurfer is available as either white matter, pial or
        inflated surface for left and right hemispheres (6 variants).
        This field could be used to request a specific variant.
        Per default, the first found variant is returned.

    Returns
    -------
    siibra.volumes.volume.Volume or None
        The reference template associated with the requested space, or
        ``None`` if no suitable template is available.
    """
    return (
        _space.Space
        .get_instance(space_spec)
        .get_template(variant=variant)
    )


# convenient access to parcellation maps
def get_map(
    parcellation: str,
    space: str,
    maptype: MapType = MapType.LABELLED,
    spec: str = "",
):
    """
    Get a parcellation map in a specified reference space.

    Parameters
    ----------
    parcellation: str
        Specification of the parcellation containing the region. This
        may be a partial parcellation name.
    space: str
        Specification of the reference space in which the map is
        requested, such as its name or identifier.
    maptype: MapType, default: ``MapType.LABELLED``
        Type of parcellation map to retrieve. Use
        ``MapType.LABELLED`` for a discrete label map or
        ``MapType.STATISTICAL`` for statistical maps.
    spec: str
        Used to distinguish between multiple matching maps of same parcellation,
        space, and map type.

    Returns
    -------
    siibra.volumes.parcellationmap.Map or siibra.volumes.sparsemap.SparseMap
        The requested parcellation map. Statistical maps may be
        represented as a sparse collection of region maps.
    """
    return (
        _parcellation.Parcellation
        .get_instance(parcellation)
        .get_map(space=space, maptype=maptype, spec=spec)
    )


# convenient access to regions of a parcellation
def get_region(parcellation: str, region: str):
    """
    Get a brain region from a parcellation.

    Parameters
    ----------
    parcellation: str
        Specification of the parcellation containing the region. This
        may be a partial parcellation name.
    region: str
        Region specification to resolve. Inexact names and identifier
        keys may be accepted if they uniquely identify a region.

    Returns
    -------
    siibra.core.region.Region
        The region matching the supplied specification.

    Raises
    ------
    ValueError
        If the region specification cannot be matched to a region.
    RuntimeError
        If the specification produces multiple matches that cannot be
        resolved to a single region.
    """
    return (
        _parcellation.Parcellation
        .get_instance(parcellation)
        .get_region(region)
    )


# convenient access to a parcellation
def get_parcellation(parcellation: str):
    """
    Get a preconfigured brain parcellation.

    Parameters
    ----------
    parcellation : str
        Specification of the parcellation containing the region. This
        may be a partial parcellation name.

    Returns
    -------
    siibra.core.parcellation.Parcellation
        The parcellation matching the supplied specification.
    """
    return _parcellation.Parcellation.get_instance(parcellation)


# convenient access to spaces
def get_space(space: str):
    """
    Get a preconfigured brain reference space.

    Parameters
    ----------
    space : str
        Specification of the reference space.

    Returns
    -------
    siibra.core.space.Space
        The reference space matching the supplied specification.
    """
    return _space.Space.get_instance(space)


def set_feasible_download_size(maxsize_gbyte):
    """
    Set the maximum download size considered feasible.

    This changes the global size threshold used by volume retrieval
    operations when assessing whether a download is feasible.

    Parameters
    ----------
    maxsize_gbyte : float
        Maximum feasible download size in gibibytes (GiB).

    Returns
    -------
    None
        This function modifies the global download-size threshold
        in place.
    """
    from .volumes import volume

    volume.gbyte_feasible = maxsize_gbyte
    logger.info(f"Set feasible download size to {maxsize_gbyte} GiB.")


def set_cache_size(maxsize_gbyte: int):
    """
    siibra runs maintenance on its local cache to keep it under a predetermined
    size of 2 gigabytes. This method changes the cache size.

    Parameters
    ----------
    maxsize_gbyte : int
    """
    assert maxsize_gbyte >= 0
    cache.SIZE_GIB = maxsize_gbyte
    logger.info(f"Set cache size to {maxsize_gbyte} GiB.")


if "SIIBRA_CACHE_SIZE_GIB" in _os.environ:
    set_cache_size(float(_os.environ.get("SIIBRA_CACHE_SIZE_GIB")))


def warm_cache(level=WarmupLevel.INSTANCE):
    """
    Preload preconfigured siibra concepts.

    Siibra relies on preconfigurations that simplify integrating various
    concepts such as parcellations, reference spaces, and multimodal data
    features. By preloading the instances, siibra commits all preconfigurations
    to the memory at once instead of committing them when required.
    """
    Warmup.warmup(level)


def __dir__():
    return [
        "atlases",
        "spaces",
        "parcellations",
        "features",
        "use_configuration",
        "extend_configuration",
        "get_region",
        "find_regions",
        "get_parcellation",
        "get_space",
        "get_map",
        "get_template",
        "MapType",
        "Point",
        "PointCloud",
        "BoundingBox",
        "QUIET",
        "VERBOSE",
        "vocabularies",
        "__version__",
        "cache",
        "warm_cache",
        "set_cache_size",
        "from_json",
    ]
