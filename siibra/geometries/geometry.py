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
"""Base class of data whose coordinates are the measurement."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
from nibabel.affines import apply_affine

from .providers.provider import GeometryProvider
from ..commons import logger
from ..core import space as _space
from ..core.region import Region
from ..locations import boundingbox as _boundingbox

if TYPE_CHECKING:
    from ..core.space import Space

FragmentSpec = Union[str, Sequence[str]]


def _abbreviate(names: Iterable[str], limit: int = 10) -> str:
    names = list(names)
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, ... ({len(names)} in total)"


class Geometry:
    """
    Coordinate data in a reference space, accessed through providers.

    The coordinates of a geometry are the measurement itself, as for cell
    positions, streamlines or the mesh of a probe. This distinguishes it
    from a :class:`~siibra.volumes.volume.Volume`, which holds values
    sampled on a grid or on a template's surface.

    A geometry may consist of named fragments, such as the bundles of a
    tractogram or the per-region files of a point distribution. Fragments
    can be linked to regions, so that data can be selected by region.

    The data is expected in the geometry's space. An optional affine maps
    the stored coordinates into it, for data that is not.
    """

    def __init__(
        self,
        space_spec: dict,
        providers: List[GeometryProvider],
        name: str = "",
        affine: np.ndarray = None,
        fragment_regions: Dict[str, Sequence[Region]] = None,
    ):
        """
        Parameters
        ----------
        space_spec: dict
            Specification of the reference space, by "@id" or "name".
        providers: list of GeometryProvider
            Sources of the data, at most one per source type. Data is read
            from the first one.
        name: str, default: ""
        affine: numpy.ndarray, default: None
            4x4 affine mapping the stored coordinates into the space.
        fragment_regions: dict, default: None
            Regions each fragment is linked to, by fragment name.
        """
        if len(providers) == 0:
            raise ValueError(f"{self.__class__.__name__} '{name}' needs at least one provider.")
        self._providers: Dict[str, GeometryProvider] = {}
        for provider in providers:
            if provider.srctype in self._providers:
                raise ValueError(
                    f"{self.__class__.__name__} '{name}' has more than one "
                    f"provider of source type '{provider.srctype}'."
                )
            self._providers[provider.srctype] = provider
        self._space_spec = space_spec or {}
        self._name = name
        if affine is not None:
            affine = np.asarray(affine, dtype=float)
            if affine.shape != (4, 4):
                raise ValueError(f"An affine must be a 4x4 matrix, got shape {affine.shape}.")
        self._affine = affine
        self._fragment_regions: Dict[str, List[Region]] = {
            str(fragment): list(regions)
            for fragment, regions in (fragment_regions or {}).items()
        }
        self._fragment_regions_checked = False

    def __str__(self):
        return f"{self.__class__.__name__} '{self.name}'"

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(space_spec={self._space_spec}, "
            f"name='{self.name}', providers={self.providers})>"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def space(self) -> "Space":
        for key in ["@id", "name"]:
            if key in self._space_spec:
                return _space.Space.get_instance(self._space_spec[key])
        return _space.Space(None, "Unspecified space", species=_space.Species.UNSPECIFIED_SPECIES)

    @property
    def species(self):
        return self.space.species

    @property
    def affine(self) -> Optional[np.ndarray]:
        """The affine mapping stored coordinates into the space, if any."""
        return None if self._affine is None else self._affine.copy()

    @property
    def providers(self) -> Dict[str, Union[str, Dict[str, str]]]:
        """The configured sources, by source type."""
        return {srctype: provider._url for srctype, provider in self._providers.items()}

    @property
    def _identity(self) -> str:
        """A stable string identifying the data, e.g. for feature ids."""
        return json.dumps([self._space_spec, self.providers], sort_keys=True)

    @property
    def _provider(self) -> GeometryProvider:
        return next(iter(self._providers.values()))

    @property
    def fragments(self) -> List[str]:
        """
        Names of the fragments of this geometry.

        Depending on the source format, listing them may require loading the data.
        """
        return self._provider.fragments

    @property
    def fragment_regions(self) -> Dict[str, List[Region]]:
        """Regions each fragment is linked to, by fragment name."""
        return {fragment: list(regions) for fragment, regions in self._fragment_regions.items()}

    def get_fragments(self, region: Union[str, Region]) -> List[str]:
        """
        Return the names of the fragments linked to a region or to any of its
        subregions. This does not load any data.

        Parameters
        ----------
        region: str or Region
            A region, or a specification resolved among the regions the
            fragments are linked to.
        """
        if not self._fragment_regions:
            raise ValueError(f"The fragments of {self} are not linked to regions.")
        query = self._resolve_region(region)
        return [
            fragment
            for fragment, regions in self._fragment_regions.items()
            if any(query.includes(r) for r in regions)
        ]

    def fetch(
        self,
        fragment: FragmentSpec = None,
        region: Union[str, Region] = None,
        format: str = None,
    ) -> Any:
        """
        Load the data of this geometry.

        Parameters
        ----------
        fragment: str or list of str, default: None
            Fragment(s) to load. A name matches exactly, or else as a unique
            case-insensitive substring of a fragment name.
        region: str or Region, default: None
            Load the fragments linked to this region or to its subregions.
            Cannot be combined with `fragment`.
        format: str, default: None
            Output format. Subclasses define the formats they support.

        Returns
        -------
        The data in the requested format. Without one, this base class returns
        the provider's own representation, with coordinates as stored.
        """
        fragments = self._select_fragments(fragment, region)
        return self._format(self._provider.fetch(fragments), format)

    def get_coordinates(
        self,
        fragment: FragmentSpec = None,
        region: Union[str, Region] = None,
    ) -> np.ndarray:
        """
        Return the coordinates of all vertices in the space, as an (N, 3) array.

        Parameters
        ----------
        fragment: str or list of str, default: None
            See :meth:`fetch`.
        region: str or Region, default: None
            See :meth:`fetch`.
        """
        fragments = self._select_fragments(fragment, region)
        coordinates = self._provider.get_coordinates(fragments)
        if self._affine is not None:
            coordinates = apply_affine(self._affine, np.asarray(coordinates))
        return coordinates

    def get_boundingbox(
        self,
        fragment: FragmentSpec = None,
        region: Union[str, Region] = None,
    ) -> Optional[_boundingbox.BoundingBox]:
        """
        Return the bounding box of the vertices, or None if there are none.

        Parameters
        ----------
        fragment: str or list of str, default: None
            See :meth:`fetch`.
        region: str or Region, default: None
            See :meth:`fetch`.
        """
        coordinates = self.get_coordinates(fragment=fragment, region=region)
        if len(coordinates) == 0:
            return None
        return _boundingbox.BoundingBox(
            coordinates.min(axis=0), coordinates.max(axis=0), space=self.space
        )

    def _format(self, data: Any, format: Optional[str]) -> Any:
        """Convert the provider's data into the requested output format."""
        if format is None:
            return data
        raise ValueError(f"{self.__class__.__name__} supports no output format '{format}'.")

    def _select_fragments(
        self,
        fragment: Optional[FragmentSpec],
        region: Optional[Union[str, Region]],
    ) -> Optional[List[str]]:
        """Exact names of the fragments to load, or None for all data."""
        if fragment is not None and region is not None:
            raise ValueError("Select data either by fragment or by region, not both.")
        if fragment is None and region is None:
            self._check_fragment_regions()
            return None
        available = self.fragments
        self._check_fragment_regions()
        if fragment is not None:
            specs = [fragment] if isinstance(fragment, str) else list(fragment)
            return list(dict.fromkeys(self._match_fragment(spec, available) for spec in specs))
        selected = [f for f in self.get_fragments(region) if f in available]
        if not selected:
            raise ValueError(f"No fragment of {self} is linked to region '{region}' or its subregions.")
        return selected

    def _match_fragment(self, spec: str, available: List[str]) -> str:
        """Resolve a fragment specification to exactly one fragment name."""
        if not available:
            raise ValueError(f"{self} has no fragments.")
        for candidates in (
            [name for name in available if name == spec],
            [name for name in available if name.lower() == spec.lower()],
            [name for name in available if spec.lower() in name.lower()],
        ):
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(
                    f"Fragment '{spec}' of {self} is ambiguous; "
                    f"it matches {_abbreviate(candidates)}."
                )
        raise ValueError(
            f"{self} has no fragment '{spec}'. Available: {_abbreviate(available)}."
        )

    def _resolve_region(self, region: Union[str, Region]) -> Region:
        """Resolve a region specification among the roots of the linked regions."""
        if isinstance(region, Region):
            return region
        if not isinstance(region, str):
            raise TypeError(f"Cannot select fragments by region {region!r}.")
        roots: List[Region] = []
        for regions in self._fragment_regions.values():
            for r in regions:
                if r.root not in roots:
                    roots.append(r.root)
        for root in roots:
            if hasattr(root, "get_region"):  # a parcellation
                try:
                    return root.get_region(region)
                except ValueError:
                    continue
            matches = root.find(region, filter_children=True)
            # find() also returns parents whose children all match, so prefer an exact name
            exact = [m for m in matches if m.name.lower() == region.lower()]
            if len(exact) == 1:
                return exact[0]
            if len(matches) == 1:
                return matches[0]
        raise ValueError(f"Cannot resolve region '{region}' among the regions linked to {self}.")

    def _check_fragment_regions(self):
        """
        Warn once if the fragments linked to regions differ from the fragments
        found in the data. Called when data is accessed, since listing the
        fragments may require loading it.
        """
        if self._fragment_regions_checked or not self._fragment_regions:
            return
        self._fragment_regions_checked = True
        available = self.fragments
        without_data = [f for f in self._fragment_regions if f not in available]
        without_regions = [f for f in available if f not in self._fragment_regions]
        if without_data:
            logger.warning(
                f"{len(without_data)} fragment(s) linked to regions are not in the "
                f"data of {self}: {_abbreviate(without_data)}"
            )
        if without_regions:
            logger.warning(
                f"{len(without_regions)} fragment(s) of {self} are not linked to "
                f"any region: {_abbreviate(without_regions)}"
            )
