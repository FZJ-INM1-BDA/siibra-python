# Copyright 2018-2025
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
"""Provides reference systems for brains."""

from typing import List

from . import concept, space as _space, parcellation as _parcellation
from ..commons import MapType, logger, InstanceTable, Species


VERSION_BLACKLIST_WORDS = ["beta", "rc", "alpha"]


class Atlas(concept.AtlasConcept, configuration_folder="atlases"):
    """
    Main class for an atlas, providing access to feasible
    combinations of available parcellations and reference
    spaces, as well as common functionalities of those.
    """

    def __init__(self, identifier: str, name: str, species: Species, **kwargs):
        """Construct an empty atlas object with a name and identifier."""

        concept.AtlasConcept.__init__(
            self,
            identifier=identifier,
            name=name,
            species=species,
            **kwargs
        )
        self._parcellation_ids: List[str] = []
        self._space_ids: List[str] = []

    def _register_space(self, space_id: str):
        self._space_ids.append(space_id)

    def _register_parcellation(self, parcellation_id: str):
        self._parcellation_ids.append(parcellation_id)

    @property
    def spaces(self):
        """Access a registry of reference spaces supported by this atlas."""
        return InstanceTable[_space.Space](
            elements={s.key: s for s in _space.Space.registry() if s.id in self._space_ids},
            matchfunc=_space.Space.match,
        )

    @property
    def parcellations(self):
        """Access a registry of parcellations supported by this atlas."""
        return InstanceTable[_parcellation.Parcellation](
            elements={p.key: p for p in _parcellation.Parcellation.registry() if p.id in self._parcellation_ids},
            matchfunc=_parcellation.Parcellation.match,
        )

    def get_parcellation(self, parcellation=None) -> "_parcellation.Parcellation":
        """
        Returns a valid parcellation object defined by the atlas. If no
        specification is provided, the default is returned.

        Parameters
        ----------
        parcellation: str, Parcellation
            specification of a parcellation or a parcellation object

        Returns
        -------
            Parcellation
        """

        if parcellation is None:
            parcellation_obj = self.parcellations[self._parcellation_ids[0]]
            if len(self._parcellation_ids) > 1:
                logger.info(f"No parcellation specified, using default: '{parcellation_obj.name}'.")
            return parcellation_obj

        if isinstance(parcellation, _parcellation.Parcellation):
            assert parcellation in self.parcellations
            return parcellation

        return self.parcellations[parcellation]

    def get_space(self, space=None) -> "_space.Space":
        """
        Returns a valid reference space object defined by the atlas. If no
        specification is provided, the default is returned.

        Parameters
        ----------
        space: str, Space
            specification of a space or a space object

        Returns
        -------
            Space
        """
        if space is None:
            space_obj = self.spaces[self._space_ids[0]]
            if len(self._space_ids) > 1:
                logger.info(f"No space specified, using default '{space_obj.name}'.")
            return space_obj

        if isinstance(space, _space.Space):
            assert space in self.spaces
            return space

        return self.spaces[space]

    def get_map(
        self,
        space: _space.Space = None,
        parcellation: _parcellation.Parcellation = None,
        maptype: MapType = MapType.LABELLED,
    ):
        """
        Returns a parcellation map in the given space.

        Parameters
        ----------

        space: Space
            The requested reference space. If None, the default is used.
        parcellation: Parcellation
            The requested parcellation. If None, the default is used.
        maptype: MapType
            Type of the map (labelled or statistical/probabilistic)

        Returns
        -------
        ParcellationMap
        """
        parc_obj = self.get_parcellation(parcellation)
        space_obj = self.get_space(space)
        return parc_obj.get_map(space=space_obj, maptype=maptype)

    def get_region(self, region, parcellation=None):
        """
        Returns a valid Region object matching the given specification.

        Parameters
        ----------
        region : str or Region
            Key, approximate name, id or instance of a brain region
        parcellation : str or Parcellation
            Key, approximate name, id or instance of a brain parcellation.
            If None, the default is used.
        """
        return self.get_parcellation(parcellation).get_region(region)

    def get_template(self, space: _space.Space = None, variant: str = None):
        """
        Returns the reference template in the desired reference space.
        If no reference space is given, the default from `Atlas.space()` is used.

        Parameters
        ----------
        space: Space
            The desired reference space
        variant: str (optional)
            Some templates are provided in different variants, e.g.
            freesurfer is available as either white matter, pial or
            inflated surface for left and right hemispheres (6 variants).
            This field could be used to request a specific variant.
            Per default, the first found variant is returned.
        """
        return self.get_space(space).get_template(variant=variant)

    def get_voi(self, space: _space.Space, point1: tuple, point2: tuple):
        """Get a volume of interest spanned by two points in the given reference space.

        Parameters
        ----------
        space: Space, str
            The target reference space, or a string specification of the space
        point1: Tuple
            A 3D coordinate given in this reference space
        point2: Tuple
            Another 3D coordinate given in this reference space

        Returns
        -------
            BoundingBox
        """
        return self.get_template(space).get_boundingbox(point1, point2)

    def find_regions(
        self,
        regionspec: str,
        all_versions: bool = False,
        filter_children: bool = True,
        find_topmost: bool = False
    ):
        """
        Find regions with the given specification in all parcellations offered
        by the atlas.

        Parameters
        ----------
        regionspec: str, regex
            - a string with a possibly inexact name (matched both against the name and the identifier key)
            - a string in '/pattern/flags' format to use regex search (acceptable flags: aiLmsux, see at https://docs.python.org/3/library/re.html#flags)
            - a regex applied to region names
        all_versions : Bool, default: False
            If True, matched regions for all versions of a parcellation are returned.
        filter_children : bool, default: True
            If False, children of matched parents will be returned.
        find_topmost : bool, default: False
            If True (requires `filter_children=True`), will return parent
            structures if all children are matched, even though the parent
            itself might not match the specification.

        Returns
        -------
        list[Region]
            list of regions matching to the regionspec
        """
        result = []
        for p in self.parcellations:
            if p.is_newest_version or all_versions:
                result.extend(
                    p.find(
                        regionspec=regionspec,
                        filter_children=filter_children,
                        find_topmost=find_topmost
                    )
                )
        return result
