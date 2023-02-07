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
"""Provides reference systems for brains."""
from . import concept, space as _space, parcellation as _parcellation

from ..commons import MapType, logger, InstanceTable, Species

from typing import List


VERSION_BLACKLIST_WORDS = ["beta", "rc", "alpha"]


class Atlas(concept.AtlasConcept, configuration_folder="atlases"):
    """
    Main class for an atlas, providing access to feasible
    combinations of available parcellations and reference
    spaces, as well as common functionalities of those.
    """

    def __init__(self, identifier: str, name: str, species: Species):
        """Construct an empty atlas object with a name and identifier."""

        concept.AtlasConcept.__init__(
            self,
            identifier=identifier,
            name=name,
            species=species
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

    def get_parcellation(self, parcellation=None):
        """Returns a valid parcellation object defined by the atlas.
        If no specification is provided, the default is returned."""

        if parcellation is None:
            parcellation_obj = self.parcellations[self._parcellation_ids[0]]
            if len(self._parcellation_ids) > 1:
                logger.info(f"No parcellation specified, using default: '{parcellation_obj.name}'.")
            return parcellation_obj

        if isinstance(parcellation, _parcellation.Parcellation):
            assert parcellation in self.parcellations
            return parcellation

        return self.parcellations[parcellation]

    def get_space(self, space=None):
        """Returns a valid reference space object defined by the atlas.
        If no specification is provided, the default is returned.

        Parameters:
            space: Space, or string specification of a space
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
        """Returns a parcellation map in the given space.

        Parameters
        ----------

        space : Space
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

        Args:
            space (Space or str): The target reference space, or a string specification of the space
            point1 (Tuple): A 3D coordinate given in this reference space
            point2 (Tuple): Another 3D coordinate given in this reference space

        Returns:
            Bounding Box
        """
        return self.get_space(space).get_bounding_box(point1, point2)

    def find_regions(
        self,
        regionspec,
        all_versions=False,
        filter_children=True,
        **kwargs
    ):
        """
        Find regions with the given specification in all
        parcellations offered by the atlas. Additional kwargs
        are passed on to Parcellation.find().

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex
            - a region object
        all_versions : Bool, default: False
            If True, matched regions for all versions of a parcellation are returned.

        Yield
        -----
        list of matching regions
        """
        result = []
        for p in self._parcellation_ids:
            parcobj = _parcellation.Parcellation.get_instance(p)
            if parcobj.is_newest_version or all_versions:
                match = parcobj.find(regionspec, filter_children=filter_children, **kwargs)
                result.extend(match)
        return result

    def __lt__(self, other: 'Atlas'):
        """
        We sort atlases by their names
        """
        return self.name < other.name
