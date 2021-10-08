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

from .concept import AtlasConcept, provide_registry
from .space import Space
from .parcellation import Parcellation

from ..commons import MapType, logger, Registry

VERSION_BLACKLIST_WORDS = ["beta", "rc", "alpha"]


@provide_registry
class Atlas(
    AtlasConcept, bootstrap_folder="atlases", type_id="juelich/iav/atlas/v1.0.0"
):
    """
    Main class for an atlas, providing access to feasible
    combinations of available parcellations and reference
    spaces, as well as common functionalities of those.
    """

    def __init__(self, identifier, name):
        """Construct an empty atlas object with a name and identifier."""

        AtlasConcept.__init__(self, identifier, name, dataset_specs=[])

        self._parcellations = []  # add with _add_parcellation
        self._spaces = []  # add with _add_space

    def _register_space(self, space):
        """Registers another reference space to the atlas."""
        self._spaces.append(space)

    def _register_parcellation(self, parcellation):
        """Registers another parcellation to the atlas."""
        self._parcellations.append(parcellation)

    @property
    def spaces(self):
        """Access a registry of reference spaces supported by this atlas."""
        return Registry(
            elements={s.key: s for s in self._spaces},
            matchfunc=Space.match_spec,
        )

    @property
    def parcellations(self):
        """Access a registry of parcellations supported by this atlas."""
        return Registry(
            elements={p.key: p for p in self._parcellations},
            matchfunc=Parcellation.match_spec,
        )

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if obj.get("@type") != "juelich/iav/atlas/v1.0.0":
            raise ValueError(
                f"{cls.__name__} construction attempt from invalid json format (@type={obj.get('@type')}"
            )
        if all(["@id" in obj, "spaces" in obj, "parcellations" in obj]):
            atlas = cls(obj["@id"], obj["name"])
            for space_id in obj["spaces"]:
                if not Space.REGISTRY.provides(space_id):
                    raise ValueError(
                        f"Invalid atlas configuration for {str(atlas)} - space {space_id} not known"
                    )
                atlas._register_space(Space.REGISTRY[space_id])
            for parcellation_id in obj["parcellations"]:
                if not Parcellation.REGISTRY.provides(parcellation_id):
                    raise ValueError(
                        f"Invalid atlas configuration for {str(atlas)} - parcellation {parcellation_id} not known"
                    )
                atlas._register_parcellation(Parcellation.REGISTRY[parcellation_id])
            return atlas
        return obj

    def get_parcellation(self, parcellation=None):
        """Returns a valid parcellation object defined by the atlas.
        If no specification is provided, the default is returned."""

        if parcellation is None:
            parcellation_obj = self._parcellations[0]
            if len(self._parcellations) > 1:
                logger.info(
                    f"No parcellation specified, using default '{parcellation_obj.name}'."
                )
        else:
            parcellation_obj = self.parcellations[parcellation]
            if parcellation_obj not in self._parcellations:
                raise ValueError(
                    f"Parcellation {parcellation_obj.name} not supported by atlas {self.name}."
                )
        return parcellation_obj

    def get_space(self, space=None):
        """Returns a valid reference space object defined by the atlas.
        If no specification is provided, the default is returned.

        Parameters:
            space: Space, or string specification of a space
        """
        if space is None:
            space_obj = self._spaces[0]
            if len(self._spaces) > 1:
                logger.info(f"No space specified, using default '{space_obj.name}'.")
        else:
            space_obj = Space.REGISTRY[space]
            if space_obj not in self._spaces:
                raise ValueError(
                    f"Space {space_obj.name} not supported by atlas {self.name}."
                )
        return space_obj

    def get_map(
        self,
        space: Space = None,
        parcellation: Parcellation = None,
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
            Type of the map (labelled or continuous/probabilistic)

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
        return self.get_parcellation(parcellation).decode_region(region)

    def get_template(self, space: Space = None):
        """
        Returns the reference template in the desired reference space.
        If no reference space is given, the default from `Atlas.space()` is used.
        """
        return self.get_space(space).get_template()

    def get_voi(self, space: Space, point1: tuple, point2: tuple):
        """Get a volume of interest spanned by two points in the given reference space.

        Args:
            space (Space or str): The target reference space, or a string specification of the space
            point1 (Tuple): A 3D coordinate given in this reference space
            point2 (Tuple): Another 3D coordinate given in this reference space

        Returns:
            Bounding Box
        """
        spaceobj = Space.REGISTRY[space]
        if spaceobj not in self._spaces:
            raise ValueError(
                f"Requested space {space} not supported by {self.__class__.__name__} {self.name}."
            )
        return spaceobj.get_bounding_box(point1, point2)

    def find_regions(
        self,
        regionspec,
        all_versions=False,
        filter_children=True,
        build_groups=False,
        groupname=None
    ):
        """
        Find regions with the given specification in all
        parcellations offered by the atlas.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex
            - a region object
        all_versions : Bool, default: False
            If True, matched regions for all versions of a parcellation are returned.
        filter_children : Boolean
            If true, children of matched parents will not be returned
        build_groups : Boolean, default: False
            If true, a group region will be formed per parellations
            which includes the resulting elements,
            in case they do not have a single common parent anyway.
        groupname : str (optional)
            Name of the resulting group region, if build_groups is True

        Yield
        -----
        list of matching regions
        """
        result = []
        for p in self._parcellations:
            if p.is_newest_version or all_versions:
                match = p.find_regions(
                    regionspec,
                    filter_children=filter_children,
                    build_group=build_groups,
                    groupname=groupname
                )
                if build_groups:
                    if match is not None:
                        result.append(match)
                else:
                    result.extend(match)
        return result

    def __lt__(self, other):
        """
        We sort atlases by their names
        """
        return self.name < other.name