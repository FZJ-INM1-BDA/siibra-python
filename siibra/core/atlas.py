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

from os import path
from typing import Any, Dict, List
from pydantic.fields import Field

from pydantic.main import BaseModel
from siibra.openminds.SANDS.v3.atlas import brainAtlas
from siibra.openminds.common import CommonConfig
from .concept import AtlasConcept, RegistrySrc, provide_openminds_registry, main_openminds_registry
from .space import Space
from .parcellation import Parcellation, BrainAtlas

from ..commons import MapType, TypedRegistry, logger, Registry

VERSION_BLACKLIST_WORDS = ["beta", "rc", "alpha"]

class MultiLevelAtlas(BaseModel):
    id: str
    name: str
    order: float = 5.0
    brain_atlases: List[BrainAtlas] = Field(
        ...,
        alias='brainAtlases'
    )
    species: List[Dict[str, str]] = []

    @property
    def parcellations(self) -> TypedRegistry[Parcellation]:

        brainatlases = [brainatlas for brainatlas in self.brain_atlases]
        parcs_dict: Dict[str, 'Parcellation'] = {
            ver.get("@id"): main_openminds_registry[ver.get("@id")]
            for ba in brainatlases
            for ver in ba.has_version
        }

        assert all([
            isinstance(parc, Parcellation)
            for _, parc in parcs_dict.items()
        ]), f'{self.name} MultiLevelAtlas.spaces, not all parcs_dict are Parcellation instance'
        
        return Registry(
            elements=parcs_dict,
            get_aliases=Parcellation.get_aliases,
        )

    @property
    def spaces(self) -> TypedRegistry[Space]:

        parcs: List['Parcellation'] = [
            parc
            for parc in self.parcellations
        ]

        space_ids: set[str] = set([
            parc.coordinate_space.get("@id")
            for parc in parcs
        ])

        space_dict = {
            space_id: main_openminds_registry[{ "@id": space_id }]
            for space_id in space_ids
        }
        return Registry(
            elements=space_dict,
            get_aliases=Space.get_aliases
        )

    def __hash__(self):
        return hash(self.id)

@provide_openminds_registry(
    bootstrap_folder="atlases",
    registry_src=RegistrySrc.GITLAB,
)
class Atlas(
    MultiLevelAtlas,
    AtlasConcept,
):
    """
    Main class for an atlas, providing access to feasible
    combinations of available parcellations and reference
    spaces, as well as common functionalities of those.
    """

    _species = None

    def __init__(self, **kwargs):
        """Construct an empty atlas object with a name and identifier."""
        brainAtlas.Model.__init__(self, **kwargs)
        AtlasConcept.__init__(self, self.id, self.name, dataset_specs=[])
        for atlas in self.brain_atlases:
            for ver in atlas.has_version:
                p: Parcellation = main_openminds_registry[ver]
                p._atlases.add(self)


    @staticmethod
    def get_species_data(species_str: str):
        if species_str == 'human':
            return {
                '@id': 'https://nexus.humanbrainproject.org/v0/data/minds/core/species/v1.0.0/0ea4e6ba-2681-4f7d-9fa9-49b915caaac9',
                'name': 'Homo sapiens'
            }
        if species_str == 'rat':
            return {
                '@id': 'https://nexus.humanbrainproject.org/v0/data/minds/core/species/v1.0.0/f3490d7f-8f7f-4b40-b238-963dcac84412',
                'name': 'Rattus norvegicus'
            }
        if species_str == 'mouse':
            return {
                '@id': 'https://nexus.humanbrainproject.org/v0/data/minds/core/species/v1.0.0/cfc1656c-67d1-4d2c-a17e-efd7ce0df88c',
                'name': 'Mus musculus'
            }
        # TODO this may not be correct. Wait for feedback and get more accurate
        if species_str == 'monkey':
            return {
                '@id': 'https://nexus.humanbrainproject.org/v0/data/minds/core/species/v1.0.0/3f75b0ad-dbcd-464e-b614-499a1b9ae86b',
                'name': 'Primates'
            }

        raise ValueError(f'species with spec {species_str} cannot be decoded')

    Config = CommonConfig

    @staticmethod
    def parse_legacy_id(atlas_id: str):
        base_id = path.basename(atlas_id)
        return f'https://openminds.ebrains.eu/instances/BrainAtlas/{base_id}'

    @classmethod
    def parse_legacy(Cls, json_input: Dict[str, Any]) -> List['Atlas']:

        possible_parc_ids = [
            Parcellation.parse_legacy_id(parc_id, spc_id)
            for parc_id in json_input.get('parcellations')
            for spc_id in json_input.get('spaces')
        ]

        actual_parc_ids = [
            parc_id
            for parc_id in possible_parc_ids
            if main_openminds_registry.provides(parc_id)
        ]

        brainatlases = [
            brainatlas
            for brainatlas in main_openminds_registry
            if isinstance(brainatlas, BrainAtlas)
            and any([
                version.get("@id") in actual_parc_ids
                for version in brainatlas.has_version
            ])
        ]

        return [Cls(
            id=Atlas.parse_legacy_id(json_input.get('@id')),
            name=json_input.get('name'),
            order=json_input.get('order'),
            brain_atlases=brainatlases,
            species=[Atlas.get_species_data(json_input.get("species"))],
        )]

    @staticmethod
    def get_aliases(key: str, atlas: 'Atlas') -> List[str]:
        return [atlas.name]

    def get_parcellation(self, parcellation=None):
        """Returns a valid parcellation object defined by the atlas.
        If no specification is provided, the default is returned."""

        if parcellation is None:
            parcellation_obj = self.parcellations[0]
            if len(self.parcellations) > 1:
                logger.info(
                    f"No parcellation specified, using default '{str(parcellation_obj)}'."
                )
        else:
            parcellation_obj = self.parcellations[parcellation]
            if parcellation_obj not in self.parcellations:
                raise ValueError(
                    f"Parcellation {str(parcellation_obj)} not supported by atlas {self.name}."
                )
        return parcellation_obj

    def get_space(self, space=None):
        """Returns a valid reference space object defined by the atlas.
        If no specification is provided, the default is returned.

        Parameters:
            space: Space, or string specification of a space
        """
        if space is None:
            space_obj = self.spaces[0]
            if len(self.spaces) > 1:
                logger.info(f"No space specified, using default '{space_obj.name}'.")
        else:
            space_obj = self.spaces[space]

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

    def get_template(self, space: Space = None, variant: str = None):
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
        groupname=None,
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
        for p in self.parcellations:
            if p.is_newest_version or all_versions:
                match = p.find_regions(
                    regionspec,
                    filter_children=filter_children,
                    build_group=build_groups,
                    groupname=groupname,
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
