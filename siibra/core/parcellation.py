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

from datetime import date
from memoization import cached
from typing import Any, Dict, List, Optional, Union


from .space import Space
from .region import Region, SiibraNode
from .concept import AtlasConcept, RegistrySrc, provide_openminds_registry, main_openminds_registry

from ..volumes import ParcellationMap, VolumeSrc
from ..commons import Registry, logger, MapType, ParcellationIndex
from ..openminds.common import CommonConfig
from ..openminds.SANDS.v3.atlas import brainAtlasVersion, brainAtlas, parcellationEntityVersion

from typing import Union
from memoization import cached
from difflib import SequenceMatcher

# NOTE : such code could be used to automatically resolve
# multiple matching parcellations for a short spec to the newset version:
#               try:
#                    collections = {m.version.collection for m in matches}
#                    if len(collections)==1:
#                        return sorted(matches,key=lambda m:m.version,reverse=True)[0]
#                except Exception as e:
#                    pass


class ParcellationVersion:
    def __init__(
        self, name=None, collection=None, prev_id=None, next_id=None, deprecated=False
    ):
        self.name = name
        self.collection = collection
        self.next_id = next_id
        self.prev_id = prev_id
        self.deprecated = deprecated

    def __eq__(self, other):
        return all([self.name == other.name, self.collection == other.collection])

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __iter__(self):
        yield "name", self.name
        yield "prev", self.prev.id if self.prev is not None else None
        yield "next", self.next.id if self.next is not None else None
        yield "deprecated", self.deprecated

    def __lt__(self, other):
        """< operator, useful for sorting by version"""
        successor = self.next
        while successor is not None:
            if successor.version == other:
                return True
            successor = successor.version.next
        return False

    @property
    def next(self):
        if self.next_id is None:
            return None
        try:
            return Parcellation.REGISTRY[self.next_id]
        except IndexError:
            return None
        except NameError:
            logger.warning("Accessing REGISTRY before its declaration!")

    @property
    def prev(self):
        if self.prev_id is None:
            return None
        try:
            return Parcellation.REGISTRY[self.prev_id]
        except IndexError:
            return None
        except NameError:
            logger.warning("Accessing REGISTRY before its declaration!")

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a
        ParcellationVersion object from a json string.
        """
        if obj is None:
            return None
        return cls(
            obj.get("name", None),
            obj.get("collectionName", None),
            prev_id=obj.get("@prev", None),
            next_id=obj.get("@next", None),
            deprecated=obj.get("deprecated", False),
        )

class BrainAtlasHasTerminology(brainAtlas.HasTerminology):
    Config = CommonConfig

class BrainAtlas(brainAtlas.Model):
    Config = CommonConfig

    @staticmethod
    def parse_legacy_id(legacy_id: str) -> str:
        return f'https://openminds.ebrains.eu/instances/BrainAtlas/{legacy_id}'

    @classmethod
    def parse_legacy(Cls, json_input: Dict[str, Any], has_versions=[]) -> List['BrainAtlas']:
        """
        pass the value of @version to parse_legacy method.
        n.b. BrainAtlas.parse_legacy, unlike its peers, have side effects
        If a brain atlas of the same collectionName already exists in the main registry,
        it will extend the has_version attr
        """
        
        brainatlas_id=BrainAtlas.parse_legacy_id(json_input.get("collectionName"))

        if main_openminds_registry.provides(brainatlas_id):
            
            logger.debug(f'BrainAtlas.parse_legacy of {brainatlas_id}: skipping adding new brain atlas. Main registry already provides.')

            brain_atlas: 'BrainAtlas' = main_openminds_registry[brainatlas_id]
            
            # TODO enable 
            # assert isinstance(brain_atlas, Cls)
            
            existing_version_ids: List[str] = [has_version.get('@id') for has_version in brain_atlas.has_version]
            for has_v in has_versions:
                adding_id: str = has_v.get('@id') 
                if adding_id in existing_version_ids:
                    logger.warning(f'adding has_version with id {adding_id} skipped. Already exist')
                else:
                    brain_atlas.has_version.append(has_v)
            return []

        brainatlas = Cls(
            id=brainatlas_id,
            type="https://openminds.ebrains.eu/sands/BrainAtlas",
            author=[{
                "@id": "https://openminds.ebrains.eu/instances/author/default_author"
            }],
            description=".",
            full_name=json_input.get("collectionName"),
            short_name=json_input.get("collectionName")[:30],
            has_terminology=BrainAtlasHasTerminology(
                short_name=f"{json_input.get('collectionName')} - terminology"[:30],
                has_entity=[]
            ),
            has_version=has_versions,
        )
        return [ brainatlas ]


@provide_openminds_registry(
    bootstrap_folder='parcellations',
    registry_src=RegistrySrc.GITLAB,
)
class Parcellation(
    brainAtlasVersion.Model,
    AtlasConcept,
):

    _atlases = set()
    _root_region = None

    def __init__(
        self,
        dataset_specs=[],
        **data
    ):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the parcellation
        name : str
            Human-readable name of the parcellation
        version : str or None
            a version specification, optional
        modality  :  str or None
            a specification of the modality used for creating the parcellation.
        regiondefs : list of dict
            json specification of regions (siibra-configuration schema)
        dataset_specs : list of dict
            json specification of dataset (siibra-configuration schema)
        maps : list of VolumeSrc (optional)
            List of already instantiated parcellation maps
            (as opposed to the dataset_specs, which still need to be instantiated)
        """
        brainAtlasVersion.Model.__init__(self,**data)
        AtlasConcept.__init__(self, self.id, self.full_name, dataset_specs)

        all_regions: List[Region] = [entity for entity in self.has_terminology_version.has_entity_version if isinstance(entity, Region)]
        root_regions = [ r for r in  all_regions if r.parent is None ]
        
        assert len(root_regions) == 1, f'expecting one and only one root region, but got {len(root_regions)}'
        self._root_region = root_regions[0]

        # self.version = version
        # self.description = ""
        # self.modality = modality
        # self._regiondefs = regiondefs
        # if maps is not None:
        #     if self._datasets_cached is None:
        #         self._datasets_cached = []
        #     self._datasets_cached.extend(maps)

    # TODO backwards compat
    @property
    def version(self) -> str:
        return self.version_identifier

    # TODO backwards compat
    @property
    def name(self) -> str:
        return self.full_name
    
    # TODO backwards compat
    @property
    def publications(self) -> List[str]:
        return self.related_publication or []

    @property
    def space(self) -> Space:
        return main_openminds_registry[self.coordinate_space]

    # TODO deprecate?
    @property
    def regiontree(self) -> Region:
        return self._root_region

    # TODO deprecate
    # instead have a root region object, and regiontree should return the root region object
    @property
    def name(self) -> str:
        return self.full_name

    @property
    def volumes(self):
        return self.has_terminology_version.defined_in

    def get_map(self, space=None, maptype: Union[str, MapType] = MapType.LABELLED):
        """
        Get the volumetric maps for the parcellation in the requested
        template space. This might in general include multiple
        3D volumes. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere. Per default, multiple maps are concatenated into a 4D
        array, but you can choose to retrieve a dict of 3D volumes instead using `return_dict=True`.

        Parameters
        ----------
        space : Space or str
            template space specification
        maptype : MapType (default: MapType.LABELLED)
            Type of map requested (e.g., continous or labelled, see commons.MapType)
            Use MapType.CONTINUOUS to request probability maps.

        Yields
        ------
        A ParcellationMap representing the volumetric map.
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]
        if space is None:
            spaces = {v.space for v in self.volumes}
            if len(spaces) == 0:
                raise RuntimeError(f'Parcellation "{str(self)}" provides no maps.')
            elif len(spaces) == 1:
                spaceobj = next(iter(spaces))
            else:
                raise ValueError(
                    f'Parcellation "{str(self)}" provides maps in multiple spaces, but no space was specified ({",".join(s.name for s in spaces)})'
                )
        else:
            spaceobj = Space.REGISTRY[space]
            if not self.supports_space(spaceobj):
                raise ValueError(
                    f'Parcellation "{self.name}" does not provide a map for space "{spaceobj.name}"'
                )

        return ParcellationMap.get_instance(self, spaceobj, maptype)

    def __hash__(self):
        return hash(self.full_name + self.coordinate_space.get('@id'))

    @property
    def labels(self):
        return self.regiontree.labels

    @property
    def names(self):
        return self.regiontree.names

    def get_colormap(self):
        """Generate a matplotlib colormap from known rgb values of label indices."""
        from matplotlib.colors import ListedColormap
        import numpy as np

        colors = {
            r.index.label: r.attrs["rgb"]
            for r in self.regiontree
            if "rgb" in r.attrs and r.index.label
        }
        pallette = np.array(
            [
                colors[i] + [1] if i in colors else [0, 0, 0, 0]
                for i in range(max(colors.keys()) + 1)
            ]
        ) / [255, 255, 255, 1]
        return ListedColormap(pallette)

    @property
    def supported_spaces(self):
        """Overwrite the method of AtlasConcept.
        For parcellations, a space is also considered as supported if one of their regions is mapped in the space.
        """
        return {
            space for region in self.regiontree for space in region.supported_spaces
        }

    def supports_space(self, space: Space) -> bool:
        """
        Return true if this parcellation supports the given space, else False.
        """
        return space == main_openminds_registry[self.coordinate_space]

    @property
    def spaces(self):
        return Registry(
            elements=[ main_openminds_registry[self.coordinate_space] ],
            get_aliases=Space.get_aliases,
        )

    @property
    def is_newest_version(self) -> bool:
        newer = [
            parc
            for parc in self.REGISTRY
            if parc.is_new_version_of and parc.is_new_version_of.get("@id") == self.id
        ]
        return len(newer) == 0

    @property
    def publications(self):
        return self._publications + super().publications

    def decode_region(self, regionspec: Union[str, int, ParcellationIndex, Region]):
        """
        Given a unique specification, return the corresponding region.
        The spec could be a label index, a (possibly incomplete) name, or a
        region object.
        This method is meant to definitely determine a valid region. Therefore,
        if no match is found, it raises a ValueError. If it finds multiple
        matches, it tries to return only the common parent node. If there are
        multiple remaining parent nodes, which is rare, a custom group region is constructed.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex,
            - a region object
            - a full ParcellationIndex

        Return
        ------
        Region object
        """
        if isinstance(regionspec, Region) and (regionspec.parcellation == self):
            return

        if isinstance(regionspec, str) and ("Group:" in regionspec):
            # seems to be a group region name - build the group region by recursive decoding.
            subspecs = regionspec.replace("Group:", "").split(",")
            return Region._build_grouptree(
                [self.decode_region(s) for s in subspecs], parcellation=self
            )

        candidates = self.regiontree.find(regionspec, filter_children=True)
        if not candidates:
            raise ValueError(
                "Regionspec {} could not be decoded under '{}'".format(
                    regionspec, self.full_name
                )
            )
        elif len(candidates) == 1:
            return candidates[0]
        else:
            if isinstance(regionspec, str):
                scores = [
                    SequenceMatcher(None, regionspec, region.name).ratio()
                    for region in candidates
                ]
                bestmatch = scores.index(max(scores))
                logger.info(
                    f"Decoding of spec {regionspec} resulted in multiple matches: "
                    f"{','.join(r.name for r in candidates)}. The closest match was chosen: {candidates[bestmatch].name}"
                )
                return candidates[bestmatch]
            raise RuntimeError(
                f"Decoding of spec {regionspec} resulted in multiple matches: {','.join(r.name for r in candidates)}."
            )

    @cached
    def find_regions(
        self, regionspec, filter_children=False, build_group=False, groupname=None
    ):
        """
        Find regions with the given specification in this parcellation.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex
            - a region object
        filter_children : Boolean
            If true, children of matched parents will not be returned
        build_group : Boolean, default: False
            If true, the result will be a single region object. To do so,
            a group region of matched elements will be created if needed.
        groupname : str (optional)
            Name of the resulting group region, if build_group is True

        Yield
        -----
        list of matching regions
        """
        found_regions = self.regiontree.find(
            regionspec,
            filter_children=filter_children,
            build_group=build_group,
            groupname=groupname,
        )

        # Perform ranking of return result, if the spec provided is a string. Otherwise, return the unsorted found_regions
        # reverse is set to True, since SequenceMatcher().ratio(), higher == better
        return (
            sorted(
                found_regions,
                reverse=True,
                key=lambda region: SequenceMatcher(
                    None, str(region), regionspec
                ).ratio(),
            )
            if type(regionspec) == str
            else found_regions
        )

    def __str__(self):
        return self.full_name

    def __repr__(self):
        return f'{self.full_name} - {self.space.id}'

    def __eq__(self, other: Union['Parcellation', str]) -> bool:
        """
        Compare this parcellation with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other, Parcellation):
            return self.id == other.id
        elif isinstance(other, str):
            return any([self.short_name == other, self.full_name == other, self.id == other])
        else:
            raise ValueError(
                "Cannot compare object of type {} to Parcellation".format(type(other))
            )

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this parcellation
        """
        return self.regiontree.__iter__()

    def __getitem__(self, regionspec: Union[str, int]):
        """
        Retrieve a region object from the parcellation by labelindex or partial name.
        """
        return self.decode_region(regionspec)

    def __lt__(self, other: 'Parcellation') -> bool:
        """
        We sort parcellations by their version
        """
        
        def get_prev_version(parc: Parcellation) -> Optional[Parcellation]:
            if not parc.is_new_version_of:
                return None
            try:
                return main_openminds_registry[parc.is_new_version_of]
            except KeyError as e:
                logger.debug(f"attempt to get prev version {str(self)} was unsuccessful.")
                return None
        
        prev_version = get_prev_version(self)
        while prev_version:
            if prev_version == other:
                return False
            prev_version = get_prev_version(prev_version)
        return True

    def _extend(self, other):
        """Extend a parcellation by additional regions
        from a parcellation extension.

        Args:
            other (Parcellation): Extension parcellation
        """
        assert other.extends == self.id
        assert isinstance(other, self.__class__)
        for region in other.regiontree:

            try:
                matched_parent = self.decode_region(region.parent.name)
            except (ValueError, AttributeError):
                continue

            conflicts = matched_parent.find(region.name, filter_children=True)
            if len(conflicts) == 1:
                merge_with = conflicts[0]
                for d in region.datasets:
                    if len(merge_with.datasets) > 0 and (d in merge_with.datasets):
                        logger.error(
                            f"Dataset '{str(d)}' already exists in '{merge_with.name}', and will not be extended."
                        )
                    logger.debug(
                        f"Extending existing region {merge_with.name} with dataset {str(d)}"
                    )
                    merge_with._datasets_cached.append(d)

            elif len(conflicts) == 0:
                logger.debug(
                    f"Extending '{matched_parent}' with '{region.name}' from '{other.name}'."
                )
                new_child = Region.copy(region)
                new_child.parcellation = matched_parent.parcellation
                new_child.extended_from = other
                new_child.parent = matched_parent

            else:
                raise RuntimeError(
                    f"Cannot extend '{matched_parent}' with '{region.name}' "
                    "due to multiple conflicting children: "
                    f"{', '.join(c.name for c in conflicts)}"
                )

    @staticmethod
    def get_aliases(key: str, parc: 'Parcellation') -> List[str]:
        return [
            f'{parc.full_name}_{parc.space.full_name}'
        ]

    @staticmethod
    def parse_legacy_id(parc_id: str, space_id: str) -> str:
        return f'https://openminds.ebrains.eu/instances/BrainAtlasVersion/{parc_id}-{space_id}'

    @classmethod
    def parse_legacy(Cls, json_input: Dict[str, Any]) -> List['Parcellation']:
        """
        The parse legacy class method for Parcellation has a lot of side effects.
        
        - the closest model in openminds sands v3 is brainAtlasVersion, which is separated on a per space basis.
        This means for multi-spaced parcellations (such as julich brain), the parse_legacy method on a single parcellation
        will return a list of brainAtlasVersion's

        - the region attribute has been renamed to .has_terminology_version.has_entity_version . Region entities (parcellationEntityVersion 's)
        are generated

        - 
        """
        assert json_input.get('@id') is not None
        if json_input.get('@extends'):
            return []

        parc_id = json_input.get('@id')
        parc_type = 'https://openminds.ebrains.eu/sands/BrainAtlasVersion'
        accessibility = {
            '@id': 'https://openminds.ebrains.eu/instances/productAccessibility/freeAccess'
        }
        author = []
        coordinate_spaces: List[str] = [dataset.get('space_id') for dataset in json_input.get('datasets') if dataset.get('space_id')]
        
        # remove duplicates
        coordinate_spaces = list(set(coordinate_spaces))
        copyright = None
        custodian = []
        description = json_input.get('description')
        digital_identifier = None
        full_documentation = {
            '@id': 'add_doi_here'
        }
        full_name = json_input.get('name')
        funding = []

        append_openminds_entities = []
        def get_mpm(spc_id: str) -> List[VolumeSrc]:
            mpms = [ file
                for dataset in json_input.get('datasets')
                if dataset.get('@type') == 'fzj/tmp/volume_type/v0.0.1' and dataset.get('space_id') == spc_id
                for file in VolumeSrc.parse_legacy(dataset) ]

            # append mpms in main openminds registry
            append_openminds_entities.extend(mpms)

            return mpms

        def get_has_terminology_version(spc_id: str):

            # TODO discuss: register regions at main_openminds_registry?

            # if regions do not have a single root, create a single root here
            parent = None
            if len(json_input.get('regions')) != 1:
                id = f"{parc_id}-root"
                parent = Region(
                    id=id,
                    name='Whole brain',
                    type='https://openminds.ebrains.eu/sands/ParcellationEntityVersion',
                    has_annotation=None,
                    lookup_label=None,
                    parcellation_id=parc_id,
                    version_identifier='12',
                    parent=None
                )

            parsed_legacy_regions = [entity
                for r in json_input.get('regions')
                for entity in Region.parse_legacy(
                    r,
                    parcellation_id=Parcellation.parse_legacy_id(parc_id, spc_id),
                    parent=parent)
            ]

            regions_entities = [ entity
                for entity in parsed_legacy_regions
                if isinstance(entity, Region)]
            
            # parsed non region entities
            # e.g. PMs
            other_parsed_entities = [ entity
                for entity in parsed_legacy_regions
                if not isinstance(entity, Region)]
            append_openminds_entities.extend(other_parsed_entities)

            if parent is not None:
                regions_entities.append(parent)

            mpm_entities = get_mpm(spc_id)
            
            return {
                "has_entity_version": regions_entities,
                "short_name": "stuff",
                "version_identifier": "stuff2",
                "version_innovation": "stuff3",
                "defined_in": mpm_entities
            }
        
        homepage = None

        license = {
            '@id': 'https://openminds.ebrains.eu/instances/licenses/ccByNc4.0	'
        }

        release_date = date(1970, 1, 1)
        short_name = json_input.get('shortName')
        version_innovation = 'place holder'

        version_json = json_input.get("@version", {
            "collectionName": json_input.get("shortName") or json_input.get("name"),
            "name": "latest"
        })

        def get_is_new_version_of(spc_id: str) -> Dict[str, str]:
            if version_json.get("@prev") is None:
                return None
            else:
                return {
                    "@id": Parcellation.parse_legacy_id(version_json.get("@prev"), spc_id)
                }

        version_identifier = version_json.get('name') or 'latest'

        brainatlases = BrainAtlas.parse_legacy(version_json, has_versions=[{
            "@id": Parcellation.parse_legacy_id(parc_id, spc)
        } for spc in coordinate_spaces])
        append_openminds_entities.extend(brainatlases)

        if len(short_name) > 30:
            short_name = short_name[:30]

        # TODO add ng volumes
        # TODO add nifti volume as file
        # TODO add brainAtlas (parent instance) for each collection
        return [
            *[
                Cls(
                    id=Parcellation.parse_legacy_id(parc_id, spc),
                    type=parc_type,
                    accessibility=accessibility,
                    coordinate_space={
                        "@id": Space.parse_legacy_id(spc)
                    },
                    copyright=None,
                    full_documentation=full_documentation,
                    full_name=full_name,
                    is_new_version_of=get_is_new_version_of(spc),
                    release_date=release_date,
                    has_terminology_version=get_has_terminology_version(spc),
                    license=license,
                    short_name=short_name,
                    version_identifier=version_identifier,
                    version_innovation=version_innovation,
                ) for spc in coordinate_spaces
            ],
            *append_openminds_entities,
        ]

    Config = CommonConfig

