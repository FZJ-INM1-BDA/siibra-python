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

from .space import Space
from .region import Region
from .concept import AtlasConcept, provide_registry
from .serializable_concept import JSONSerializable
from .datasets import DatasetJsonModel, OriginDescription, EbrainsDataset, EbrainsKgV3DatasetVersion, EbrainsKgV3Dataset

from ..commons import logger, MapType, ParcellationIndex, Registry
from ..volumes import ParcellationMap
from ..openminds.SANDS.v3.atlas.brainAtlasVersion import (
    Model as BrainAtlasVersionModel,
    HasTerminologyVersion,
)
from ..openminds.base import ConfigBaseModel, SiibraAtIdModel

from datetime import date
from typing import List, Optional, Set, Union
from memoization import cached
from difflib import SequenceMatcher
from pydantic import Field


SIIBRA_PARCELLATION_MODEL_TYPE="minds/core/parcellationatlas/v1.0.0"
BRAIN_ATLAS_VERSION_TYPE="https://openminds.ebrains.eu/sands/BrainAtlasVersion"

class AtlasType:
    DETERMINISTIC_ATLAS="https://openminds.ebrains.eu/instances/atlasType/deterministicAtlas"
    PARCELLATION_SCHEME="https://openminds.ebrains.eu/instances/atlasType/parcellationScheme"
    PROBABILISTIC_ATLAS="https://openminds.ebrains.eu/instances/atlasType/probabilisticAtlas"


class SiibraParcellationVersionModel(ConfigBaseModel):
    name: str
    deprecated: Optional[bool]
    prev: Optional[SiibraAtIdModel]
    next: Optional[SiibraAtIdModel]


class SiibraParcellationModel(ConfigBaseModel):
    id: str = Field(..., alias="@id")
    type: str = Field(SIIBRA_PARCELLATION_MODEL_TYPE, const=True, alias="@type")
    name: str
    modality: Optional[str]
    datasets: List[DatasetJsonModel]
    brain_atlas_versions: List[BrainAtlasVersionModel] = Field(..., alias="brainAtlasVersions")
    version: Optional[SiibraParcellationVersionModel]
    

# NOTE : such code could be used to automatically resolve
# multiple matching parcellations for a short spec to the newset version:
#               try:
#                    collections = {m.version.collection for m in matches}
#                    if len(collections)==1:
#                        return sorted(matches,key=lambda m:m.version,reverse=True)[0]
#                except Exception as e:
#                    pass


class ParcellationVersion(JSONSerializable):
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
    
    @classmethod
    def get_model_type(Cls):
        raise AttributeError("ParcellationVersion.@type cannot be determined")

    @property
    def model_id(self):
        return super().model_id

    def to_model(self, **kwargs) -> SiibraParcellationVersionModel:
        assert self.prev is None or isinstance(self.prev, Parcellation), f"parcellationVersion to_model failed. expected .prev, if defined, to be instance of Parcellation, but is {self.prev.__class__} instead"
        assert self.next is None or isinstance(self.next, Parcellation), f"parcellationVersion to_model failed. expected .next, if defined, to be instance of Parcellation, but is {self.next.__class__} instead"
        return SiibraParcellationVersionModel(
            name=self.name,
            deprecated=self.deprecated,
            prev=SiibraAtIdModel(
                id=self.prev.model_id
            ) if self.prev is not None else None,
            next=SiibraAtIdModel(
                id=self.next.model_id
            ) if self.next is not None else None,
        )


@provide_registry
class Parcellation(
    AtlasConcept,
    JSONSerializable,
    bootstrap_folder="parcellations",
    type_id="minds/core/parcellationatlas/v1.0.0",
):

    _regiontree_cached = None

    def __init__(
        self,
        identifier: str,
        name: str,
        version=None,
        modality=None,
        regiondefs=[],
        dataset_specs=[],
        maps=None,
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
        AtlasConcept.__init__(self, identifier, name, dataset_specs)
        self.version = version
        self._description = ""
        self.modality = modality
        self._regiondefs = regiondefs
        if maps is not None:
            if self._datasets_cached is None:
                self._datasets_cached = []
            self._datasets_cached.extend(maps)
        self.atlases = set()
    
    @property
    def description(self):

        metadata_datasets = [info
            for info in self.infos
            if isinstance(info, EbrainsDataset)
            or isinstance(info, EbrainsKgV3DatasetVersion)
            or isinstance(info, EbrainsKgV3Dataset)
            or isinstance(info, OriginDescription)]

        if len(metadata_datasets) == 0:
            return self._description
        
        if len(metadata_datasets) > 1:
            logger.debug(f"Parcellation.description multiple metadata_datasets found. Using the first one.")
        
        return metadata_datasets[0].description

    @property
    def regiontree(self):
        if self._regiontree_cached is None:
            self._regiontree_cached = Region(
                self.name, self, ParcellationIndex(None, None)
            )
            try:
                self._regiontree_cached.children = tuple(
                    Region._from_json(regiondef, self) for regiondef in self._regiondefs
                )
            except Exception as e:
                logger.error(f"Could not generate child regions for {self.name}")
                raise (e)
        return self._regiontree_cached

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
    def supported_spaces(self) -> Set[Space]:
        """Overwrite the method of AtlasConcept.
        For parcellations, a space is also considered as supported if one of their regions is mapped in the space.
        """
        return list(
            set(super().supported_spaces) 
            | {space for region in self.regiontree for space in region.supported_spaces}
        )

    def supports_space(self, space: Space):
        """
        Return true if this parcellation supports the given space, else False.
        """
        return any(s.matches(space) for s in self.supported_spaces)

    @property
    def spaces(self):
        return Registry(
            matchfunc=Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    @property
    def is_newest_version(self):
        return (self.version is None) or (self.version.next is None)

    @property
    def publications(self):
        return self._publications + super().publications

    def decode_region(self, regionspec: Union[str, int, ParcellationIndex, Region], find_topmost=True):
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
        find_topmost : Bool, default: True
            If True, will automatically return the parent of a decoded region the decoded region is its only child.

        Return
        ------
        Region object
        """
        if isinstance(regionspec, Region) and (regionspec.parcellation == self):
            return regionspec

        if isinstance(regionspec, str) and regionspec.startswith("Group:"):
            # seems to be a group region name - build the group region by recursive decoding.
            logger.info(f"Decoding group region: {regionspec}")
            regions = [
                self.decode_region(s) for s in regionspec.replace("Group:", "").split(",")
            ]
            # refuse to group regions with any of their existing children
            cleaned_regions = [
                r for r in regions 
                if not any(r in r2.descendants for r2 in regions)
            ]
            if len(cleaned_regions) == 1:
                logger.debug(f"Group reduced to a single parent: {cleaned_regions[0].name}")
                return cleaned_regions[0]
            else:
                return Region._build_grouptree(cleaned_regions, parcellation=self)

        candidates = self.regiontree.find(regionspec, filter_children=True, find_topmost=find_topmost)
        if not candidates:
            raise ValueError(
                "Regionspec {} could not be decoded under '{}'".format(
                    regionspec, self.name
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
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        """
        Compare this parcellation with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other, Parcellation):
            return self.id == other.id
        elif isinstance(other, str):
            return any([self.name == other, self.key == other, self.id == other])
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

    def __lt__(self, other):
        """
        We sort parcellations by their version
        """
        if (self.version is None) or (other.version is None):
            raise RuntimeError(
                f"Attempted to sort non-versioned instances of {self.__class__.__name__}."
            )
        return self.version.__lt__(other.version)

    def _extend(self, other):
        """Extend a parcellation by additional regions
        from a parcellation extension.

        Args:
            other (Parcellation): Extension parcellation
        """
        assert other.extends == self.id
        assert isinstance(other, self.__class__)
        for region in other.regiontree:

            if region.parent is None:
                continue
    
            try:
                matched_parent = self.decode_region(region.parent.name, find_topmost=False)
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
                    f"Extending '{matched_parent}' in '{self.name}' with '{region.name}' from '{other.name}'."
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

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a Parcellation
        object from a json string.
        """
        assert obj.get("@type", None) == "minds/core/parcellationatlas/v1.0.0"

        # create the parcellation, it will create a parent region node for the regiontree.
        result = cls(
            obj["@id"],
            obj["shortName"],
            regiondefs=obj["regions"],
            dataset_specs=obj.get("datasets", []),
        )

        if "@version" in obj:
            result.version = ParcellationVersion._from_json(obj["@version"])

        if "modality" in obj:
            result.modality = obj["modality"]

        if "description" in obj:
            result._description = obj["description"]

        if "publications" in obj:
            result._publications = obj["publications"]

        if "@extends" in obj:
            result.extends = obj["@extends"]

        return result

    def get_brain_atlas_version_id(self, space: Space) -> str:
        return f"{self.id}/{space.model_id}"

    def get_brain_atlas_version_name(self, space: Space) -> str:
        return f"{self.name} in {space.to_model().full_name}"

    @classmethod
    def get_model_type(Cls):
        return SIIBRA_PARCELLATION_MODEL_TYPE

    @property
    def model_id(self):
        return self.id

    def to_model(self, **kwargs) -> SiibraParcellationModel:

        dois = [url.get("doi") if url.get("doi").startswith("http") else f"https://doi.org/{url.get('doi')}"
            for info in self.infos
            if hasattr(info, 'urls')
            for url in info.urls
            if url.get("doi")]
        ebrains_doi = dois[0] if len(dois) > 0 else None
        return SiibraParcellationModel(
            id=self.model_id,
            type=SIIBRA_PARCELLATION_MODEL_TYPE,
            name=self.name,
            modality=self.modality,
            datasets=[ds.to_model() for ds in self.datasets if isinstance(ds, OriginDescription) or isinstance(ds, EbrainsDataset)],
            brain_atlas_versions=[BrainAtlasVersionModel(
                id=self.get_brain_atlas_version_id(spc),
                type=BRAIN_ATLAS_VERSION_TYPE,
                atlas_type={
                    # TODO fix
                    "@id": AtlasType.PROBABILISTIC_ATLAS
                },
                accessibility={
                    # TODO fix
                    "@id": ""
                },
                coordinate_space={
                    "@id": spc.model_id
                },
                description=self.description[:2000],
                full_documentation={
                    # TODO fix
                    "@id": ""
                },
                full_name=self.get_brain_atlas_version_name(spc),
                has_terminology_version=HasTerminologyVersion(
                    has_entity_version=[{
                        "@id": r.model_id
                    } for r in self]
                ),
                license={
                    # TODO fix
                    "@id": ""
                },
                release_date=date(1970,1,1),
                short_name=self.name[:30],
                version_identifier=f"{self.version} in {spc.to_model().full_name}",
                version_innovation=self.description,
                digital_identifier={
                    "@id": ebrains_doi,
                    "@type": "https://openminds.ebrains.eu/core/DOI"
                } if ebrains_doi is not None else None
            ) for spc in self.supported_spaces],
            version=self.version.to_model(**kwargs) if self.version is not None else None
        )
