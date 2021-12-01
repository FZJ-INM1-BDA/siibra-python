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

from ..commons import logger, MapType, ParcellationIndex, Registry
from ..volumes import ParcellationMap

import difflib
from typing import Union
from memoization import cached

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


@provide_registry
class Parcellation(
    AtlasConcept,
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
        self.description = ""
        self.modality = modality
        self._regiondefs = regiondefs
        if maps is not None:
            if self._datasets_cached is None:
                self._datasets_cached = []
            self._datasets_cached.extend(maps)
        self.atlases = set()

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

    def decode_region(
        self, regionspec: Union[str, int, ParcellationIndex, Region], build_group=True
    ):
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
        candidates = self.regiontree.find(regionspec, filter_children=True)
        if not candidates:
            raise ValueError(
                "Regionspec {} could not be decoded under '{}'".format(
                    regionspec, self.name
                )
            )
        elif len(candidates) == 1:
            return candidates[0]
        else:
            if build_group:
                logger.debug(
                    f"The specification '{regionspec}' resulted more than one region. A group region is returned."
                )
                return Region._build_grouptree(candidates, self)
            else:
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
            groupname=groupname
        )

        # Perform ranking of return result, if the spec provided is a string. Otherwise, return the unsorted found_regions
        # reverse is set to True, since SequenceMatcher().ratio(), higher == better
        return sorted(found_regions,reverse=True, key=lambda region: difflib.SequenceMatcher(None, str(region), regionspec).ratio()) if type(regionspec) == str else found_regions

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
            result.description = obj["description"]

        if "publications" in obj:
            result._publications = obj["publications"]

        if "@extends" in obj:
            result.extends = obj["@extends"]

        return result
