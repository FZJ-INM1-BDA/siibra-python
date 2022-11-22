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

from ..registry import Preconfigure, ObjectLUT, REGISTRY
from ..commons import logger, MapType, ParcellationIndex
from ..volumes import ParcellationMap

from typing import Set, Union
from difflib import SequenceMatcher
from os import path


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
        self, name, parcellation, collection=None, prev_filename=None, next_filename=None, deprecated=False
    ):
        self.name = name
        self.collection = collection
        self.parcellation = parcellation
        self.next_filename = next_filename
        self.prev_filename = prev_filename
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
        # check predecessors of other
        pred = other.prev
        while pred is not None:
            if pred.version == self:
                return True
            pred = pred.version.prev

        # check successors of self
        succ = self.next
        while succ is not None:
            if succ.version == other:
                return True
            succ = succ.version.next

        return False

    def find_parcellation(self, preconf_fname):
        if preconf_fname is None:
            return None
        if preconf_fname.startswith('./'):
            preconf_fname = path.join(
                path.dirname(self.parcellation._preconfiguration_file),
                preconf_fname[2:]
            )
        matches = [
            p for p in REGISTRY.Parcellation
            if p._preconfiguration_file == preconf_fname
        ]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) == 0:
            raise RuntimeError(f"No parcellations found for preconfiguration file {preconf_fname}")
        else:
            raise RuntimeError(f"Mulitple parcellations found for preconfiguration file {preconf_fname}")

    @property
    def next(self):
        return self.find_parcellation(self.next_filename)

    @property
    def prev(self):
        return self.find_parcellation(self.prev_filename)


@Preconfigure("parcellations")
class Parcellation(
    Region,  # parcellations are also used to represent the root nodes of region hierarchies
    type_id="minds/core/parcellationatlas/v1.0.0",
):

    def __init__(
        self,
        identifier: str,
        name: str,
        regions: tuple[Region] = (),
        shortname: str = "",
        description: str = "",
        version: ParcellationVersion = None,
        modality: str = None,
        publications: list = [],
        ebrains_ids: dict = {},
    ):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the parcellation
        name : str
            Human-readable name of the parcellation
        regions: list or Region
        shortname: str
            Shortform of human-readable name (optional)
        description: str
            Textual description of the parcellation
        version : str or None
            Version specification, optional
        modality  :  str or None
            Specification of the modality used for creating the parcellation
        publications: list
            List of ssociated publications, each a dictionary with "doi" and/or "citation" fields
        ebrains_ids : dict
            Identifiers of EBRAINS entities corresponding to this Parcellation. 
            Key: EBRAINS KG schema, value: EBRAINS KG @id
        """
        Region.__init__(
            self, 
            name=name, 
            children=regions, 
            parent=None,
            shortname=shortname,
            description=description,
            publications=publications,
            ebrains_ids=ebrains_ids
        )
        self.version = version
        self.modality = modality
        self.atlases = set()

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
            spaceobj = REGISTRY.Space[space]
            if not self.supports_space(spaceobj):
                raise ValueError(
                    f'Parcellation "{self.name}" does not provide a map for space "{spaceobj.name}"'
                )

        return ParcellationMap.get_instance(self, spaceobj, maptype)

    def get_colormap(self):
        """Generate a matplotlib colormap from known rgb values of label indices."""
        from matplotlib.colors import ListedColormap
        import numpy as np

        colors = {
            r.index.label: r.attrs["rgb"]
            for r in self
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
            | {space for region in self for space in region.supported_spaces}
        )

    def supports_space(self, space: Space):
        """
        Return true if this parcellation supports the given space, else False.
        """
        return any(s.matches(space) for s in self.supported_spaces)

    @property
    def spaces(self):
        return ObjectLUT(
            matchfunc=Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    @property
    def is_newest_version(self):
        return (self.version is None) or (self.version.next is None)

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

        candidates = self.find(regionspec, filter_children=True, find_topmost=find_topmost)
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
        found_regions = self.find(
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

    @classmethod
    def _from_json(cls, obj):
        """
        Provides an object hook for the json library to construct a Parcellation
        object from a json string.
        """
        assert obj.get("@type", None) == "siibra/parcellation/v0.0.1"

        # construct child region objects
        regions = []
        for regionspec in obj.get("regions", []):  
            try:
                regions.append(Region._from_json(regionspec))
            except Exception as e:
                print(regionspec)
                raise e

        # create the parcellation. This will create a parent region node for the regiontree.
        parcellation = cls(
            identifier=obj["@id"],
            name=obj["name"],
            regions=regions,
            shortname=obj.get("shortName"),
            description=obj.get("description", ""),
            modality=obj.get('modality', ""),
            publications=obj.get("publications", []),
            ebrains_ids=obj.get("ebrains", {}),
        )

        # add version object, if any is specified
        versionspec = obj.get('@version', None)
        if versionspec is not None:
            version = ParcellationVersion(
                name=versionspec.get("name", None),
                parcellation=parcellation,
                collection=versionspec.get("collectionName", None),
                prev_filename=versionspec.get("@prev", None),
                next_filename=versionspec.get("@next", None),
                deprecated=versionspec.get("deprecated", False)
            )
            parcellation.version = version

        return parcellation
 

