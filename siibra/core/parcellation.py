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

from ..commons import logger, MapType, MapIndex, InstanceTable

from typing import Set, Union, List


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
        self, name, parcellation, collection=None, prev_id=None, next_id=None, deprecated=False
    ):
        self.name = name
        self.collection = collection
        self.parcellation = parcellation
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
        yield "prev", self.prev_id
        yield "next", self.next_id
        yield "deprecated", self.deprecated

    def __lt__(self, other):
        """
        < operator, useful for sorting by version
        FIXME: this is only by name, not recursing into parcellations, to avoid importing the registry here.
        """
        return self.name < other.name


class Parcellation(Region, configuration_folder="parcellations"):

    _CACHED_REGION_SEARCHES = {}

    def __init__(
        self,
        identifier: str,
        name: str,
        regions: Union[List[Region], Region] = (),
        shortname: str = "",
        description: str = "",
        version: ParcellationVersion = None,
        modality: str = None,
        publications: list = [],
        datasets: list = [],
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
        datasets : list
            datasets associated with this region
        """
        Region.__init__(
            self,
            name=name,
            children=regions,
            parent=None,
            shortname=shortname,
            description=description,
            publications=publications,
            datasets=datasets,
            modality=modality
        )
        self._id = identifier
        self.version = version

    @property
    def id(self):
        return self._id

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
        from ..volumes import Map
        if not isinstance(maptype, MapType):
            maptype = MapType[maptype.upper()]
        candidates = [
            m for m in Map.registry()
            if m.space.matches(space)
            and m.maptype == maptype
            and m.parcellation.matches(self)
        ]
        if len(candidates) == 0:
            logger.error(f"No {maptype} map in {space} available for {str(self)}")
            return None
        if len(candidates) > 1:
            logger.warn(f"Multiple {maptype} maps in {space} available for {str(self)}, choosing the first.")
        return candidates[0]

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

    @classmethod
    def find_regions(cls, region_spec: str, parents_only=True):
        MEM = cls._CACHED_REGION_SEARCHES
        if region_spec not in MEM:
            MEM[region_spec] = [
                r
                for p in cls.registry()
                for r in p.find(regionspec=region_spec)
            ]
        if parents_only:
            return [
                r for r in MEM[region_spec]
                if not any(_ in r.children for _ in MEM[region_spec])
            ]
        else:
            return MEM[region_spec]

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
        return InstanceTable(
            matchfunc=Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    @property
    def is_newest_version(self):
        return (self.version is None) or (self.version.next_id is None)

    def get_region(self, regionspec: Union[str, int, MapIndex, Region], find_topmost=True):
        """
        Given a unique specification, return the corresponding region.
        The spec could be a label index, a (possibly incomplete) name, or a
        region object.
        This method is meant to definitely determine a valid region. Therefore,
        if no match is found, it raises a ValueError. If it finds multiple
        matches, it tries to return only the common parent node.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex,
            - a region object
            - a full MapIndex
        find_topmost : Bool, default: True
            If True, will automatically return the parent of a decoded region the decoded region is its only child.

        Return
        ------
        Region object
        """
        if isinstance(regionspec, Region) and (regionspec.parcellation == self):
            return regionspec

        candidates = self.find(regionspec, filter_children=True, find_topmost=find_topmost)
        if len(candidates) > 1 and isinstance(regionspec, str):
            # if we have an exact match of words in one region, discard other candidates.
            querywords = {w.lower() for w in regionspec.split()}
            for c in candidates:
                targetwords = {w.lower() for w in c.name.split()}
                if len(querywords & targetwords) == len(targetwords):
                    logger.debug(
                        f"Candidates {', '.join(_.name for _ in candidates if _ != c)} "
                        f"will be ingored, because candidate {c.name} is a full match to {regionspec}."
                    )
                    candidates = [c]

        if not candidates:
            raise ValueError(
                "Regionspec {} could not be decoded under '{}'".format(
                    regionspec, self.name
                )
            )
        elif len(candidates) == 1:
            return candidates[0]
        else:
            raise RuntimeError(
                f"Spec {regionspec} resulted in multiple matches: {', '.join(r.name for r in candidates)}."
            )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __getitem__(self, regionspec: Union[str, int]):
        """
        Retrieve a region object from the parcellation by labelindex or partial name.
        """
        return self.get_region(regionspec)

    def __lt__(self, other):
        """
        We sort parcellations by their version
        """
        if (self.version is None) or (other.version is None):
            raise RuntimeError(
                f"Attempted to sort non-versioned instances of {self.__class__.__name__}."
            )
        return self.version.__lt__(other.version)
