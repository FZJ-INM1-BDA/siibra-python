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
"""Hierarchal brain regions and metadata."""

from functools import lru_cache
import re
from typing import Union, List, TYPE_CHECKING
try:
    from typing import Literal
except ImportError:
    # support python 3.7
    from typing_extensions import Literal

from . import region
from ..commons import logger, MapType, Species
from ..volumes import parcellationmap
from ..exceptions import MapNotFound


if TYPE_CHECKING:
    from .space import Space

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

    def __lt__(self, other: 'ParcellationVersion'):
        """
        < operator, useful for sorting by version
        FIXME: this is only by name, not recursing into parcellations, to avoid importing the registry here.
        """
        return self.name < other.name


class Parcellation(region.Region, configuration_folder="parcellations"):

    def __init__(
        self,
        identifier: str,
        name: str,
        species: Union[Species, str],
        regions: Union[List[region.Region], region.Region] = (),
        shortname: str = "",
        description: str = "",
        version: ParcellationVersion = None,
        modality: str = None,
        publications: list = [],
        datasets: list = [],
        prerelease: bool = False,
    ):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the parcellation
        name : str
            Human-readable name of the parcellation
        species: str or Species
            Specification of the species
        regions: list or Region
        shortname: str
            Short form of human-readable name (optional)
        description: str
            Textual description of the parcellation
        version : str or None
            Version specification, optional
        modality  :  str or None
            Specification of the modality used for creating the parcellation
        publications: list
            List of associated publications, each a dictionary with "doi"
            and/or "citation" fields
        datasets : list
            datasets associated with this region
        """
        region.Region.__init__(
            self,
            name=name,
            children=regions,
            parent=None,
            shortname=shortname,
            description=description,
            publications=publications,
            datasets=datasets,
            modality=modality,
            prerelease=prerelease,
        )
        self._species_cached = Species.decode(species)
        self._id = identifier
        self.version = version

    @property
    def id(self):
        return self._id

    def matches(self, spec):
        if spec not in self._CACHED_MATCHES:
            if isinstance(spec, str):
                if all(
                    w in self.shortname.lower()
                    for w in re.split(r'\s+', spec.lower())
                ):
                    self._CACHED_MATCHES[spec] = True
        return super().matches(spec)

    def get_map(
        self,
        space: Union[str, "Space"],
        maptype: Union[Literal['labelled', 'statistical'], MapType] = MapType.LABELLED,
        spec: str = ""
    ):
        """
        Get the maps for the parcellation in the requested template space.

        This might in general include multiple 3D volumes. For example,
        the Julich-Brain atlas provides two separate maps, one per hemisphere.
        Per default, multiple maps are concatenated into a 4D array, but you
        can choose to retrieve a dict of 3D volumes instead using
        `return_dict=True`.

        Parameters
        ----------
        space: Space or str
            reference space specification such as name, id, or a `Space` instance.
        maptype: MapType or str
            Type of map requested (e.g., statistical or labelled).
            Use MapType.STATISTICAL to request probability maps.
            Defaults to MapType.LABELLED.
        spec: str, optional
            In case of multiple matching maps for the given parcellation, space
            and type, use this field to specify keywords matching the desired
            parcellation map name. Otherwise, siibra will default to the first
            in the list of matches (and inform with a log message)
        Returns
        -------
        parcellationmap.Map or SparseMap
            A ParcellationMap representing the volumetric map or
            a SparseMap representing the list of statistical maps.
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]
        assert isinstance(maptype, MapType), "Possible values of `maptype` are `MapType`s, 'labelled', 'statistical'."

        candidates = [
            m for m in parcellationmap.Map.registry()
            if m.space.matches(space)
            and m.maptype == maptype
            and m.parcellation.matches(self)
        ]
        if len(candidates) == 0:
            raise MapNotFound(f"No '{maptype}' map in '{space}' available for {str(self)}")
        if len(candidates) > 1:
            spec_candidates = [
                c for c in candidates if all(w.lower() in c.id.lower() for w in spec.split())
            ]
            if len(spec_candidates) == 0:
                raise MapNotFound(f"'{spec}' does not match any options from {[c.name for c in candidates]}.")
            if len(spec_candidates) > 1:
                logger.warning(
                    f"Multiple maps are available in this specification of space, parcellation, and map type.\n"
                    f"Choosing the first map from {[c.name for c in spec_candidates]}."
                )
            return spec_candidates[0]
        return candidates[0]

    @property
    def is_newest_version(self):
        return (self.version is None) or (self.version.next_id is None)

    def _split_group_spec(self, spec: str):
        """
        Split a group region specification as produced in older siibra versions
        into the subregion specs.
        This is used when decoding datasets that still include region name
        specifications from old siibra versions, such as some connectivity matrices.
        """
        spec = re.sub(r'Group: *', '', spec)
        for substr in re.findall(r'\(.*?\)', spec):
            # temporarily replace commas inside brackets with a placeholder
            # because these are not region spec delimiters
            spec = spec.replace(substr, re.sub(r', *', '##', substr))
        # process the comma separated substrings
        candidates = list({
            self.get_region(re.sub(r'##', ', ', s))
            for s in re.split(r', *', spec)
        })
        if len(candidates) > 0:
            return candidates
        else:
            return [spec]

    def get_region(
        self,
        regionspec: Union[str, region.Region],
        find_topmost: bool = False,
        allow_tuple: bool = False
    ):
        """
        Given a unique specification, return the corresponding region.

        The spec could be a (possibly incomplete) name, or a region object.
        This method is meant to definitely determine a valid region. Therefore,
        if no match is found, it raises a ValueError. If multiple matches are
        found, the method tries to return only the common parent node. If there
        is no common parent, an exception is raised, except when
        allow_tuple=True - then a tuple of matched regions is returned.

        Parameters
        ----------
        regionspec: str, Region
            - a string with a possibly inexact name (matched both against the name and the identifier key)
            - a Region object
        find_topmost: bool, default: False
            If True, will automatically return the parent of a decoded region
            the decoded region is its only child.
        allow_tuple: bool, default: False
            If multiple candidates without a common parent are found,
            return a tuple of matches instead of raising an exception.

        Returns
        -------
        Region
            A region object defined in the parcellation.

            Note
            ----
            If the spec exactly matched with more than one region, the first
            will be returned.

        Raises
        ------
        RuntimeError
            If the spec matches multiple regions
        ValueError
            If the spec cannot be matched against any region.
        """
        assert isinstance(regionspec, (str, region.Region)), f"get_region takes str or Region but you provided {type(regionspec)}"
        if isinstance(regionspec, region.Region) and (regionspec.parcellation == self):
            return regionspec

        # if there exist an exact match of region spec to region name, return
        if isinstance(regionspec, str):
            exact_match = [
                region
                for region in self
                if region.name == regionspec or region.key == regionspec
            ]
            if len(exact_match) == 1:
                return exact_match[0]
            if len(exact_match) > 1:
                logger.info(f"Found multiple region with exact match to {regionspec}. Returning the first one.")
                return exact_match[0]

        if regionspec.startswith("Group"):  # backwards compatibility with old "Group: <region a>, <region b>" specs
            candidates = {
                r
                for spec in self._split_group_spec(regionspec)
                for r in self.find(spec, filter_children=True, find_topmost=find_topmost)
            }
        else:
            candidates = self.find(regionspec, filter_children=True, find_topmost=find_topmost)

        if len(candidates) > 1 and isinstance(regionspec, str):
            # if we have an exact match of words in one region, discard other candidates.
            querywords = {w.replace(',', '').lower() for w in regionspec.split()}
            full_matches = []
            for c in candidates:
                targetwords = {w.lower() for w in c.name.split()}
                if len(querywords & targetwords) == len(targetwords):
                    full_matches.append(c)
            if len(full_matches) == 1:
                candidates = full_matches

        if not candidates:
            raise ValueError(f"'{regionspec}' could not be decoded under '{self.name}'")
        elif len(candidates) == 1:
            return candidates[0]
        else:
            if allow_tuple:
                return tuple(candidates)
            raise RuntimeError(
                f"Spec {regionspec!r} resulted in multiple matches: {', '.join(r.name for r in candidates)}."
            )

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
            logger.warning(
                f"Sorting non-versioned instances of {self.__class__.__name__} "
                f"by name: {self.name}, {other.name}"
            )
            return self.name < other.name
        return self.version.__lt__(other.version)


@lru_cache(maxsize=128)
def find_regions(
    regionspec: str,
    filter_children=True,
    find_topmost=False
):
    """
    Find regions matching the given region specification across all parcellation
    instances in the registry.

    Parameters
    ----------
    regionspec: str, regex, Region
        - a string with a possibly inexact name (matched both against the name and the identifier key)
        - a string in '/pattern/flags' format to use regex search (acceptable flags: aiLmsux) (see https://docs.python.org/3/library/re.html#flags)
        - a regex applied to region names
        - a Region object
    filter_children : bool, default: True
        If True, children of matched parents will not be returned
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
    for p in Parcellation.registry():
        result.extend(
            p.find(
                regionspec=regionspec,
                filter_children=filter_children,
                find_topmost=find_topmost
            )
        )
    return result
