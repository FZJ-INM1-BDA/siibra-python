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
"""Representation of a brain region."""

from . import concept, structure, space as _space, parcellation as _parcellation
from .assignment import Qualification, AnatomicalAssignment

from ..retrieval.cache import cache_user_fn
from ..locations import location, pointcloud, boundingbox as _boundingbox
from ..volumes import parcellationmap, volume
from ..commons import (
    logger,
    MapType,
    create_key,
    clear_name,
    InstanceTable,
)
from ..exceptions import NoMapAvailableError, SpaceWarpingFailedError

import re
import anytree
from typing import List, Union, Iterable, Dict, Callable, Tuple, Set
from difflib import SequenceMatcher
from ebrains_drive import BucketApiClient
import json
from functools import wraps, reduce
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


REGEX_TYPE = type(re.compile("test"))

THRESHOLD_STATISTICAL_MAPS = None


class Region(anytree.NodeMixin, concept.AtlasConcept, structure.BrainStructure):
    """
    Representation of a region with name and more optional attributes
    """

    _regex_re = re.compile(r'^\/(?P<expression>.+)\/(?P<flags>[a-zA-Z]*)$')
    _accepted_flags = "aiLmsux"

    _GETMASK_CACHE = {}
    _GETMASK_CACHE_MAX_ENTRIES = 1

    def __init__(
        self,
        name: str,
        children: List["Region"] = [],
        parent: "Region" = None,
        shortname: str = "",
        description: str = "",
        modality: str = "",
        publications: list = [],
        datasets: list = [],
        rgb: str = None,
        spec=None,
        prerelease: bool = False,
    ):
        """
        Constructs a new Region object.

        Parameters
        ----------
        name : str
            Human-readable name of the region
        children: list[Region]
        parent: Region
        shortname: str
            Shortform of human-readable name (optional)
        description: str
            Textual description of the parcellation
        modality:  str or None
            Specification of the modality used for specifying this region
        publications: list
            List of associated publications, each a dictionary with "doi"
            and/or "citation" fields
        datasets: list
            datasets associated with this region
        rgb: str, default: None
            Hexcode of preferred color of this region (e.g. "#9FE770")
        spec: dict, default: None
            The preconfigured specification.
        """
        anytree.NodeMixin.__init__(self)
        concept.AtlasConcept.__init__(
            self,
            identifier=None,  # lazy property implementation below
            name=clear_name(name),
            species=None,  # lazy property implementation below
            shortname=shortname,
            description=description,
            modality=modality,
            publications=publications,
            datasets=datasets,
            spec=spec,
            prerelease=prerelease,
        )

        # anytree node will take care to use this appropriately
        self.parent: "Region" = parent
        self.children: List["Region"] = children
        # convert hex to int tuple if rgb is given
        self.rgb = (
            None if rgb is None
            else tuple(int(rgb[p:p + 2], 16) for p in [1, 3, 5])
        )
        self._supported_spaces: Set[_space.Space] = None  # computed on 1st call of self.supported_spaces
        self._str_aliases = None
        self.find = lru_cache(maxsize=3)(self.find)

    def get_related_regions(self) -> Iterable["RegionRelationAssessments"]:
        """
        Get assessments on relations of this region to others defined on EBRAINS.

        Yields
        ------
        Qualification

        Example
        -------
        >>> region = siibra.get_region("monkey", "PG")
        >>> for assessment in region.get_related_regions():
        >>>    print(assessment)
        'PG' is homologous to 'Area PGa (IPL)'
        'PG' is homologous to 'Area PGa (IPL) left'
        'PG' is homologous to 'Area PGa (IPL) right'
        'PG' is homologous to 'Area PGa (IPL)'
        'PG' is homologous to 'Area PGa (IPL) left'
        'PG' is homologous to 'Area PGa (IPL) right'
        'PG' is homologous to 'Area PGa (IPL)'
        'PG' is homologous to 'Area PGa (IPL) right'
        'PG' is homologous to 'Area PGa (IPL) left'
        """
        yield from RegionRelationAssessments.parse_from_region(self)

    @property
    def id(self):
        if self.parent is None:
            return create_key(self.name)
        else:
            return f"{self.parent.root.id}_{create_key(self.name)}"

    @property
    def parcellation(self):
        if isinstance(self.root, _parcellation.Parcellation):
            return self.root
        else:
            return None

    @property
    def species(self):
        # lazy request of the root parcellation's species
        if self._species_cached is None:
            self._species_cached = self.parcellation.species
        return self._species_cached

    @staticmethod
    def copy(other: 'Region'):
        """
        copy constructor must detach the parent to avoid problems with
        the Anytree implementation.
        """
        # create an isolated object, detached from the other's tree
        region = Region(
            name=other.name,
            children=[Region.copy(c) for c in other.children],
            parent=None,
            shortname=other.shortname,
            description=other.description,
            modality=other.modality,
            publications=other.publications,
            datasets=other.datasets,
            rgb=other.rgb)

        for c in region.children:
            c.parent = region
        return region

    @property
    def names(self):
        return {r.name for r in self}

    def __eq__(self, other):
        """
        Compare this region with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other, Region):
            return self.id == other.id
        if isinstance(other, str):
            if not self._str_aliases:
                self._str_aliases = {
                    self.name,
                    self.key,
                    self.id,
                }
                if self._spec:
                    ebrain_ids = [
                        value
                        for value in self._spec.get("ebrains", {}).values()
                        if isinstance(value, str)
                    ]
                    ebrain_nested_ids = [
                        _id
                        for value in self._spec.get("ebrains", {}).values() if isinstance(value, list)
                        for _id in value
                    ]
                    assert all(isinstance(_id, str) for _id in ebrain_nested_ids)
                    all_ebrain_ids = [
                        *ebrain_ids,
                        *ebrain_nested_ids
                    ]

                    self._str_aliases.update(all_ebrain_ids)

            return other in self._str_aliases
        return False

    def __hash__(self):
        return hash(self.id)

    def has_parent(self, parent):
        return parent in [a for a in self.ancestors]

    def includes(self, region):
        """
        Determine whether this region-tree includes the given region.

        Parameters
        ----------
        region: Region

        Returns
        -------
            bool
                True if the region is in the region-tree.
        """
        return region == self or region in self.descendants

    def find(
        self,
        regionspec,
        filter_children=False,
        find_topmost=True,
    ) -> List['Region']:
        """
        Find regions that match the given region specification in the subtree
        headed by this region.

        Parameters
        ----------
        regionspec: str, regex, Region
            - a string with a possibly inexact name (matched both against the name and the identifier key)
            - a string in '/pattern/flags' format to use regex search (acceptable flags: aiLmsux, see at https://docs.python.org/3/library/re.html#flags)
            - a regex applied to region names
            - a Region object
        filter_children : bool, default: False
            If True, children of matched parents will not be returned
        find_topmost : bool, default: True
            If True (requires `filter_children=True`), will return parent
            structures if all children are matched, even though the parent
            itself might not match the specification.

        Returns
        -------
        list[Region]
            list of regions matching to the regionspec

        Tip
        ---
        See example 01-003, find regions.
        """
        if isinstance(regionspec, str):
            # convert the specified string into a regex for matching
            regex_match = self._regex_re.match(regionspec)
            if regex_match:
                flags = regex_match.group('flags')
                expression = regex_match.group('expression')

                for flag in flags or []:  # catch if flags is nullish
                    if flag not in self._accepted_flags:
                        raise Exception(f"only accepted flag are in {self._accepted_flags}. {flag} is not within them")
                search_regex = (f"(?{flags})" if flags else "") + expression
                regionspec = re.compile(search_regex)

        candidates = list(
            anytree.search.findall(self, lambda node: node.matches(regionspec))
        )

        if len(candidates) > 1 and filter_children:
            filtered = []
            for region in candidates:
                children_included = [c for c in region.children if c in candidates]
                if len(children_included) > 0:
                    filtered.append(region)
                else:
                    if region.parent not in candidates:
                        filtered.append(region)

            # find any non-matched regions of which all children are matched
            if find_topmost:
                complete_parents = list(
                    {
                        r.parent
                        for r in filtered
                        if (r.parent is not None)
                        and all((c in filtered) for c in r.parent.children)
                    }
                )

                if len(complete_parents) == 0:
                    candidates = filtered
                else:
                    # filter child regions again
                    filtered += complete_parents
                    candidates = [
                        r for r in filtered
                        if (r.parent not in filtered) or r == regionspec
                    ]
            else:
                candidates = filtered

        # ensure the result is a list
        if candidates is None:
            candidates = []
        elif isinstance(candidates, Region):
            candidates = [candidates]
        else:
            candidates = list(candidates)

        found_regions = sorted(set(candidates), key=lambda r: r.depth)

        # reverse is set to True, since SequenceMatcher().ratio(), higher == better
        return (
            sorted(
                found_regions,
                reverse=True,
                key=lambda region: SequenceMatcher(None, str(region), regionspec).ratio(),
            )
            if isinstance(regionspec, str) else found_regions
        )

    def matches(self, regionspec):
        """
        Checks whether this region matches the given region specification.

        Parameters
        ----------
        regionspec: str, regex, Region
            - a string with a possibly inexact name, which is matched both against the name and the identifier key,
            - a regex applied to region names,
            - a region object

        Returns
        -------
        bool
            If the regionspec matches to the Region.
        """
        if regionspec not in self._CACHED_MATCHES:
            def splitstr(s):
                return [w for w in re.split(r"[^a-zA-Z0-9.\-]", s) if len(w) > 0]

            if regionspec is None:
                self._CACHED_MATCHES[regionspec] = False

            elif isinstance(regionspec, Region):
                self._CACHED_MATCHES[regionspec] = self == regionspec

            elif isinstance(regionspec, str):
                # string is given, perform lazy string matching
                q = regionspec.lower().strip()
                if q == self.key.lower().strip():
                    self._CACHED_MATCHES[regionspec] = True
                elif q == self.id.lower().strip():
                    self._CACHED_MATCHES[regionspec] = True
                elif q == self.name.lower().strip():
                    self._CACHED_MATCHES[regionspec] = True
                else:
                    # match if all words of the query are also included in the region name
                    W = splitstr(clear_name(self.name.lower()))
                    Q = splitstr(clear_name(regionspec))
                    self._CACHED_MATCHES[regionspec] = all([any(
                        q.lower() == w or 'v' + q.lower() == w
                        for w in W
                    ) for q in Q])

            # TODO since dropping 3.6 support, maybe reimplement as re.Pattern ?
            elif isinstance(regionspec, REGEX_TYPE):
                # match regular expression
                self._CACHED_MATCHES[regionspec] = any(regionspec.search(s) is not None for s in [self.name, self.key])

            elif isinstance(regionspec, (list, tuple)):
                self._CACHED_MATCHES[regionspec] = any(self.matches(_) for _ in regionspec)

            else:
                raise TypeError(
                    f"Cannot interpret region specification of type '{type(regionspec)}'"
                )

        return self._CACHED_MATCHES[regionspec]

    def get_regional_mask(
        self,
        space: Union[str, _space.Space],
        maptype: MapType = MapType.LABELLED,
        threshold: float = 0.0,
    ) -> volume.FilteredVolume:
        """
        Get a binary mask of this region in the given space,
        using the specified MapTypes.

        Parameters
        ----------
        space: Space or str
            The requested reference space
        maptype: MapType, default: SIIBRA_DEFAULT_MAPTYPE
            The type of map to be used ('labelled' or 'statistical')
        threshold: float, default: 0.0
            When fetching a statistical map, use this threshold to convert
            it to a binary mask.

        Returns
        -------
        Volume (use fetch() to get a NiftiImage)
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]

        threshold_info = "" if maptype == MapType.LABELLED else f"(threshold: {threshold}) "
        # check cache
        getmask_hash = hash(f"{self.id} - {space} - {maptype}{threshold_info}")
        if getmask_hash in self._GETMASK_CACHE:
            return self._GETMASK_CACHE[getmask_hash]

        name = f"Mask {threshold_info}of '{self.name} ({self.parcellation})' in "
        try:
            regional_map = self.get_regional_map(space=space, maptype=maptype)
            if maptype == MapType.LABELLED:
                assert threshold == 0.0, f"threshold can only be set for {MapType.STATISTICAL} maps."
                result = volume.FilteredVolume(
                    parent_volume=regional_map,
                    label=regional_map.label,
                    fragment=regional_map.fragment
                )
                result._boundingbox = None
            if maptype == MapType.STATISTICAL:
                result = volume.FilteredVolume(
                    parent_volume=regional_map,
                    threshold=threshold
                )
                if threshold == 0.0:
                    result._boundingbox = regional_map._boundingbox
            name += f"'{result.space}'"
        except NoMapAvailableError as e:
            # This region is not mapped directly in any map in the registry.
            # Try building a map from the child regions
            if (len(self.children) > 0) and self.mapped_in_space(space, recurse=True):
                mapped_descendants: List[Region] = [
                    d for d in self.descendants if d.mapped_in_space(space, recurse=False)
                ]
                logger.info(f"{self.name} is not mapped in {space}. Merging the masks of its {len(mapped_descendants)} map descendants.")
                descendant_volumes = [
                    descendant.get_regional_mask(space=space, maptype=maptype, threshold=threshold)
                    for descendant in mapped_descendants
                ]
                result = volume.FilteredVolume(
                    volume.merge(descendant_volumes),
                    label=1
                )
                name += f"'{result.space}' (built by merging the mask {threshold_info} of its descendants)"
            else:
                raise e
        result._name = name

        while len(self._GETMASK_CACHE) > self._GETMASK_CACHE_MAX_ENTRIES:
            self._GETMASK_CACHE.pop(next(iter(self._GETMASK_CACHE)))
        self._GETMASK_CACHE[getmask_hash] = result
        return result

    def get_regional_map(
        self,
        space: Union[str, _space.Space],
        maptype: MapType = MapType.LABELLED,
    ) -> Union[volume.FilteredVolume, volume.Volume, volume.Subvolume]:
        """
        Get a volume representing this region in the given space and MapType.

        Note
        ----
        If a region is not mapped in any of the `Map`s in the registry, then
        siibra will get the maps of its children recursively and merge them.
        If no map is available this way as well, an exception is raised.

        Parameters
        ----------
        space: Space or str
            The requested reference space
        maptype: MapType, default: SIIBRA_DEFAULT_MAPTYPE
            The type of map to be used ('labelled' or 'statistical')

        Returns
        -------
        Volume (use fetch() to get a NiftiImage)
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]

        # prepare space instance
        if isinstance(space, str):
            space = _space.Space.get_instance(space)

        # see if we find a map supporting the requested region
        for m in parcellationmap.Map.registry():
            if (
                m.space.matches(space)
                and m.parcellation == self.parcellation
                and m.provides_image
                and m.maptype == maptype
                and self.name in m.regions
            ):
                return m.get_volume(region=self)
        raise NoMapAvailableError(
            f"{self.name} is not mapped in {space} as a {str(maptype)} map."
            " Please try getting the children or getting the mask."
        )

    def mapped_in_space(self, space, recurse: bool = False) -> bool:
        """
        Verifies whether this region is defined by an explicit map in the given space.

        Parameters
        ----------
        space: Space or str
            reference space
        recurse: bool, default: False
            If True, check if itself or all child regions are mapped instead recursively.
        Returns
        -------
        bool
        """
        from ..volumes.parcellationmap import Map
        for m in Map.registry():
            if (
                m.space.matches(space)
                and m.parcellation.matches(self.parcellation)
                and self.name in m.regions
            ):
                return True
        if recurse and not self.is_leaf:
            # check if all children are mapped instead
            return all(c.mapped_in_space(space, recurse=True) for c in self.children)
        return False

    @property
    def supported_spaces(self) -> List[_space.Space]:
        """
        The list of spaces for which a mask could be extracted from an existing
        map or combination of masks of its children.
        """
        if self._supported_spaces is None:
            self._supported_spaces = sorted({
                s for s in _space.Space.registry()
                if self.mapped_in_space(s, recurse=True)
            })
        return self._supported_spaces

    @property
    def spaces(self):
        return InstanceTable(
            matchfunc=_space.Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    def __contains__(self, other: Union[location.Location, 'Region']) -> bool:
        if isinstance(other, Region):
            return len(self.find(other)) > 0
        else:
            try:
                regionmap = self.get_regional_mask(space=other.space)
                return regionmap.__contains__(other)
            except NoMapAvailableError:
                return False

    def assign(self, other: structure.BrainStructure) -> AnatomicalAssignment:
        """
        Compute assignment of a location to this region.

        Two cases:
        1) other is location -> get region map, call regionmap.assign(other)
        2) other is region -> just do a semantic check for the regions

        Parameters
        ----------
        other : Location or Region

        Returns
        -------
        AnatomicalAssignment or None
            None if there is no Qualification found.
        """
        if (self, other) in self._ASSIGNMENT_CACHE:
            return self._ASSIGNMENT_CACHE[self, other]
        if (other, self) in self._ASSIGNMENT_CACHE:
            if self._ASSIGNMENT_CACHE[other, self] is None:
                return None
            return self._ASSIGNMENT_CACHE[other, self].invert()

        if isinstance(other, (location.Location, volume.Volume)):
            if self.mapped_in_space(other.space, recurse=True):
                regionmap = self.get_regional_mask(other.space)
                self._ASSIGNMENT_CACHE[self, other] = regionmap.assign(other)
                return self._ASSIGNMENT_CACHE[self, other]

            if isinstance(other, _boundingbox.BoundingBox):  # volume.intersection(bbox) gets boundingbox anyway
                try:
                    regionbbox_otherspace = self.get_boundingbox(other.space, restrict_space=False)
                    if regionbbox_otherspace is not None:
                        self._ASSIGNMENT_CACHE[self, other] = regionbbox_otherspace.assign(other)
                        return self._ASSIGNMENT_CACHE[self, other]
                except Exception as e:
                    logger.debug(e)

            assignment_result = None
            for targetspace in self.supported_spaces:
                try:
                    other_warped = other.warp(targetspace)
                    regionmap = self.get_regional_mask(targetspace)
                    assignment_result = regionmap.assign(other_warped)
                except SpaceWarpingFailedError:
                    try:
                        regionbbox_targetspace = self.get_boundingbox(
                            targetspace, restrict_space=True
                        )
                        if regionbbox_targetspace is None:
                            continue
                        regionbbox_warped = regionbbox_targetspace.warp(other.space)
                    except SpaceWarpingFailedError:
                        continue
                    assignment_result = regionbbox_warped.assign(other)
                except Exception as e:
                    logger.debug(e)
                    continue
                break
            self._ASSIGNMENT_CACHE[self, other] = assignment_result
        else:  # other is a Region
            assert isinstance(other, Region)
            if self == other:
                qualification = Qualification.EXACT
            elif self.__contains__(other):
                qualification = Qualification.CONTAINS
            elif other.__contains__(self):
                qualification = Qualification.CONTAINED
            else:
                qualification = None
            if qualification is None:
                self._ASSIGNMENT_CACHE[self, other] = None
            else:
                self._ASSIGNMENT_CACHE[self, other] = AnatomicalAssignment(self, other, qualification)
        return self._ASSIGNMENT_CACHE[self, other]

    def tree2str(self):
        """Render region-tree as a string"""
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node
            in anytree.RenderTree(self, style=anytree.render.ContRoundStyle)
        )

    def render_tree(self):
        """Prints the tree representation of the region"""
        print(self.tree2str())

    def get_boundingbox(
        self,
        space: _space.Space,
        maptype: MapType = MapType.LABELLED,
        threshold_statistical: float = 0.0,
        restrict_space: bool = True,
        **fetch_kwargs
    ) -> Union[_boundingbox.BoundingBox, None]:
        """
        Compute the bounding box of this region in the given space.

        Parameters
        ----------
        space: Space or str
            Requested reference space
        maptype: MapType, default: MapType.LABELLED
            Type of map to build ('labelled' will result in a binary mask,
            'statistical' attempts to build a statistical mask, possibly by
            elementwise maximum of statistical maps of children)
        threshold_statistical: float, default: 0.0
            When masking a statistical map, use this threshold to convert
            it to a binary mask before finding its bounding box.
        restrict_space: bool, default: False
            If True, it will not try to fetch maps from other spaces and warp
            its boundingbox to requested space.

        Returns
        -------
        BoundingBox
        """
        spaceobj = _space.Space.get_instance(space)
        try:
            mask = self.get_regional_mask(
                spaceobj, maptype=maptype, threshold=threshold_statistical
            )
            return mask.get_boundingbox(
                clip=True,
                background=0.0,
                **fetch_kwargs
            )
        except (RuntimeError, ValueError):
            if restrict_space:
                return None
            for other_space in self.parcellation.spaces - spaceobj:
                try:
                    mask = self.get_regional_mask(
                        other_space,
                        maptype=maptype,
                        threshold=threshold_statistical,
                    )
                    bbox = mask.get_boundingbox(clip=True, background=0.0, **fetch_kwargs)
                    if bbox is not None:
                        try:
                            bbox_warped = bbox.warp(spaceobj)
                        except SpaceWarpingFailedError:
                            continue
                        logger.debug(
                            f"No bounding box for {self.name} defined in {spaceobj.name}, "
                            f"warped the bounding box from {other_space.name} instead."
                        )
                        return bbox_warped
                except RuntimeError:
                    continue
        logger.error(f"Could not compute bounding box for {self.name}.")
        return None

    def compute_centroids(
        self,
        space: _space.Space,
        maptype: MapType = MapType.LABELLED,
        threshold_statistical: float = 0.0,
        split_components: bool = True,
        **fetch_kwargs,
    ) -> pointcloud.PointCloud:
        """
        Compute the centroids of the region in the given space.

        Parameters
        ----------
        space: Space
            reference space in which the computation will be performed
        maptype: MapType, default: MapType.LABELLED
            Type of map to build ('labelled' will result in a binary mask,
            'statistical' attempts to build a statistical mask, possibly by
            elementwise maximum of statistical maps of children)
        threshold_statistical: float, default: 0.0
            When masking a statistical map, use this threshold to convert
            it to a binary mask before finding its centroids.

        Returns
        -------
        PointCloud
            Found centroids (as Point objects) in a PointCloud

        Note
        ----
        A region can generally have multiple centroids if it has multiple
        connected components in the map.
        """
        props = self.spatial_props(
            space=space,
            maptype=maptype,
            threshold_statistical=threshold_statistical,
            split_components=split_components,
            **fetch_kwargs,
        )
        return pointcloud.PointCloud(
            [c.centroid for c in props],
            space=space
        )

    def spatial_props(
        self,
        space: _space.Space,
        maptype: MapType = MapType.LABELLED,
        threshold_statistical: float = 0.0,
        split_components: bool = True,
        **fetch_kwargs,
    ):
        """
        Compute spatial properties for connected components of this region in the given space.

        Parameters
        ----------
        space: Space
            reference space in which the computation will be performed
        maptype: MapType, default: MapType.LABELLED
            Type of map to build ('labelled' will result in a binary mask,
            'statistical' attempts to build a statistical mask, possibly by
            elementwise maximum of statistical maps of children)
        threshold_statistical: float, default: 0.0
            if not None, masks will be preferably constructed by thresholding
            statistical maps with the given value.

        Returns
        -------
        List
            List of region's component spatial properties
        """
        if not isinstance(space, _space.Space):
            space = _space.Space.get_instance(space)

        # build binary mask of the image
        try:
            region_vol = self.get_regional_mask(
                space, maptype=maptype, threshold=threshold_statistical
            )
        except NoMapAvailableError:
            raise ValueError(
                f"Spatial properties of {self.name} cannot be computed in {space.name}. "
                "This region is only mapped in these spaces: "
                f"{', '.join(s.name for s in self.supported_spaces)}"
            )

        return region_vol.compute_spatial_props(
            split_components=split_components, **fetch_kwargs
        )

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)

    def intersection(self, other: "location.Location") -> "location.Location":
        """Use this region for filtering a location object."""

        if self.mapped_in_space(other.space, recurse=True):
            try:
                volume = self.get_regional_mask(other.space)
                if volume is not None:
                    return volume.intersection(other)
            except NotImplementedError:
                intersections = [child.intersection(other) for child in self.children]
                return reduce(lambda a, b: a.union(b), intersections)

        for space in self.supported_spaces:
            if space.provides_image:
                try:
                    volume = self.get_regional_mask(space)
                    if volume is not None:
                        intersection = volume.intersection(other)
                        logger.info(f"Warped {other} to {space} to find the intersection.")
                        return intersection
                except SpaceWarpingFailedError:
                    continue

        return None


@cache_user_fn
def _get_related_regions_str(pe_id: str) -> Tuple[Tuple[str, str, str, str], ...]:
    logger.info("LONG CALC...", pe_id)
    return_val = []
    region_relation_assessments = RegionRelationAssessments.translate_pes(pe_id, pe_id)
    for asgmt in region_relation_assessments:
        assert isinstance(asgmt, RegionRelationAssessments), f"Expecting type to be of RegionRelationAssessments, but is {type(asgmt)}"
        assert isinstance(asgmt.assigned_structure, Region), f"Expecting assigned structure to be of type Region, but is {type(asgmt.assigned_structure)}"
        return_val.append((
            asgmt.assigned_structure.parcellation.id,
            asgmt.assigned_structure.name,
            asgmt.qualification.name,
            asgmt.explanation
        ))
    return tuple(return_val)


def get_peid_from_region(region: Region) -> str:
    """
    Given a region, obtain the Parcellation Entity ID.

    Parameters
    ----------
    region : Region

    Returns
    -------
    str
    """
    if region._spec:
        region_peid = region._spec.get("ebrains", {}).get("openminds/ParcellationEntity")
        if region_peid:
            return region_peid
    # In some cases (e.g. Julich Brain, PE is defined on the parent leaf nodes)
    if region.parent and region.parent._spec:
        parent_peid = region.parent._spec.get("ebrains", {}).get("openminds/ParcellationEntity")
        if parent_peid:
            return parent_peid
    return None


def get_related_regions(region: Region) -> Iterable["RegionRelationAssessments"]:
    """
    Get assessments on relations of a region to others defined on EBRAINS.

    Parameters
    ----------
    region: Region

    Yields
    ------
    Qualification

    Example
    -------
    >>> region = siibra.get_region("monkey", "PG")
    >>> for assessment in siibra.core.region.get_related_regions(region):
    >>>    print(assessment)
    'PG' is homologous to 'Area PGa (IPL)'
    'PG' is homologous to 'Area PGa (IPL) left'
    'PG' is homologous to 'Area PGa (IPL) right'
    'PG' is homologous to 'Area PGa (IPL)'
    'PG' is homologous to 'Area PGa (IPL) left'
    'PG' is homologous to 'Area PGa (IPL) right'
    'PG' is homologous to 'Area PGa (IPL)'
    'PG' is homologous to 'Area PGa (IPL) right'
    'PG' is homologous to 'Area PGa (IPL) left'
    """
    logger.info("get related region called")
    pe_id = get_peid_from_region(region)
    if not pe_id:
        return []

    for parc_id, region_name, qual, explanation in _get_related_regions_str(pe_id):
        parc = _parcellation.Parcellation.get_instance(parc_id)
        found_region = parc.get_region(region_name)
        yield RegionRelationAssessments(region, found_region, qual, explanation)


_get_reg_relation_asmgt_types: Dict[str, Callable] = {}


def _register_region_reference_type(ebrain_type: str):
    def outer(fn: Callable):
        _get_reg_relation_asmgt_types[ebrain_type] = fn

        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner
    return outer


class RegionRelationAssessments(AnatomicalAssignment[Region]):
    """
    A collection of methods on finding related regions and the quantification
    of the relationship.
    """

    anony_client = BucketApiClient()

    @staticmethod
    def get_uuid(long_id: Union[str, Dict]) -> str:
        """
        Returns the uuid portion of either a fully formed openminds id, or get
        the 'id' property first, and extract the uuid portion of the id.

        Parameters
        ----------
        long_id: str, dict[str, str]

        Returns
        -------
        str

        Raises
        ------
        AssertionError
        RuntimeError
        """
        if isinstance(long_id, str):
            pass
        elif isinstance(long_id, dict):
            long_id = long_id.get("id")
            assert isinstance(long_id, str)
        else:
            raise RuntimeError("uuid arg must be str or object")
        uuid_search = re.search(r"(?P<uuid>[a-f0-9-]+)$", long_id)
        assert uuid_search, "uuid not found"
        return uuid_search.group("uuid")

    @staticmethod
    def parse_id_arg(_id: Union[str, List[str]]) -> List[str]:
        """
        Normalizes the ebrains id property. The ebrains id field can be either
        a str or list[str]. This method normalizes it to always be list[str].

        Parameters
        ----------
        _id: strl, list[str]

        Returns
        -------
        list[str]

        Raises
        ------
        RuntimeError
        """
        if isinstance(_id, list):
            assert all(isinstance(_i, str) for _i in _id), "all instances of pev should be str"
        elif isinstance(_id, str):
            _id = [_id]
        else:
            raise RuntimeError("parse_pev error: arg must be either list of str or str")
        return _id

    @classmethod
    def get_object(cls, obj: str):
        """
        Gets given a object (path), loads the content and serializes to json.
        Relative to the bucket 'reference-atlas-data'.

        Parameters
        ----------
        obj: str

        Returns
        -------
        dict
        """
        bucket = cls.anony_client.buckets.get_bucket("reference-atlas-data")
        return json.loads(bucket.get_file(obj).get_content())

    @classmethod
    def get_snapshot_factory(cls, type_str: str):
        """
        Factory method for given type.

        Parameters
        ----------
        type_str: str

        Returns
        -------
        Callable[[str|list[str]], dict]
        """
        def get_objects(_id: Union[str, List[str]]):
            _id = cls.parse_id_arg(_id)
            with ThreadPoolExecutor() as ex:
                return list(
                    ex.map(
                        cls.get_object,
                        [f"ebrainsquery/v3/{type_str}/{_}.json" for _ in _id]
                    ))
        return get_objects

    @classmethod
    def parse_relationship_assessment(cls, src: "Region", assessment):
        """
        Given a region, and the fetched assessment json, yield
        RegionRelationAssignment object.

        Parameters
        ----------
        src: Region
        assessment: dict

        Returns
        -------
        Iterable[RegionRelationAssessments]
        """
        all_regions = [
            region
            for p in _parcellation.Parcellation.registry()
            for region in p
        ]

        overlap = assessment.get("qualitativeOverlap")
        targets = assessment.get("relationAssessment") or assessment.get("inRelationTo")
        assert len(overlap) == 1, f"should be 1&o1 overlap {len(overlap)!r} "
        overlap, = overlap
        for target in targets:
            target_id = cls.get_uuid(target)

            found_targets = [
                region
                for region in all_regions
                if region == target_id
            ]

            for found_target in found_targets:
                yield cls(
                    query_structure=src,
                    assigned_structure=found_target,
                    qualification=Qualification.parse_relation_assessment(overlap)
                )

            if "https://openminds.ebrains.eu/sands/ParcellationEntity" in target.get("type"):
                pev_uuids = [
                    cls.get_uuid(has_version)
                    for pe in cls.get_snapshot_factory("ParcellationEntity")(target_id)
                    for has_version in pe.get("hasVersion")
                ]
                for reg in all_regions:
                    if reg in pev_uuids:
                        yield cls(
                            query_structure=src,
                            assigned_structure=reg,
                            qualification=Qualification.parse_relation_assessment(overlap)
                        )

    @classmethod
    @_register_region_reference_type("openminds/CustomAnatomicalEntity")
    def translate_cae(cls, src: "Region", _id: Union[str, List[str]]):
        """Register how CustomAnatomicalEntity should be parsed

        Parameters
        ----------
        src: Region
        _id: str|list[str]

        Returns
        -------
        Iterable[RegionRelationAssessments]
        """
        caes = cls.get_snapshot_factory("CustomAnatomicalEntity")(_id)
        for cae in caes:
            for ass in cae.get("relationAssessment", []):
                yield from cls.parse_relationship_assessment(src, ass)

    @classmethod
    @_register_region_reference_type("openminds/ParcellationEntity")
    def translate_pes(cls, src: "Region", _id: Union[str, List[str]]):
        """
        Register how ParcellationEntity should be parsed

        Parameters
        ----------
        src: Region
        _id: str|list[str]

        Returns
        -------
        Iterable[RegionRelationAssessments]
        """
        pes = cls.get_snapshot_factory("ParcellationEntity")(_id)

        all_regions = [
            region
            for p in _parcellation.Parcellation.registry()
            for region in p
        ]

        for pe in pes:
            for region in all_regions:
                if region is src:
                    continue
                region_peid = get_peid_from_region(region)
                if region_peid and (region_peid in pe.get("id")):
                    yield cls(
                        query_structure=src,
                        assigned_structure=region,
                        qualification=Qualification.OTHER_VERSION
                    )

            # homologuous
            relations = pe.get("inRelationTo", [])
            for relation in relations:
                yield from cls.parse_relationship_assessment(src, relation)

    @classmethod
    @_register_region_reference_type("openminds/ParcellationEntityVersion")
    def translate_pevs(cls, src: "Region", _id: Union[str, List[str]]):
        """
        Register how ParcellationEntityVersion should be parsed

        Parameters
        ----------
        src: Region
        _id: str|list[str]

        Returns
        -------
        Iterable[RegionRelationAssessments]
        """
        pe_uuids = [
            uuid for uuid in
            {
                cls.get_uuid(pe)
                for pev in cls.get_snapshot_factory("ParcellationEntityVersion")(_id)
                for pe in pev.get("isVersionOf")
            }
        ]
        yield from cls.translate_pes(src, pe_uuids)

    @classmethod
    def parse_from_region(cls, region: "Region") -> Iterable["RegionRelationAssessments"]:
        """
        Main entry on how related regions should be retrieved. Given a region,
        retrieves all RegionRelationAssessments

        Parameters
        ----------
        region: Region

        Returns
        -------
        Iterable[RegionRelationAssessments]
        """
        if not region._spec:
            return None
        for ebrain_type, ebrain_ref in region._spec.get("ebrains", {}).items():
            if ebrain_type in _get_reg_relation_asmgt_types:
                fn = _get_reg_relation_asmgt_types[ebrain_type]
                yield from fn(cls, region, ebrain_ref)
