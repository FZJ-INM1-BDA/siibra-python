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
"""Representation of a brain region."""
from . import concept, space as _space, parcellation as _parcellation
from .relation_qualification import Qualification as RegionRelationship, RelationAssignment

from ..locations import boundingbox, point, pointset
from ..volumes import parcellationmap

from ..commons import (
    logger,
    MapType,
    affine_scaling,
    create_key,
    clear_name,
    siibra_tqdm,
    InstanceTable,
    SIIBRA_DEFAULT_MAPTYPE,
    SIIBRA_DEFAULT_MAP_THRESHOLD
)

import numpy as np
import re
import anytree
from typing import List, Set, Union, Iterable, Dict, Callable
from nibabel import Nifti1Image
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from ebrains_drive import BucketApiClient
import json
from functools import wraps
from concurrent.futures import ThreadPoolExecutor


REGEX_TYPE = type(re.compile("test"))

THRESHOLD_STATISTICAL_MAPS = None


@dataclass
class SpatialPropCmpt:
    centroid: point.Point
    volume: int


@dataclass
class SpatialProp:
    cog: SpatialPropCmpt = None
    components: List[SpatialPropCmpt] = field(default_factory=list)
    space: _space.Space = None


class Region(anytree.NodeMixin, concept.AtlasConcept):
    """
    Representation of a region with name and more optional attributes
    """

    _regex_re = re.compile(r'^\/(?P<expression>.+)\/(?P<flags>[a-zA-Z]*)$')
    _accepted_flags = "aiLmsux"

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
        self.parent = parent
        self.children = children
        # convert hex to int tuple if rgb is given
        self.rgb = (
            None if rgb is None
            else tuple(int(rgb[p:p + 2], 16) for p in [1, 3, 5])
        )
        self._supported_spaces = None  # computed on 1st call of self.supported_spaces
        self._CACHED_REGION_SEARCHES = {}
        self._str_aliases = None

    def get_related_regions(self) -> Iterable["RegionRelationAssessments"]:
        """
        Get assements on relations of this region to others defined on EBRAINS.

        Yields
        ------
        RegionRelationAssessments

        Example
        -------
        >>> region = siibra.get_region("monkey", "PG")^M
        >>> for assesment in region.get_related_regions():
        >>>    print(assesment)
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
        regionspec: str, regex, int, Region
            - a string with a possibly inexact name (matched both against the name and the identifier key)
            - a string in '/pattern/flags' format to use regex search (acceptable flags: aiLmsux)
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
        key = (regionspec, filter_children, find_topmost)
        MEM = self._CACHED_REGION_SEARCHES
        if key in MEM:
            return MEM[key]

        if isinstance(regionspec, str):
            regex_match = self._regex_re.match(regionspec)
            if regex_match:
                flags = regex_match.group('flags')
                expression = regex_match.group('expression')

                for flag in flags or []:  # catch if flags is nullish
                    if flag not in self._accepted_flags:
                        raise Exception(f"only accepted flag are in { self._accepted_flags }. {flag} is not within them")
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
        MEM[key] = (
            sorted(
                found_regions,
                reverse=True,
                key=lambda region: SequenceMatcher(None, str(region), regionspec).ratio(),
            )
            if type(regionspec) == str
            else found_regions
        )

        return MEM[key]

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
        if regionspec is None:
            return False

        def splitstr(s):
            return [w for w in re.split(r"[^a-zA-Z0-9.\-]", s) if len(w) > 0]

        if isinstance(regionspec, Region):
            return self == regionspec

        elif isinstance(regionspec, str):
            # string is given, perform lazy string matching
            q = regionspec.lower().strip()
            if q == self.key.lower().strip():
                return True
            elif q == self.id.lower().strip():
                return True
            elif q == self.name.lower().strip():
                return True
            else:
                # match if all words of the query are also included in the region name
                W = splitstr(clear_name(self.name.lower()))
                Q = splitstr(clear_name(regionspec))
                return all([any(
                    q.lower() == w or 'v' + q.lower() == w
                    for w in W
                ) for q in Q])

        # TODO since dropping 3.6 support, maybe reimplement as re.Pattern ?
        elif isinstance(regionspec, REGEX_TYPE):
            # match regular expression
            return any(regionspec.search(s) is not None for s in [self.name, self.key])
        elif isinstance(regionspec, (list, tuple)):
            return any(self.matches(_) for _ in regionspec)
        else:
            raise TypeError(
                f"Cannot interpret region specification of type '{type(regionspec)}'"
            )

    def fetch_regional_map(
        self,
        space: Union[str, _space.Space],
        maptype: MapType = SIIBRA_DEFAULT_MAPTYPE,
        threshold: float = SIIBRA_DEFAULT_MAP_THRESHOLD,
        via_space: Union[str, _space.Space] = None
    ):
        """
        Attempts to build a binary mask of this region in the given space,
        using the specified MapTypes.

        Parameters
        ----------
        space: Space or str
            The requested reference space
        maptype: MapType, default: SIIBRA_DEFAULT_MAPTYPE
            The type of map to be used ('labelled' or 'statistical')
        threshold: float, optional
            When fetching a statistical map, use this threshold to convert
            it to a binary mask
        via_space: Space or str
            If specified, fetch the map in this space first, and then perform
            a linear warping from there to the requested space.

            Tip
            ---
                You might want to use this if a map in the requested space
                is not available.

            Note
            ----
                This linear warping is an affine approximation of the
                nonlinear deformation, computed from the warped corner points
                of the bounding box (see siibra.locations.BoundingBox.estimate_affine()).
                It does not require voxel resampling, just replaces the affine
                matrix, but is less accurate than a full nonlinear warping,
                which is currently not supported in siibra-python for images.
        Returns
        -------
        Nifti1Image
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]
        result = None

        # prepare space instances
        if isinstance(space, str):
            space = _space.Space.get_instance(space)
        fetch_space = space if via_space is None else via_space
        if isinstance(fetch_space, str):
            fetch_space = _space.Space.get_instance(fetch_space)

        for m in parcellationmap.Map.registry():
            if (
                m.space.matches(fetch_space)
                and m.parcellation == self.parcellation
                and m.provides_image
                and m.maptype == maptype
                and self.name in m.regions
            ):
                result = m.fetch(region=self, format='image')
                if (maptype == MapType.STATISTICAL) and (threshold is not None):
                    logger.info(f"Thresholding statistical map at {threshold}")
                    result = Nifti1Image(
                        (result.get_fdata() > threshold).astype('uint8'),
                        result.affine
                    )
                break
        else:
            # all children are mapped instead
            dataobj = None
            affine = None
            if all(c.mapped_in_space(fetch_space) for c in self.children):
                for c in siibra_tqdm(
                    self.children,
                    desc=f"Building mask of {self.name}",
                    leave=False,
                    unit=" child region"
                ):
                    mask = c.fetch_regional_map(fetch_space, maptype, threshold)
                    if dataobj is None:
                        dataobj = np.asanyarray(mask.dataobj)
                        affine = mask.affine
                    else:
                        if np.linalg.norm(mask.affine - affine) > 1e-12:
                            raise NotImplementedError(
                                f"Child regions of {self.name} have different voxel spaces "
                                "and the aggregated subtree mask is not supported. "
                                f"Try fetching masks of the children: {self.children}"
                            )
                        updates = mask.get_fdata() > dataobj
                        dataobj[updates] = mask.get_fdata()[updates]
            if dataobj is not None:
                result = Nifti1Image(dataobj, affine)

        if result is None:
            raise RuntimeError(f"Cannot build mask for {self.name} from {maptype} maps in {fetch_space}")

        if via_space is not None:
            # fetch used an intermediate reference space provided by 'via_space'.
            # We will now transform the affine to match the desired target space.
            bbox = boundingbox.BoundingBox.from_image(result, fetch_space)
            transform = bbox.estimate_affine(space)
            result = Nifti1Image(result.dataobj, np.dot(transform, result.affine))
            logger.info(
                f"Regional map was fetched from {fetch_space.name}, "
                f"then linearly corrected to match {space.name}."
            )

        return result

    def mapped_in_space(self, space, recurse: bool = True) -> bool:
        """
        Verifies wether this region is defined by an explicit map in the given space.

        Parameters
        ----------
        space: Space or str
            reference space
        recurse: bool, default: True
            If True, check if all child regions are mapped instead
        Returns
        -------
        bool
        """
        from ..volumes.parcellationmap import Map
        for m in Map.registry():
            # Use and operant for efficiency (short circuiting logic)
            # Put the most inexpensive logic first
            if (
                self.name in m.regions
                and m.space.matches(space)
                and m.parcellation.matches(self.parcellation)
            ):
                return True
        if recurse and not self.is_leaf:
            # check if all children are mapped instead
            return all(c.mapped_in_space(space, recurse=True) for c in self.children)
        return False

    @property
    def supported_spaces(self) -> Set[_space.Space]:
        """
        The set of spaces for which a mask could be extracted.
        Overwrites the corresponding method of AtlasConcept.
        """
        if self._supported_spaces is None:
            self._supported_spaces = {s for s in _space.Space.registry() if self.mapped_in_space(s)}
        return self._supported_spaces

    def supports_space(self, space: _space.Space):
        """
        Return true if this region supports the given space, else False.
        """
        return any(s.matches(space) for s in self.supported_spaces)

    @property
    def spaces(self):
        return InstanceTable(
            matchfunc=_space.Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

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

    def get_bounding_box(
        self,
        space: _space.Space,
        maptype: MapType = MapType.LABELLED,
        threshold_statistical=None,
    ):
        """Compute the bounding box of this region in the given space.

        Parameters
        ----------
        space: Space or str
            Requested reference space
        maptype: MapType, default: MapType.LABELLED
            Type of map to build ('labelled' will result in a binary mask,
            'statistical' attempts to build a statistical mask, possibly by
            elementwise maximum of statistical maps of children)
        threshold_statistical: float, or None
            if not None, masks will be preferably constructed by thresholding
            statistical maps with the given value.
        Returns
        -------
        BoundingBox
        """
        spaceobj = _space.Space.get_instance(space)
        try:
            mask = self.fetch_regional_map(
                spaceobj, maptype=maptype, threshold=threshold_statistical
            )
            return boundingbox.BoundingBox.from_image(mask, space=spaceobj)
        except (RuntimeError, ValueError):
            for other_space in self.parcellation.spaces - spaceobj:
                try:
                    mask = self.fetch_regional_map(
                        other_space,
                        maptype=maptype,
                        threshold=threshold_statistical,
                    )
                    logger.warning(
                        f"No bounding box for {self.name} defined in {spaceobj.name}, "
                        f"will warp the bounding box from {other_space.name} instead."
                    )
                    bbox = boundingbox.BoundingBox.from_image(mask, space=other_space)
                    if bbox is not None:
                        return bbox.warp(spaceobj)
                except RuntimeError:
                    continue
        logger.error(f"Could not compute bounding box for {self.name}.")
        return None

    def compute_centroids(self, space: _space.Space) -> pointset.PointSet:
        """
        Compute the centroids of the region in the given space.

        Parameters
        ----------
        space: Space
            reference space in which the computation will be performed
        Returns
        -------
        PointSet
            Found centroids (as Point objects) in a PointSet
        Note
        ----
        A region can generally have multiple centroids if it has multiple
        connected components in the map.
        """
        props = self.spatial_props(space)
        return pointset.PointSet(
            [c.centroid for c in props.components],
            space=space
        )

    def spatial_props(
        self,
        space: _space.Space,
        maptype: MapType = MapType.LABELLED,
        threshold_statistical=None,
    ) -> SpatialProp:
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
        threshold_statistical: float, or None
            if not None, masks will be preferably constructed by thresholding
            statistical maps with the given value.

        Returns
        -------
        Dict
            Dictionary of region's spatial properties
        """
        from skimage import measure

        if not isinstance(space, _space.Space):
            space = _space.Space.get_instance(space)

        result = SpatialProp(space=space)

        if not self.mapped_in_space(space):
            logger.warning(
                f"Spatial properties of {self.name} cannot be computed in {space.name}. "
                "This region is only mapped in these spaces: "
                f"{', '.join(s.name for s in self.supported_spaces)}"
            )
            return result

        # build binary mask of the image
        pimg = self.fetch_regional_map(
            space, maptype=maptype, threshold=threshold_statistical
        )

        # determine scaling factor from voxels to cube mm
        scale = affine_scaling(pimg.affine)

        # compute properties of labelled volume
        A = np.asarray(pimg.get_fdata(), dtype=np.int32).squeeze()
        C = measure.label(A)

        # compute spatial properties of each connected component
        for label in range(1, C.max() + 1):
            nonzero = np.c_[np.nonzero(C == label)]
            result.components.append(
                SpatialPropCmpt(
                    centroid=point.Point(
                        np.dot(pimg.affine, np.r_[nonzero.mean(0), 1])[:3], space=space
                    ),
                    volume=nonzero.shape[0] * scale,
                )
            )

        # sort by volume
        result.components.sort(key=lambda cmp: cmp.volume, reverse=True)

        return result

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)


_get_reg_relation_asmgt_types: Dict[str, Callable] = {}


def _register_region_reference_type(ebrain_type: str):
    def outer(fn: Callable):
        _get_reg_relation_asmgt_types[ebrain_type] = fn

        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner
    return outer


class RegionRelationAssessments(RelationAssignment[Region]):

    anony_client = BucketApiClient()

    @staticmethod
    def get_uuid(long_id: Union[str, Dict]):
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
        if isinstance(_id, list):
            assert all(isinstance(_i, str) for _i in _id), "all instances of pev should be str"
        elif isinstance(_id, str):
            _id = [_id]
        else:
            raise RuntimeError("parse_pev error: arg must be either list of str or str")
        return _id

    @classmethod
    def get_object(cls, obj: str):
        bucket = cls.anony_client.buckets.get_bucket("reference-atlas-data")
        return json.loads(bucket.get_file(obj).get_content())

    @classmethod
    def get_snapshot_factory(cls, type_str: str):
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
                yield cls(query_structure=src, assigned_structure=found_target, qualification=RegionRelationship.parse_relation_assessment(overlap))

            if "https://openminds.ebrains.eu/sands/ParcellationEntity" in target.get("type"):
                pev_uuids = [
                    cls.get_uuid(has_version)
                    for pe in cls.get_snapshot_factory("ParcellationEntity")(target_id)
                    for has_version in pe.get("hasVersion")
                ]
                for reg in all_regions:
                    if reg in pev_uuids:
                        yield cls(query_structure=src, assigned_structure=reg, qualification=RegionRelationship.parse_relation_assessment(overlap))

    @classmethod
    @_register_region_reference_type("openminds/CustomAnatomicalEntity")
    def translate_cae(cls, src: "Region", _id: Union[str, List[str]]):
        caes = cls.get_snapshot_factory("CustomAnatomicalEntity")(_id)
        for cae in caes:
            for ass in cae.get("relationAssessment", []):
                yield from cls.parse_relationship_assessment(src, ass)

    @classmethod
    @_register_region_reference_type("openminds/ParcellationEntityVersion")
    def translate_pevs(cls, src: "Region", _id: Union[str, List[str]]):
        pe_uuids = [
            uuid for uuid in
            {
                cls.get_uuid(pe)
                for pev in cls.get_snapshot_factory("ParcellationEntityVersion")(_id)
                for pe in pev.get("isVersionOf")
            }
        ]
        pes = cls.get_snapshot_factory("ParcellationEntity")(pe_uuids)

        all_regions = [
            region
            for p in _parcellation.Parcellation.registry()
            for region in p
        ]

        for pe in pes:

            # other versions
            has_versions = pe.get("hasVersion", [])
            for has_version in has_versions:
                uuid = cls.get_uuid(has_version)

                # ignore if uuid is referring to src region
                if uuid == src:
                    continue

                found_targets = [
                    region
                    for region in all_regions
                    if region == uuid
                ]
                if len(found_targets) == 0:
                    logger.warning(f"other version with uuid {uuid} not found")
                    continue

                for found_target in found_targets:
                    yield cls(
                        query_structure=src,
                        assigned_structure=found_target,
                        qualification=RegionRelationship.OTHER_VERSION
                    )

            # homologuous
            relations = pe.get("inRelationTo", [])
            for relation in relations:
                yield from cls.parse_relationship_assessment(src, relation)

    @classmethod
    def parse_from_region(cls, region: "Region") -> Iterable["RegionRelationAssessments"]:
        if not region._spec:
            return None
        for ebrain_type, ebrain_ref in region._spec.get("ebrains", {}).items():
            if ebrain_type in _get_reg_relation_asmgt_types:
                fn = _get_reg_relation_asmgt_types[ebrain_type]
                yield from fn(cls, region, ebrain_ref)
