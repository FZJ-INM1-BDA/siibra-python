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

from .concept import AtlasConcept
from .space import Space
from ..locations import PointSet, Point, BoundingBox

from ..commons import (
    logger,
    MapIndex,
    MapType,
    compare_maps,
    affine_scaling,
    create_key,
    clear_name,
    InstanceTable,
    SIIBRA_DEFAULT_MAPTYPE,
    SIIBRA_DEFAULT_MAP_THRESHOLD,
)
from ..retrieval.repositories import GitlabConnector

import numpy as np
import re
import anytree
from typing import List, Set
from nibabel import Nifti1Image
from difflib import SequenceMatcher


REGEX_TYPE = type(re.compile("test"))

THRESHOLD_CONTINUOUS_MAPS = None


class Region(anytree.NodeMixin, AtlasConcept):
    """
    Representation of a region with name and more optional attributes
    """

    CONNECTOR = GitlabConnector(
        server="https://jugit.fz-juelich.de", project=3009, reftag="master"
    )

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
    ):
        """
        Constructs a new Region object.

        Parameters
        ----------
        name : str
            Human-readable name of the region
        children: list of Regions,
        parent: Region
        shortname: str
            Shortform of human-readable name (optional)
        description: str
            Textual description of the parcellation
        modality  :  str or None
            Specification of the modality used for specifying this region
        publications: list
            List of ssociated publications, each a dictionary with "doi" and/or "citation" fields
        datasets : list
            datasets associated with this region
        rgb: str, default None
            Hexcode of preferred color of this region (e.g. "#9FE770")
        """
        anytree.NodeMixin.__init__(self)
        AtlasConcept.__init__(
            self,
            identifier=None,  # Region overwrites the property function below!
            name=clear_name(name),
            shortname=shortname,
            description=description,
            modality=modality,
            publications=publications,
            datasets=datasets,
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

    @property
    def id(self):
        if self.parent is None:
            return create_key(self.name)
        else:
            return f"{self.parent.root.id}_{create_key(self.name)}"

    @property
    def parcellation(self):
        return self.root

    @staticmethod
    def copy(other):
        """
        copy contructor must detach the parent to avoid problems with
        the Anytree implementation.
        """
        # create an isolated object, detached from the other's tree
        region = Region(other.name, other.parcellation, other.index, other.attrs)

        # Build the new subtree recursively
        region.children = tuple(Region.copy(c) for c in other.children)
        for c in region.children:
            c.parent = region
        return region

    @property
    def labels(self):
        return {r.index.label for r in self if r.index.label is not None}

    @property
    def names(self):
        return {r.key for r in self}

    def __eq__(self, other):
        """
        Compare this region with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other, Region):
            return self.id == other.id
        elif isinstance(other, str):
            return any([self.name == other, self.key == other, self.id == other])
        else:
            raise ValueError(
                f"Cannot compare object of type {type(other)} to {self.__class__.__name__}"
            )

    def __hash__(self):
        return hash(self.id)

    def has_parent(self, parent):
        return parent in [a for a in self.ancestors]

    def includes(self, region):
        """
        Determine wether this regiontree includes the given region.
        """
        return region == self or region in self.descendants

    def find(
        self,
        regionspec,
        filter_children=False,
        find_topmost=True,
    ):
        """
        Find regions that match the given region specification in the subtree
        headed by this region.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - a regex applied to region names,
            - an integer, which is interpreted as a labelindex,
            - a full MapIndex
            - a region object
        filter_children : Boolean
            If true, children of matched parents will not be returned
        find_topmost : Bool, default: True
            If True, will return parent structures if all children are matched,
            even though the parent itself might not match the specification.

        Yield
        -----
        list of matching regions
        """
        key = (regionspec, filter_children, find_topmost)
        MEM = self._CACHED_REGION_SEARCHES
        if key in MEM:
            return MEM[key]

        if isinstance(regionspec, str) and regionspec in self.names:
            # key is given, this gives us an exact region
            match = anytree.search.find_by_attr(self, regionspec, name="key")
            MEM[key] = [] if match is None else [match]
            return MEM[key]

        candidates = list(
            set(anytree.search.findall(self, lambda node: node.matches(regionspec)))
        )

        if len(candidates) > 1 and filter_children:

            filtered = []
            for region in candidates:
                children_included = [c for c in region.children if c in candidates]
                # if the parcellation index matches only approximately,
                # while a child has an exact matching index, use the child.
                if len(children_included) > 0:
                    if not (
                        isinstance(regionspec, MapIndex)
                        and (region.index != regionspec)
                        and any(c.index == regionspec for c in children_included)
                    ):
                        filtered.append(region)
                else:
                    if region.parent not in candidates:
                        filtered.append(region)
                    else:
                        if (
                            isinstance(regionspec, MapIndex)
                            and (region.index == regionspec)
                            and (region.parent.index != regionspec)
                        ):
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
                    candidates = [r for r in filtered if r.parent not in filtered]
            else:
                candidates = filtered

        # ensure the result is a list
        if candidates is None:
            candidates = []
        elif isinstance(candidates, Region):
            candidates = [candidates]
        else:
            candidates = list(candidates)

        found_regions = sorted(candidates, key=lambda r: r.depth)

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
        Checks wether this region matches the given region specification.

        Parameters
        ---------

        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - a regex applied to region names,
            - a region object

        Yield
        -----
        True or False
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
        else:
            raise TypeError(
                f"Cannot interpret region specification of type '{type(regionspec)}'"
            )

    def build_mask(
        self,
        space,
        maptype: MapType = SIIBRA_DEFAULT_MAPTYPE,
        threshold_continuous: float = SIIBRA_DEFAULT_MAP_THRESHOLD
    ):
        """
        Attempts to build a binary mask of this region in the given space,
        using the specified maptypes.
        """
        from ..volumes import Map  # keep here to avoid circular import
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]
        result = None
        for m in Map.registry():
            if all([
                m.space.matches(space),
                not m.is_surface,
                m.maptype == maptype,
                m.parcellation == self.parcellation,
                self.name in m.regions
            ]):
                result = m.fetch(index=m.get_index(self.name))
                if threshold_continuous is not None:
                    assert maptype == MapType.CONTINUOUS
                    logger.info(f"Thresholding continuous map at {threshold_continuous}")
                    result = Nifti1Image(
                        (result.get_fdata() > threshold_continuous).astype('uint8'),
                        result.affine
                    )
                break
        else:
            # all children are mapped instead
            dataobj = None
            affine = None
            if all(c.mapped_in_space(space) for c in self.children):
                for c in self.children:
                    mask = c.build_mask(space, maptype, threshold_continuous)
                    if dataobj is None:
                        dataobj = np.asanyarray(mask.dataobj)
                        affine = mask.affine
                    else:
                        assert np.linalg.norm(mask.affine - affine) < 1e-12
                        updates = mask.get_fdata() > dataobj
                        dataobj[updates] = mask.get_fdata()[updates]
            if dataobj is not None:
                result = Nifti1Image(dataobj, affine)

        if result is None:
            raise RuntimeError(f"Cannot build mask for {self.name} from {maptype} maps in {space}")

        return result

    def mapped_in_space(self, space) -> bool:
        """
        Verifies wether this region is defined by an explicit map in the given space.
        """
        from ..volumes.parcellationmap import Map
        for m in Map.registry():
            if all([
                m.space.matches(space),
                m.parcellation.matches(self.parcellation),
                self.name in m.regions,
            ]):
                return True
        if not self.is_leaf:
            # check if all children are mapped instead
            return all(c.mapped_in_space(space) for c in self.children)
        return False

    @property
    def supported_spaces(self) -> Set[Space]:
        """
        The set of spaces for which a mask could be extracted.
        Overwrites the corresponding method of AtlasConcept.
        """
        if self._supported_spaces is None:
            self._supported_spaces = {s for s in Space.registry() if self.mapped_in_space(s)}
        return self._supported_spaces

    def supports_space(self, space: Space):
        """
        Return true if this region supports the given space, else False.
        """
        return any(s.matches(space) for s in self.supported_spaces)

    @property
    def spaces(self):
        return InstanceTable(
            matchfunc=Space.matches,
            elements={s.key: s for s in self.supported_spaces},
        )

    def __getitem__(self, labelindex):
        """
        Given an integer label index, return the corresponding region.
        If multiple matches are found, return the unique parent if possible.
        Otherwise, return None

        Parameters
        ----------

        regionlabel: int
            label index of the desired region.

        Return
        ------
        Region object
        """
        if not isinstance(labelindex, int):
            raise TypeError(
                "Index access into the regiontree expects label indices of integer type"
            )

        # first test this head node
        if self.index.label == labelindex:
            return self

        # Consider children, and return the one with smallest depth
        matches = list(
            filter(lambda x: x is not None, [c[labelindex] for c in self.children])
        )
        if matches:
            parentmatches = [m for m in matches if m.parent not in matches]
            if len(parentmatches) == 1:
                return parentmatches[0]

        return None

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.tree2str()

    def tree2str(self):
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node in anytree.RenderTree(self)
        )

    def get_bounding_box(
        self,
        space: Space,
        maptype: MapType = MapType.LABELLED,
        threshold_continuous=None,
    ):
        """Compute the bounding box of this region in the given space.

        Parameters
        ----------
        space : Space or str):
            Requested reference space
        maptype: MapType
            Type of map to build ('labelled' will result in a binary mask,
            'continuous' attempts to build a continuous mask, possibly by
            elementwise maximum of continuous maps of children )
        threshold_continuous: float, or None
            if not None, masks will be preferably constructed by thresholding
            continuous maps with the given value.
        Returns:
            BoundingBox
        """
        spaceobj = Space.get_instance(space)
        try:
            mask = self.build_mask(
                spaceobj, maptype=maptype, threshold_continuous=threshold_continuous
            )
            return BoundingBox.from_image(mask, space=spaceobj)
        except (RuntimeError, ValueError):
            for other_space in self.parcellation.spaces - spaceobj:
                try:
                    mask = self.build_mask(
                        other_space,
                        maptype=maptype,
                        threshold_continuous=threshold_continuous,
                    )
                    logger.warn(
                        f"No bounding box for {self.name} defined in {spaceobj.name}, "
                        f"will warp the bounding box from {other_space.name} instead."
                    )
                    bbox = BoundingBox.from_image(mask, space=other_space)
                    if bbox is not None:
                        return bbox.warp(spaceobj)
                except RuntimeError:
                    continue
        logger.error(f"Could not compute bounding box for {self.name}.")
        return None

    def find_peaks(self, space: Space, min_distance_mm=5):
        """
        Find peaks of the region's continuous map in the given space, if any.

        Arguments:
        ----------
        space : Space
            requested reference space
        min_distance_mm : float
            Minimum distance between peaks in mm

        Returns:
        --------
        peaks: PointSet
        pmap: continuous map
        """
        spaceobj = Space.get_instance(space)
        pmap = self.get_regional_map(spaceobj, MapType.CONTINUOUS)
        if pmap is None:
            logger.warn(
                f"No continuous map found for {self.name} in {spaceobj.name}, "
                "cannot compute peaks."
            )
            return PointSet([], space)

        from skimage.feature.peak import peak_local_max

        img = pmap.fetch()
        dist = int(min_distance_mm / affine_scaling(img.affine) + 0.5)
        voxels = peak_local_max(
            img.get_fdata(),
            exclude_border=False,
            min_distance=dist,
        )
        return (
            PointSet(
                [np.dot(img.affine, [x, y, z, 1])[:3] for x, y, z in voxels],
                space=spaceobj,
            ),
            img,
        )

    def centroids(self, space: Space) -> List[Point]:
        """Compute the centroids of the region in the given space.

        Note that a region can generally have multiple centroids
        if it has multiple connected components in the map.
        """
        props = self.spatial_props(space)
        return [c["centroid"] for c in props["components"]]

    def spatial_props(
        self,
        space: Space,
        maptype: MapType = MapType.LABELLED,
        threshold_continuous=None,
    ):
        """
        Compute spatial properties for connected components of this region in the given space.

        Parameters
        ----------
        space : Space
            the space in which the computation shall be performed
        maptype: MapType
            Type of map to build ('labelled' will result in a binary mask,
            'continuous' attempts to build a continuous mask, possibly by
            elementwise maximum of continuous maps of children )
        threshold_continuous: float, or None
            if not None, masks will be preferably constructed by thresholding
            continuous maps with the given value.

        Return
        ------
        dictionary of regionprops.
        """
        from skimage import measure

        if not isinstance(space, Space):
            space = Space.get_instance(space)

        if not self.mapped_in_space(space):
            raise RuntimeError(
                f"Spatial properties of {self.name} cannot be computed in {space.name}. "
                "This region is only mapped in these spaces: "
                ", ".join(s.name for s in self.supported_spaces)
            )

        # build binary mask of the image
        pimg = self.build_mask(
            space, maptype=maptype, threshold_continuous=threshold_continuous
        )

        # determine scaling factor from voxels to cube mm
        scale = affine_scaling(pimg.affine)

        # compute properties of labelled volume
        A = np.asarray(pimg.get_fdata(), dtype=np.int32).squeeze()
        C = measure.label(A)

        # compute spatial properties of each connected component
        result = {"space": space, "components": []}
        for label in range(1, C.max() + 1):
            props = {}
            nonzero = np.c_[np.nonzero(C == label)]
            props["centroid"] = Point(
                np.dot(pimg.affine, np.r_[nonzero.mean(0), 1])[:3], space=space
            )
            props["volume"] = nonzero.shape[0] * scale

            result["components"].append(props)

        return result

    def compare(
        self,
        img: Nifti1Image,
        space: Space,
        use_maptype: MapType = MapType.CONTINUOUS,
        threshold_continuous: float = None,
        resolution_mm: float = None,
    ):
        """
        Compare the given image to the map of this region in the specified space.

        Parameters:
        -----------
        img: Nifti1Image
            Image to compare with
        space: Space
            Reference space to use
        use_maptype: MapType
            Type of map to build ('labelled' will result in a binary mask,
            'continuous' attempts to build a continuous mask, possibly by
            elementwise maximum of continuous maps of children )
        threshold_continuous: float, or None
            if not None, masks will be preferably constructed by thresholding
            continuous maps with the given value.
        """
        mask = self.build_mask(
            space,
            resolution_mm,
            maptype=use_maptype,
            threshold_continuous=threshold_continuous,
        )
        return compare_maps(mask, img)

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)


if __name__ == "__main__":

    definition = {
        "name": "Interposed Nucleus (Cerebellum) - left hemisphere",
        "rgb": [170, 29, 10],
        "labelIndex": 251,
        "ngId": "jubrain mni152 v18 left",
        "children": [],
        "position": [-9205882, -57128342, -32224599],
        "originDatasets": [
            {
                "kgId": "658a7f71-1b94-4f4a-8f15-726043bbb52a",
                "kgSchema": "minds/core/dataset/v1.0.0",
                "filename": "Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]",
            }
        ],
    }
