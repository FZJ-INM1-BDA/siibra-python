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
from .location import PointSet, Point, BoundingBox

from ..commons import (
    logger,
    MapIndex,
    MapType,
    compare_maps,
    affine_scaling,
    create_key,
    clear_name,
)
from ..registry import REGISTRY
from ..retrieval.repositories import GitlabConnector

import numpy as np
import nibabel as nib
import re
import anytree
from typing import List, Union
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
        children: list["Region"] = [],
        parent: "Region" = None,
        shortname: str = "",
        description: str = "",
        modality: str = "",
        publications: list = [],
        ebrains_ids: dict = {},
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
        ebrains_ids : dict
            Identifiers of EBRAINS entities corresponding to this Parcellation.
            Key: EBRAINS KG schema, value: EBRAINS KG @id
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
            ebrains_ids=ebrains_ids,
        )

        # anytree node will take care to use this appropriately
        self.parent = parent
        self.children = children
        # convert hex to int tuple if rgb is given
        self.rgb = (
            None if rgb is None
            else tuple(int(rgb[p:p + 2], 16) for p in [1, 3, 5])
        )

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

        return self.__hash__() == other.__hash__()

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
        if isinstance(regionspec, str) and regionspec in self.names:
            # key is given, this gives us an exact region
            match = anytree.search.find_by_attr(self, regionspec, name="key")
            return [] if match is None else [match]

        candidates = list(
            set(anytree.search.findall(self, lambda node: node.matches(regionspec)))
        )

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

        # Python 3.6 does not support re.Pattern
        elif isinstance(regionspec, REGEX_TYPE):
            # match regular expression
            return any(regionspec.search(s) is not None for s in [self.name, self.key])
        else:
            raise TypeError(
                f"Cannot interpret region specification of type '{type(regionspec)}'"
            )

    def build_mask(
        self,
        space: Space,
        resolution_mm=None,
        maptype: Union[str, MapType] = MapType.LABELLED,
        threshold_continuous=None,
        consider_other_types=True,
    ):
        """
        Returns a mask where nonzero values denote
        voxels corresponding to the region.

        Parameters
        ----------
        space : Space
            The desired template space.
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm.
            If None, the native resolution is used.
            Currently, this only works for the BigBrain volume.
        maptype: Union[str, MapType]
            Type of map to build ('labelled' will result in a binary mask,
            'continuous' attempts to build a continuous mask, possibly by
            elementwise maximum of continuous maps of children )
        threshold_continuous: float, or None
            if not None, masks will be preferably constructed by thresholding
            continuous maps with the given value.
        consider_other_types: Boolean, default: True
            If a mask for the requested maptype cannot be created, try other maptypes.
        """
        spaceobj = REGISTRY.Space[space]
        if spaceobj.is_surface:
            raise NotImplementedError(
                "Region masks for surface spaces are not yet supported."
            )

        mask = None
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]

        if self.has_regional_map(spaceobj, maptype):
            # the region has itself a map of that type linked
            mask = self.get_regional_map(space, maptype).fetch(
                resolution_mm=resolution_mm
            )
        else:
            # retrieve  map of that type from the region's corresponding parcellation map
            parcmap = self.parcellation.get_map(spaceobj, maptype)
            mask = parcmap.fetch_regionmap(self, resolution_mm=resolution_mm)

        if mask is None:
            # Attempt to produce a map from the child regions.
            # We only accept this if all child regions produce valid masks.
            # NOTE We ignore extension regions here, since this is a complex case currently (e.g. iam regions in BigBrain)
            logger.debug(
                f"Merging child regions to build mask for their parent {self.name}:"
            )
            maskdata = None
            affine = None
            for c in self.children:
                childmask = c.build_mask(
                    space, resolution_mm, maptype, threshold_continuous
                )
                if childmask is None:
                    logger.warning(f"No success getting mask for child {c.name}")
                    break
                if maskdata is None:
                    affine = childmask.affine
                    maskdata = np.asanyarray(childmask.dataobj)
                else:
                    assert (childmask.affine == affine).all()
                    maskdata = np.maximum(maskdata, np.asanyarray(childmask.dataobj))
            else:
                # we get here only if the for loop was not interrupted by 'break'
                if maskdata is not None:
                    return Nifti1Image(maskdata, affine)

        if mask is None:
            # No map of the requested type found for the region.
            logger.warn(
                f"Could not compute {maptype.name.lower()} mask for {self.name} in {spaceobj.name}."
            )
            if consider_other_types:
                for other_maptype in set(MapType) - {maptype}:
                    mask = self.build_mask(
                        space,
                        resolution_mm,
                        other_maptype,
                        threshold_continuous,
                        consider_other_types=False,
                    )
                    if mask is not None:
                        logger.info(
                            f"A mask was generated from map type {other_maptype.name.lower()} instead."
                        )
                        return mask
            return None

        if (threshold_continuous is not None) and (maptype == MapType.CONTINUOUS):
            data = np.asanyarray(mask.dataobj) > threshold_continuous
            logger.info(
                f"Mask built using a continuous map thresholded at {threshold_continuous}."
            )
            # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
            assert np.any(data > 0)
            mask = nib.Nifti1Image(data.astype("uint8").squeeze(), mask.affine)

        return mask

    def mapped_in_space(self, space):
        """
        Verifies wether this region is defined by an explicit map in the given space.
        """
        # the simple case: the region has a non-empty parcellation index,
        # and its parcellation has a volumetric map in the requested space.
        if self.index != MapIndex(None, None) and len(
            [v for v in self.parcellation.volumes if v.space == space]
        ):
            return True

        # Some regions have explicit regional maps
        for maptype in ["labelled", "continuous"]:
            if self.has_regional_map(space, maptype):
                return True

        # The last option is that this region has children,
        # and all of them are mapped in the requested space.
        if self.is_leaf:
            return False

        for child in self.children:
            if not child.mapped_in_space(space):
                return False
        return True

    @property
    def supported_spaces(self) -> List[Space]:
        """
        The list of spaces for which a mask could be extracted.
        Overwrites the corresponding method of AtlasConcept.
        """
        return [s for s in REGISTRY.Space if self.mapped_in_space(s)]

    def has_regional_map(self, space: Space, maptype: Union[str, MapType]):
        """
        Tests wether a specific map of this region is available.
        """
        return self.get_regional_map(space, maptype) is not None

    def get_regional_map(self, space: Space, maptype: Union[str, MapType]):
        """
        Retrieves and returns a specific map of this region, if available
        (otherwise None). This is typically a probability or otherwise
        continuous map, as opposed to the standard label mask from the discrete
        parcellation.

        Parameters
        ----------
        space : Space, or str
            Specifier for the template space
        maptype : MapType
            Type of map (e.g. continuous, labelled - see commons.MapType)
        """
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]

        def maptype_ok(vsrc, maptype):
            if vsrc.map_type is None:
                return (maptype == MapType.CONTINUOUS) == (vsrc.is_float())
            else:
                return vsrc.map_type == maptype

        available = self.get_volumes(space)

        suitable = [v for v in available if maptype_ok(v, maptype)]
        if len(suitable) == 1:
            return suitable[0]
        elif len(suitable) == 0:
            return None
        else:
            raise NotImplementedError(
                f"Multiple regional maps found for {self} in {space}. This case is not expected."
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
        spaceobj = REGISTRY.Space[space]
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
        spaceobj = REGISTRY.Space[space]
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
            space = REGISTRY.Space[space]

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
