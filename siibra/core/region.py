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
from .space import PointSet, Space, Point, BoundingBox

from ..commons import (
    logger,
    Registry,
    ParcellationIndex,
    MapType,
    compare_maps,
    affine_scaling,
)
from ..retrieval.repositories import GitlabConnector

import numpy as np
import nibabel as nib
from memoization import cached
import re
import anytree
from typing import Union
from nibabel import Nifti1Image


REMOVE_FROM_NAME = [
    "hemisphere",
    " -",
    # region string used in receptor features sometimes contains both/Both keywords
    # when they are present, the regions cannot be parsed properly
    "both",
    "Both",
]

REGEX_TYPE=type(re.compile('test'))

class Region(anytree.NodeMixin, AtlasConcept):
    """
    Representation of a region with name and more optional attributes
    """

    CONNECTOR = GitlabConnector(
        server="https://jugit.fz-juelich.de", project=3009, reftag="master"
    )

    @staticmethod
    def _clear_name(name):
        result = name
        for word in REMOVE_FROM_NAME:
            result = result.replace(word, "")
        return " ".join(w for w in result.split(" ") if len(w))

    def __init__(
        self,
        name,
        parcellation,
        index: ParcellationIndex,
        attrs={},
        parent=None,
        children=None,
        dataset_specs=[],
    ):
        """
        Constructs a new region object.

        Parameters
        ----------
        name : str
            Human-readable name of the rgion
        parcellation : Parcellation
            the parcellation object that this region belongs to
        parcellaton : int
            the integer label index used to mark the region in a labelled brain volume
        index : ParcellationIndex
            the integer label index used to specify one of muliple available maps, if any (otherwise None)
        attrs : dict
            A dictionary of arbitrary additional information
        parent : Region
            Parent of this region, if any
        volumes : Dict of VolumeSrc
            VolumeSrc objects indexed by (Space,MapType), representing available image datasets for this region map.
        """
        regionname = __class__._clear_name(name)
        # regions are not modelled with an id yet in the configuration, so we create one here
        id = f"{parcellation.id}-{AtlasConcept._create_key((regionname+str(index))).replace('NONE','X')}"
        AtlasConcept.__init__(
            self, identifier=id, name=regionname, dataset_specs=dataset_specs
        )
        self.parcellation = parcellation
        self.index = index
        self.attrs = attrs
        self.parent = parent
        # this is only used for regions added by parcellation extension:
        self.extended_from = None
        child_has_mapindex = False
        if children:
            self.children = children
            for c in self.children:
                c.parent = self
                c.parcellation = self.parcellation
                if c.index.map is not None:
                    child_has_mapindex = True

        if (self.index.map is None) and (not child_has_mapindex):
            self.index.map = 0

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
        region._datasets_cached = []
        for d in other.datasets:
            region._datasets_cached.append(d)

        return region

    @property
    def labels(self):
        return {r.index.label for r in self if r.index.label is not None}

    @property
    def names(self):
        return Registry(elements={r.key: r.name for r in self})

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """
        Identify each region by its parcellation and region key.
        """
        return hash(self.parcellation.key + self.key)

    def has_parent(self, parent):
        return parent in [a for a in self.ancestors]

    def includes(self, region):
        """
        Determine wether this regiontree includes the given region.
        """
        return region == self or region in self.descendants

    @cached
    def find(
        self, regionspec, filter_children=False, build_group=False, groupname=None
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
            - a full ParcellationIndex
            - a region object
        filter_children : Boolean
            If true, children of matched parents will not be returned
        build_group : Boolean, default: False
            If true, the result will be a single region object, or None.
            If needed,a group region of matched elements will be created.
        groupname : str (optional)
            Name of the resulting group region, if build_group is True

        Yield
        -----
        list of matching regions if build_group==False, else Region
        """
        if isinstance(regionspec, str) and regionspec in self.names:
            # key is given, this gives us an exact region
            match = anytree.search.find_by_attr(self, regionspec, name="key")
            if match is None:
                return []
            else:
                return [match]

        result = list(
            set(anytree.search.findall(self, lambda node: node.matches(regionspec)))
        )
        if len(result) > 1 and filter_children:

            # filter regions whose parent is in the list
            filtered = [r for r in result if r.parent not in result]

            # find any non-matched regions of which all children are matched
            complete_parents = list(
                {
                    r.parent
                    for r in filtered
                    if (r.parent is not None)
                    and all((c in filtered) for c in r.parent.children)
                }
            )

            if len(complete_parents) == 0:
                result = filtered
            else:
                # filter child regions again
                filtered += complete_parents
                result = [r for r in filtered if r.parent not in filtered]

        # ensure the result is a list
        if result is None:
            result = []
        elif isinstance(result, Region):
            result = [result]
        else:
            result = list(result)

        if build_group:
            # return a single region as the result
            if len(result) == 1:
                return result[0]
            elif len(result) > 1:
                return Region._build_grouptree(
                    result, self.parcellation, name=groupname
                )
            else:
                return None
        else:
            return result

    @cached
    def matches(self, regionspec):
        """
        Checks wether this region matches the given region specification.

        Parameters
        ---------

        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - a regex applied to region names,
            - an integer, which is interpreted as a labelindex,
            - a full ParcellationIndex
            - a region object

        Yield
        -----
        True or False
        """

        def splitstr(s):
            return [w for w in re.split(r"[^a-zA-Z0-9.-]", s) if len(w) > 0]

        if isinstance(regionspec, Region):
            return self == regionspec
        elif isinstance(regionspec, int):
            # argument is int - a labelindex is expected
            return self.index.label == regionspec
        elif isinstance(regionspec, ParcellationIndex):
            return self.index == regionspec
        elif isinstance(regionspec, str):
            # string is given, perform some lazy string matching
            q = regionspec.lower().strip()
            if q == self.key.lower().strip():
                return True
            elif q == self.name.lower().strip():
                return True
            else:
                words = splitstr(self.name.lower())
                return all(
                    [
                        w.lower() in words
                        for w in splitstr(__class__._clear_name(regionspec))
                    ]
                )
        # Python 3.6 does not support re.Pattern
        elif isinstance(regionspec, REGEX_TYPE):
            # match regular expression
            return any(regionspec.search(s) is not None for s in [self.name, self.key])
        else:
            raise TypeError(
                f"Cannot interpret region specification of type '{type(regionspec)}'"
            )

    @cached
    def build_mask(
        self,
        space: Space,
        resolution_mm=None,
        maptype: MapType = MapType.LABELLED,
        threshold_continuous=None,
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
        maptype: MapType
            Type of map to build ('labelled' will result in a binary mask,
            'continuous' attempts to build a continuous mask, possibly by
            elementwise maximum of continuous maps of children )
        threshold_continuous: float, or None
            if not None, masks will be preferably constructed by thresholding
            continuous maps with the given value.
        """
        spaceobj = Space.REGISTRY[space]
        mask = affine = None
        if isinstance(maptype, str):
            maptype = MapType[maptype.upper()]

        if self.has_regional_map(spaceobj, maptype):
            mask = self.get_regional_map(space, maptype).fetch(resolution_mm=resolution_mm)
        else:
            parcmap = self.parcellation.get_map(spaceobj,  maptype)
            mask = parcmap.fetch_regionmap(self, resolution_mm=resolution_mm)

        if threshold_continuous is not None:
            assert(maptype==MapType.CONTINUOUS)
            data = np.asanyarray(mask.dataobj) > threshold_continuous
            assert(any(data)) 
            mask = nib.Nifti1Image(data.astype('uint8').squeeze(), mask.affine)

        if mask is None:
            logger.warn(f"Could not compute {maptype} mask for {self.name} in {spaceobj.name}.")

        return mask

    def defined_in_space(self, space):
        """
        Verifies wether this region is defined by a in the given space.
        """
        # the simplest case: the region has a non-empty parcellation index. Then we can assume it is mapped.
        if (
            self.index != ParcellationIndex(None, None) and
            len([v for v in self.parcellation.volumes if v.space==space])
        ):
            # Region has a non-empty parcellation index, 
            # *and* the parcellation provides a volumetric map in the requested space.
            return True

        for maptype in ["labelled", "continuous"]:
            if self.has_regional_map(space, maptype):
                return True

        return False

    @property
    def supported_spaces(self):
        """
        The list of spaces for which a mask could be extracted. 
        Overwrites the corresponding method of AtlasConcept.
        """
        return [s for s in self.parcellation.spaces if self.defined_in_space(s)]

    def has_regional_map(self, space: Space, maptype: Union[str, MapType]):
        """
        Tests wether a specific map of this region is available.
        """
        return self.get_regional_map(space, maptype) is not None

    # @cached
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
        If multiple matches are found, return the unique parent if possible,
        otherwise create an artificial parent node.
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
            else:
                # create an articicial parent region from the multiple matches
                custom_parent = Region._build_grouptree(
                    parentmatches, self.parcellation
                )
                assert custom_parent.index.label == labelindex
                logger.warning(
                    "Label index {} resolves to multiple regions. A customized region subtree is returned: {}".format(
                        labelindex, custom_parent.name
                    )
                )
                return custom_parent
        return None

    @staticmethod
    def _build_grouptree(regions, parcellation, name=None):
        """
        Creates an artificial subtree from a list of regions by adding a group
        parent and adding the regions as deep copies recursively.
        """
        # determine appropriate labelindex
        indices = []
        for tree in regions:
            indices.extend([r.index for r in tree])
        unique = set(indices)
        index = (
            next(iter(unique)) if len(unique) == 1 else ParcellationIndex(None, None)
        )

        if name is None:
            name = "Group: " + ",".join([r.name for r in regions])
        group = Region(
            name,
            parcellation,
            index,
            children=[Region.copy(r) for r in regions],
        )
        return group

    def __str__(self):
        return f"{self.parcellation.name}: {self.name}"

    def __repr__(self):
        return "\n".join(
            "%s%s" % (pre, node.name)
            if node.extended_from is None
            else "%s%s [extension region]" % (pre, node.name)
            for pre, _, node in anytree.RenderTree(self)
        )

    @cached
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
        spaceobj = Space.REGISTRY[space]
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
        spaceobj = Space.REGISTRY[space]
        pmap = self.get_regional_map(spaceobj, MapType.CONTINUOUS)
        if pmap is None:
            logger.warn(
                f"No continuous map found for {self.name} in {spaceobj.name}, "
                "cannot compute peaks."
            )
            return PointSet([], space)

        from skimage.feature.peak import peak_local_max

        img = pmap.fetch()
        dist = int(min_distance_mm / affine_scaling(img.affine) + .5)
        voxels = peak_local_max(
            img.get_fdata(),
            exclude_border=False,
            min_distance=dist,
        )
        return PointSet(
            [np.dot(img.affine, [x, y, z, 1])[:3] for x, y, z in voxels],
            space=spaceobj,
        ), img

    def centroids(self, space: Space):
        """ Compute the centroids of the region in the given space.
        
        Note that a region can generally have multiple centroids
        if it has multiple connected components in the map.
        """
        props = self.spatial_props(space)
        return [c['centroid'] for c in props['components']]

    @cached
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
            space = Space.REGISTRY[space]

        if not self.defined_in_space(space):
            raise RuntimeError(
                f"Spatial properties of {self.name} cannot be computed in {space.name}."
            )

        # build binary mask of the image
        pimg = self.build_mask(space)

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

    def print_tree(self):
        """
        Returns the hierarchy of all descendants of this region as a tree.
        """
        print(self.__repr__())

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this subtree
        (including this parent region)
        """
        return anytree.PreOrderIter(self)

    @classmethod
    def _from_json(cls, jsonstr, parcellation):
        """
        Provides an object hook for the json library to construct a Region
        object from a json definition.
        """

        # first construct any child objects
        # This is important due to the bottom-up way the tree gets
        # constructed in the Region constructor.
        children = []
        if "children" in jsonstr:
            if jsonstr["children"] is not None:
                for regiondef in jsonstr["children"]:
                    children.append(Region._from_json(regiondef, parcellation))

        # determine labelindex
        labelindex = jsonstr.get("labelIndex", None)
        if labelindex is None and len(children) > 0:
            L = [c.index.label for c in children]
            if (len(L) > 0) and (L.count(L[0]) == len(L)):
                labelindex = L[0]

        # Setup the region object
        pindex = ParcellationIndex(label=labelindex, map=jsonstr.get("mapIndex", None))
        result = cls(
            name=jsonstr["name"],
            parcellation=parcellation,
            index=pindex,
            attrs=jsonstr,
            children=children,
            dataset_specs=jsonstr.get("datasets", []),
        )

        return result


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
