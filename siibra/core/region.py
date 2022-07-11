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
from .serializable_concept import JSONSerializable
from .space import PointSet, Space, Point, BoundingBox, UnitOfMeasurement

from ..commons import (
    logger,
    Registry,
    ParcellationIndex,
    MapType,
    compare_maps,
    affine_scaling,
)
from ..retrieval.repositories import GitlabConnector
from ..openminds.SANDS.v3.atlas.parcellationEntityVersion import (
    Model as ParcellationEntityVersionModel,
    Coordinates,
    BestViewPoint,
    HasAnnotation,
)
from ..openminds.SANDS.v3.atlas.parcellationEntity import (
    Model as ParcellationEntityModel
)

import numpy as np
import nibabel as nib
from memoization import cached
import re
import anytree
from typing import List, Union
from nibabel import Nifti1Image

OPENMINDS_PARCELLATION_ENTITY_VERSION_TYPE="https://openminds.ebrains.eu/sands/ParcellationEntityVersion"

REMOVE_FROM_NAME = [
    "hemisphere",
    " -",
    # region string used in receptor features sometimes contains both/Both keywords
    # when they are present, the regions cannot be parsed properly
    "both",
    "Both",
]

REPLACE_IN_NAME = {
    "ctx-lh-": "left ",
    "ctx-rh-": "right ",
}

REGEX_TYPE = type(re.compile("test"))

THRESHOLD_CONTINUOUS_MAPS = None

class Region(anytree.NodeMixin, AtlasConcept, JSONSerializable):
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
        for search, repl in REPLACE_IN_NAME.items():
            result = result.replace(search, repl)
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
        if children:
            self.children = children
            for c in self.children:
                c.parent = self
                c.parcellation = self.parcellation

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
        Identify a region by its parcellation key, region key, and parcellation index
        """
        return hash(f"{self.parcellation.key}_{self.key}_{self.index.map}_{self.index.label}")

    def has_parent(self, parent):
        return parent in [a for a in self.ancestors]

    def includes(self, region):
        """
        Determine wether this regiontree includes the given region.
        """
        return region == self or region in self.descendants

    @cached
    def find(
        self, regionspec, filter_children=False, build_group=False, groupname=None, find_topmost=True
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
        find_topmost : Bool, default: True
            If True, will return parent structures if all children are matched, 
            even though the parent itself might not match the specification.

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
                        isinstance(regionspec, ParcellationIndex)
                        and (region.index != regionspec)
                        and any(c.index == regionspec for c in children_included)
                    ):
                        filtered.append(region)
                else:
                    if region.parent not in candidates:
                        filtered.append(region)
                    else:
                        if (
                            isinstance(regionspec, ParcellationIndex)
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

        if build_group:
            # return a single region as the result
            if len(candidates) == 1:
                return candidates[0]
            elif len(candidates) > 1:
                return Region._build_grouptree(
                    candidates, self.parcellation, name=groupname
                )
            else:
                return None
        else:
            return sorted(candidates, key=lambda r: r.depth)

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
            return [w for w in re.split(r"[^a-zA-Z0-9.]", s) if len(w) > 0]

        if isinstance(regionspec, Region):
            return self == regionspec
        elif isinstance(regionspec, int):
            # argument is int - a labelindex is expected
            return self.index.label == regionspec
        elif isinstance(regionspec, ParcellationIndex):
            if self.index.map is None:
                return self.index.label == regionspec.label
            else:
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

    @property
    def is_custom_group(self):
        """ 
        Determine wether this region object is a custom group, 
        thus not part of the actual region hierarchy. 
        """
        return (
            self not in self.parcellation 
            and all(c in self.parcellation for c in self.descendants)
        )

    @cached
    def build_mask(
        self,
        space: Space,
        resolution_mm=None,
        maptype: MapType = MapType.LABELLED,
        threshold_continuous=None,
        consider_other_types=True
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
        consider_other_types: Boolean, default: True
            If a mask for the requested maptype cannot be created, try other maptypes.
        """
        spaceobj = Space.REGISTRY[space]
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
            logger.debug(f"Merging child regions to build mask for their parent {self.name}:")
            maskdata = None
            affine = None
            for c in self.children:
                if c.extended_from is not None:
                    continue
                childmask = c.build_mask(space, resolution_mm, maptype, threshold_continuous)
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
                for other_maptype in (set(MapType) - {maptype}):
                    mask = self.build_mask(
                        space, resolution_mm, other_maptype, threshold_continuous, consider_other_types=False
                    )
                    if mask is not None:
                        logger.info(
                            f"A mask was generated from map type {other_maptype.name.lower()} instead."
                        )
                        return mask
            return None

        if (threshold_continuous is not None) and (maptype == MapType.CONTINUOUS):
            data = np.asanyarray(mask.dataobj) > threshold_continuous
            logger.info(f"Mask built using a continuous map thresholded at {threshold_continuous}.")
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
        if self.index != ParcellationIndex(None, None) and len(
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
            if (child.extended_from is None) and not child.mapped_in_space(space):
                return False
        return True

    @property
    def supported_spaces(self):
        """
        The list of spaces for which a mask could be extracted.
        Overwrites the corresponding method of AtlasConcept.
        """
        return [s for s in Space.REGISTRY if self.mapped_in_space(s)]

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
        return f"{self.name}"

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

        if not self.mapped_in_space(space):
            raise RuntimeError(
                f"Spatial properties of {self.name} cannot be computed in {space.name}. "
                "This region is only mapped in these spaces: "
                ", ".join(s.name for s in self.supported_spaces)
            )

        # build binary mask of the image
        pimg = self.build_mask(space, maptype=maptype, threshold_continuous=threshold_continuous)

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

    @classmethod
    def get_model_type(Cls):
        return OPENMINDS_PARCELLATION_ENTITY_VERSION_TYPE

    @property
    def model_id(self):
        from .. import parcellations
        if self.parcellation is parcellations.SUPERFICIAL_FIBRE_BUNDLES:
            return f"https://openminds.ebrains.eu/instances/parcellationEntityVersion/SWMA_2018_{self.name}"
        import hashlib
        def get_unique_id(id):
            return hashlib.md5(id.encode("utf-8")).hexdigest()
        # there exists several instances where same region, with same sub region exist in jba2.9
        # (e.g. ch123)
        # this is so that these regions can be distinguished from each other (ie decend from magnocellular group within septum or magnocellular group within horizontal limb of diagnoal band)
        # if not distinguished, one cannot uniquely identify the parent with parent_id
        return f"https://openminds.ebrains.eu/instances/parcellationEntityVersion/{get_unique_id(self.id + str(self.parent or 'None') + str(self.children))}"

    def to_model(self, detail=False, space: Space=None, **kwargs) -> ParcellationEntityVersionModel:
        if detail:
            assert isinstance(self.parent, JSONSerializable), f"Region.parent must be a JSONSerializable"
        if space:
            assert isinstance(space, Space), f"space kwarg must be of instance Space"
        
        pev = ParcellationEntityVersionModel(
            id=self.model_id,
            type=self.get_model_type(),
            has_parent=[{
                '@id': self.parent.model_id
            }] if (self.parent is not None) else None,
            name=self.name,
            ontology_identifier=None,
            relation_assessment=None,
            version_identifier=f"{self.parcellation.name} - {self.name}",
            version_innovation=self.descriptions[0] if hasattr(self, 'descriptions') and len(self.descriptions) > 0 else None
        )

        from .. import parcellations, spaces
        from ..volumes import VolumeSrc, NeuroglancerVolume, GiftiSurfaceLabeling
        from .datasets import EbrainsDataset

        if space is not None:
            def vol_to_id_dict(vol: VolumeSrc):
                return {
                    "@id": vol.model_id
                }
            
            """
            TODO
            It is not exactly clear, given space, if or not self.index is relevant.
            for e.g.

            ```python
            import siibra
            p = siibra.parcellations['2.9']
            
            fp1 = p.decode_region('fp1')
            fp1_left = p.decode_region('fp1 left')
            print(fp1.index) # prints (None/212)
            print(fp1_left.index) # prints (0/212)

            hoc1 = p.decode_region('hoc1')
            hoc1_left = p.decode_region('hoc1 left')
            print(hoc1.index) # prints (None/8)
            print(hoc1_left.index) # prints (0/8)
            ```

            The only way (without getting the whole map), that I can think of, is:
            
            ```python
            volumes_in_correct_space = [v for v in [*parcellation.volumes, *self.volumes] if v.space is space]
            if (
                (len(volumes_in_correct_space) == 1 and self.index.map is None)
                or (len(volumes_in_correct_space) > 1 and self.index.map is not None)
            ):
                pass # label_index is relevant
            ```

            addendum:
            In parcellations such as difumo, both nifti & neuroglancer volumes will be present.
            As a result, parc_volumes are filtered for NeuroglancerVolume.
            """

            self_volumes = [vol for vol in self.volumes if vol.space is space]
            parc_volumes = [vol for vol in self.parcellation.volumes if vol.space is space]

            vol_in_space = [v for v in [*self_volumes, *parc_volumes]
                            if isinstance(v, NeuroglancerVolume)
                            or isinstance(v, GiftiSurfaceLabeling) ]
            len_vol_in_space = len(vol_in_space)
            internal_identifier = "unknown"
            if (
                (len_vol_in_space == 1 and self.index.map is None)
                or (len_vol_in_space > 1 and self.index.map is not None)
            ):
                internal_identifier = self.index.label or "unknown"

            pev.has_annotation = HasAnnotation(
                internal_identifier=internal_identifier,
                criteria_quality_type={
                    # TODO check criteriaQualityType
                    "@id": "https://openminds.ebrains.eu/instances/criteriaQualityType/asserted"
                },
                display_color="#{0:02x}{1:02x}{2:02x}".format(*self.attrs.get('rgb')) if self.attrs.get('rgb') else None,
            )
            # seems to be the only way to convey link between PEV and dataset
            ebrains_ds = [{ "@id": "https://doi.org/{}".format(url.get("doi")) }
                for ds in self.datasets
                if isinstance(ds, EbrainsDataset)
                for url in ds.urls
                if url.get("doi")]

            try:

                # self.index.label can sometimes be None. e.g. "basal forebrain"
                # in such a case, do not populate visualized in
                if self.index.label is not None:

                    # self.index.map can sometimes be None, but label is defined
                    if self.index.map is None:

                        # In rare instances, e.g. julich brain 2.9, "Ch 123 (Basal Forebrain)"
                        # self.index.map is undefined (expect a single volume?)
                        # but there exist multiple volumes (in the example, one for left/ one for right hemisphere)
                        if len(vol_in_space) == 1:
                            pev.has_annotation.visualized_in = vol_to_id_dict(vol_in_space[0])
                    else:
                        pev.has_annotation.visualized_in = vol_to_id_dict(vol_in_space[self.index.map])
            except IndexError:
                pass
                


            # temporary workaround to https://github.com/FZJ-INM1-BDA/siibra-python/issues/185
            # in big brain jba29, without probing region.volumes, it is impossible to tell the labelindex of the region
            # adding a custom dataset, in the format of:
            # siibra_python_ng_precomputed_labelindex://{VOLUME_ID}#{LABEL_INDEX}

            # also, it appears extension regions such as MGB-MGBd (CGM, Metathalamus) do not have index defined
            # see https://github.com/FZJ-INM1-BDA/siibra-python/issues/185#issuecomment-1119317697
            BIG_BRAIN_SPACE = spaces['big brain']
            precomputed_labels = []
            if space is BIG_BRAIN_SPACE:
                big_brain_volume = [vol
                    for vol in self.volumes
                    if isinstance(vol, NeuroglancerVolume)
                    and vol.space is BIG_BRAIN_SPACE]

                precomputed_labels = [{ "@id": f"siibra_python_ng_precomputed_labelindex://{vol.model_id}#{vol.detail.get('neuroglancer/precomputed', {}).get('labelIndex')}" }
                    for vol in self.volumes
                    if isinstance(vol, NeuroglancerVolume)
                    and vol.space is BIG_BRAIN_SPACE]

                if len(big_brain_volume) == 1:
                    pev.has_annotation.visualized_in = vol_to_id_dict(big_brain_volume[0])

            pev.has_annotation.inspired_by = [
                *[vol_to_id_dict(vol) for vol in parc_volumes],
                *[vol_to_id_dict(vol) for vol in self_volumes],
                *ebrains_ds,
                *precomputed_labels,
            ]
            
            if detail:
                try:
                    centroids = self.centroids(space)
                    assert len(centroids) > 0, f"Region.to_model detailed flag set, expect a single centroid as return for centroid(space) call, but got none."
                    if len(centroids) != 1:
                        logger.warn(f"Region.to_model detailed flag set. Can only handle one and only one centroid, but got {len(centroids)}. Using the first one, if available, or return None")
                    pev.has_annotation.best_view_point = BestViewPoint(
                        coordinate_space={
                            "@id": space.model_id
                        },
                        coordinates=[Coordinates(
                            value=pt,
                            unit={
                                "@id": UnitOfMeasurement.MILLIMETER
                            }
                        ) for pt in centroids[0]]
                    )
                except AssertionError as e:
                    # no centroids found. Log warning, but do not raise.
                    logger.warn(e)
                except NotImplementedError:
                    # Region masks for surface spaces are not yet supported. for surface-based spaces
                    pass

        # per https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#pullrequestreview-872257424
        # and https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#discussion_r799479218
        # also https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#discussion_r799572025
        if self.parcellation is parcellations.SUPERFICIAL_FIBRE_BUNDLES:
            is_lh = "lh" in self.name
            is_rh = "rh" in self.name

            if is_lh:
                pev.version_identifier = f"2018, lh"
            if is_rh:
                pev.version_identifier = f"2018, rh"
            
            pev.lookup_label = f"SWMA_2018_{self.name}"


            # remove lh/rh prefix
            superstructure_name = re.sub(r"^(lh_|rh_)", "", self.name)
            # remove _[\d] suffix
            superstructure_name = re.sub(r"_\d+$", "", superstructure_name)

            superstructure_lookup_label = f"SWMA_{superstructure_name}"
            superstructure_id = f"https://openminds.ebrains.eu/instances/parcellationEntity/{superstructure_lookup_label}"

            pev.has_parent = [{
                "@id": superstructure_id
            }]
        return pev

    def to_parcellation_entities(self, **kwargs) -> List[ParcellationEntityModel]:
        import hashlib
        def get_unique_id(id):
            return hashlib.md5(id.encode("utf-8")).hexdigest()
        pe_id = f"https://openminds.ebrains.eu/instances/parcellationEntity/{get_unique_id(self.id)}"
        pe = ParcellationEntityModel(
            id=pe_id,
            type="https://openminds.ebrains.eu/sands/ParcellationEntity",
            has_parent=[{
                "@id": f"https://openminds.ebrains.eu/instances/parcellationEntity/{get_unique_id(self.parent.id)}"
            }] if self.parent else None,
            name=self.name,
            has_version=[{
                "@id": self.to_model(**kwargs).id
            }]
        )
        return_list = [pe]

        from .. import parcellations
        # per https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#pullrequestreview-872257424
        # and https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#discussion_r799479218
        # also https://github.com/HumanBrainProject/openMINDS_SANDS/pull/158#discussion_r799572025
        if self.parcellation is parcellations.SUPERFICIAL_FIBRE_BUNDLES:
            return_list = []

            is_lh = "lh" in self.name
            is_rh = "rh" in self.name
            
            if not is_lh and not is_rh:
                raise RuntimeError(f"PE for superficial bundle can only be generated for lh/rh")
            
            def get_pe_model(name:str, parent_ids:List[str]=None, has_versions_ids:List[str]=None) -> ParcellationEntityModel:
                p = ParcellationEntityModel(**pe.dict())
                p.name = name
                p.lookup_label = f"SWMA_{name}"
                p.id = f"https://openminds.ebrains.eu/instances/parcellationEntity/{p.lookup_label}"
                p.has_parent = [{ "@id": _id } for _id in parent_ids] if parent_ids else None
                p.has_version = [{ "@id": _id } for _id in has_versions_ids] if has_versions_ids else None
                return p

            # remove lh/rh prefix
            superstructure_name = re.sub(r"^(lh_|rh_)", "", self.name)
            # remove _[\d] suffix
            superstructure_name = re.sub(r"_\d+$", "", superstructure_name)
            superstructure = get_pe_model(superstructure_name, ["https://openminds.ebrains.eu/instances/parcellationEntity/SWMA_superficialFibreBundles"])

            substructure_name = re.sub(r"^(lh_|rh_)", "", self.name)
            substructure = get_pe_model(substructure_name, [superstructure.id], [self.to_model(**kwargs).id])

            return_list.append(superstructure)
            return_list.append(substructure)
            
        return return_list

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
