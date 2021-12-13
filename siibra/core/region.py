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

from .concept import AtlasConcept, RegistrySrc, main_openminds_registry, provide_openminds_registry
from .space import PointSet, Space, Point, BoundingBox

from ..commons import (
    logger,
    ParcellationIndex,
    MapType,
    compare_maps,
    affine_scaling,
)

import numpy as np
import nibabel as nib
from memoization import cached
import re
import anytree
from anytree import NodeMixin
from typing import Any, Dict, Generic, List, Tuple, TypeVar, Union
from nibabel import Nifti1Image
from ..openminds.SANDS.v3.atlas import parcellationEntityVersion, parcellationEntity
from ..openminds.common import CommonConfig

REMOVE_FROM_NAME = [
    "hemisphere",
    " -",
    # region string used in receptor features sometimes contains both/Both keywords
    # when they are present, the regions cannot be parsed properly
    "both",
    "Both",
]

REGEX_TYPE=type(re.compile('test'))

class MixedNode(NodeMixin):

    # ref can also be a parcellation
    ref = None
    def __init__(self, ref):
        self.ref = ref

class SiibraNode:
    def __init__(self):
        # if this line raises ValueError: object has no field "_node"
        # ensure the class has _node defined under class
        self._node = MixedNode(self)

    @property
    def children(self) -> List:
        return [c.ref for c in self._node.children]

    @property
    def parent(self):
        return self._node.parent and self._node.parent.ref

    # do not use attribute setter
    # see https://github.com/samuelcolvin/pydantic/issues/1577
    def set_parent(self, value):
        self._node.parent = value and value._node

    def find(self, *arg, **kwargs) -> List['SiibraNode']:
        """
        Proxy to [anytree.search.findall](https://anytree.readthedocs.io/en/latest/_modules/anytree/search.html)

        filter_: callable
        
        """
        if len(arg) > 0 and type(arg[0]) == str:
            region_spec = arg[0]
            kwargs['filter_'] = lambda node: node.ref.matches(region_spec)
            arg = arg[1:]

        return [mixed_node.ref for mixed_node in anytree.search.findall(
            self._node,
            *arg,
            **kwargs
        )]

class Region(parcellationEntityVersion.Model, AtlasConcept, SiibraNode):
    """
    Representation of a region with name and more optional attributes
    """

    @staticmethod
    def _clear_name(name):
        result = name
        for word in REMOVE_FROM_NAME:
            result = result.replace(word, "")
        return " ".join(w for w in result.split(" ") if len(w))

    Config = CommonConfig

    _parcellation = None
    _parcellation_id = None
    _node = None

    @property
    def index(self) -> str:
        return self.lookup_label

    @property
    def _parcellation(self):
        try:
            return main_openminds_registry[self._parcellation_id] if self._parcellation_id else None
        except IndexError as e:
            logger.warning(f"cannot find parcellation with parc id {self._parcellation_id}")
            return None

    def __init__(
        self,
        parcellation_id=None,
        attrs={},
        parent=None,
        children=None,
        dataset_specs=[],
        **data,
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
        
        # regions are not modelled with an id yet in the configuration, so we create one here
        name = data.get('name')
        index = data.get('labelIndex')
        regionname = __class__._clear_name(name)
        

        parcellationEntityVersion.Model.__init__(self,**data)
        AtlasConcept.__init__(
            self, identifier=self.id, name=regionname
        )
        SiibraNode.__init__(self)
        self.set_parent(parent)

        self._parcellation_id = parcellation_id

        # TODO check if these properties are used anywhere

        # self.attrs = attrs
        # # this is only used for regions added by parcellation extension:
        # self.extended_from = None
        # child_has_mapindex = False

        
    @staticmethod
    def copy(other: 'Region'):
        """
        copy contructor must detach the parent to avoid problems with
        the Anytree implementation.
        """
        # create an isolated object, detached from the other's tree
        region = Region(**other.dict())

        # Build the new subtree recursively
        region.children = tuple(Region.copy(c) for c in other.children)
        for c in region.children:
            c.parent = region
        region._datasets_cached = []
        # for d in other.datasets:
        #     region._datasets_cached.append(d)

        return region

    # TODO fix
    # @property
    # def labels(self):
    #     return {r.index.label for r in self if r.index.label is not None}

    # TODO fix
    @property
    def names(self):
        return [self.name]
        # return Registry(elements={r.key: r.name for r in self})

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """
        Identify each region by its parcellation and region key.
        """
        return hash((str(self._parcellation.__hash__()) if self._parcellation else f'UNKNOWN_PARC:{self._parcellation_id}') + self.name)

    # TODO breaking change
    # has_parent is an existing attribute for model
    # method must be renamed has_parent -> has_node_parent
    def has_node_parent(self, parent: 'Region'):
        return parent._node in [a for a in self._node.ancestors]

    def includes(self, region: 'Region'):
        """
        Determine wether this regiontree includes the given region.
        """
        return region._node is self._node or region._node in self._node.descendants

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

        if AtlasConcept.matches(self,regionspec):
            return True

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
            
            # TODO
            # what is self.key?

            # if q == self.key.lower().strip():
            #     return True
            if q == self.name.lower().strip():
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
            return any(regionspec.search(s) is not None for s in [self.name])
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
        return f"{self._parcellation.full_name if self._parcellation else f'unknown parc - {self._parcellation_id}'}: {self.name}"

    def __repr__(self):
        return "\n".join(
            "%s%s" % (pre, node.name)
            if hasattr(node, 'extended_from') and node.extended_from is None
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
    def parse_legacy(Cls, json_input: Dict[str, Any], parcellation_id=None, parent:'Region'=None) -> List['Region']:
        regionname = Cls._clear_name(json_input.get('name'))
        idx=json_input.get('labelIndex')
        
        id = f"{parcellation_id}-{AtlasConcept._create_key((regionname+str(idx))).replace('NONE','X')}"
        has_annotation=None
        has_parent=None
        this_region = Cls(
            id=id,
            name=regionname,
            type='https://openminds.ebrains.eu/sands/ParcellationEntityVersion',
            has_annotation=has_annotation,
            lookup_label=str(idx) if idx else None,
            parcellation_id=parcellation_id,
            version_identifier='12',
            parent=parent,
        )

        # TODO: parse _datasrc properly
        children = [child
            for c in json_input.get('children', [])
            for child in Cls.parse_legacy(
                json_input=c,
                parcellation_id=parcellation_id,
                parent=this_region,
            )
        ]

        return [ this_region, *children ]


@provide_openminds_registry(registry_src=RegistrySrc.EMPTY)
class VersionlessRegion(
    parcellationEntity.Model,
    SiibraNode
):
    def __init__(self, **kwarg):
        super().__init__(**kwarg)

    # TODO this method is still flawed.
    # need to check no duplicates are added
    @classmethod
    def add_region(Cls, region: Region):
        if not region.name:
            return
        if ('left' in region.name) or ('right' in region.name):
            return

        def get_lookup_label(name: str):
            if name == 'cingulate gyrus, frontal part':
                return 'frontal-cingulate-gyrus'
            import re
            processed_name = name
            # rule, replace all brackets (and preceding white space)
            processed_name =  re.sub(r'\s+\(.*?\)$', '', processed_name)
            # rule, replace all ,s with _
            processed_name = re.sub(r',', '_', processed_name)
            # rule, replace all one or more continuous white space characters with -
            processed_name = re.sub(r'\s+', '-', processed_name)
            # rule: remove all 's entirely
            processed_name = re.sub(r'\'', '', processed_name)
            return 'JBA_{}'.format(processed_name)

        def get_id(name: str):
            return 'https://openminds.ebrains.eu/instances/parcellationEntity/' + get_lookup_label(name)

        def get_has_version_from_existing(name:str):
            import requests
            url = 'https://raw.githubusercontent.com/HumanBrainProject/openMINDS_SANDS/v3/instances/atlas/parcellationEntity/JBA/{lookuplabel}.jsonld'.format(
                lookuplabel=get_lookup_label(name)
            )
            response = requests.get(url)
            if response.status_code >= 400:
                logger.warning('cannot find existing entry for {name}, with url: {url}'.format(
                    name=name,
                    url=url))
                return None
            return response.json().get('hasVersion', [])
        
        parent_id = region.parent and hasattr(region.parent, 'name') and get_id(region.parent.name)
        has_version = get_has_version_from_existing(region.name) if all([
            ('left' in c.name) or ('right' in c.name)
            for c in region.children or []
        ]) else None
        Cls.REGISTRY.add(
            region.name,
            Cls(
                id=get_id(region.name),
                type='https://openminds.ebrains.eu/sands/ParcellationEntity',
                has_parent=[{ '@id': parent_id }] if parent_id else None,
                has_version=has_version,
                lookup_label=get_lookup_label(region.name),
                name=region.name
            )
        )
    Config = CommonConfig


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
