# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import logger, spaces, volumesrc, parcellationmap
from .space import Space
from .region import Region
from .config import ConfigurationRegistry
from .commons import create_key,MapType
import numpy as np
import nibabel as nib
from nilearn import image
from tqdm import tqdm
from memoization import cached

class ParcellationVersion:
    def __init__(self, name=None, prev_id=None, next_id=None):
        self.name=name
        self.next_id=next_id
        self.prev_id=prev_id
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return  self.name

    def __iter__(self):
        yield 'name', self.name
        yield 'prev', self.prev.id if self.prev is not None else None
        yield 'next', self.next.id if self.next is not None else None

    @property
    def next(self):
        if self.next_id is None:
            return None
        try:
            return REGISTRY[self.next_id]
        except IndexError:
            return None
        except NameError:
            logger.warning('Accessing REGISTRY before its declaration!')
    
    @property
    def prev(self):
        if self.prev_id is None:
            return None
        try:
            return REGISTRY[self.prev_id]
        except IndexError:
            return None
        except NameError:
            logger.warning('Accessing REGISTRY before its declaration!')

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a
        ParcellationVersion object from a json string.
        """
        if obj is None:
            return None
        return ParcellationVersion(obj.get('name', None), prev_id=obj.get('@prev', None), next_id=obj.get('@next', None))

class Parcellation:

    _regiontree_cached = None

    def __init__(self, identifier : str, name : str, version=None, modality=None, regiondefs=[]):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the parcellation
        name : str
            Human-readable name of the parcellation
        version : str or None
            a version specification, optional
        modality  :  str or None
            a specification of the modality used for creating the parcellation.
        """
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.version = version
        self.publications = []
        self.description = ""
        self.volume_src = {}
        self.modality = modality
        self._regiondefs = regiondefs

        # If set, thresholded continuous maps will be preferred
        # over static labelled maps for building and using region masks.
        # This will influence the shape of region masks used for filtering.
        self.continuous_map_threshold = None

    @property
    def regiontree(self):
        if self._regiontree_cached is None:
            self._regiontree_cached = Region(self.name,self)
            try:
                    self._regiontree_cached.children = tuple( 
                        Region.from_json(regiondef,self) 
                        for regiondef in self._regiondefs )
            except Exception as e:
                logger.error(f"Could not generate child regions for {self.name}")
                raise(e)
        return self._regiontree_cached

    def get_volume_src(self, space: Space):
        """
        Get volumes sources for the parcellation in the requested template space.

        Parameters
        ----------
        space : Space
            template space

        Yields
        ------
        A list of volume sources
        """
        if not self.supports_space(space):
            raise ValueError('Parcellation "{}" does not provide volume sources for space "{}"'.format(
                str(self), str(space) ))
        return self.volume_src[space]

    @cached
    def get_map(self, space: Space=None, maptype:MapType=MapType.LABELLED, resolution_mm=None):
        """
        Get the volumetric maps for the parcellation in the requested
        template space. This might in general include multiple 
        3D volumes. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere. Per default, multiple maps are concatenated into a 4D
        array, but you can choose to retrieve a dict of 3D volumes instead using `return_dict=True`.

        Parameters
        ----------
        space : Space
            template space 
        maptype : MapType (default: MapType.LABELLED)
            Type of map requested (e.g., continous or labelled, see commons.MapType)
            Use MapType.CONTINUOUS to request probability maps.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen.

        Yields
        ------
        A ParcellationMap representing the volumetric map.
        """
        if space is None:
            space = next(iter(self.volume_src.keys()))
            if len(self.volume_src)>1:
                logger.warning(f'Parcellation "{str(self)}" provides maps in multiple spaces, but no space was specified.\nUsing the first, "{str(space)}"')

        if not self.supports_space(space):
            raise ValueError('Parcellation "{}" does not provide a map for space "{}"'.format(
                str(self), str(space) ))

        return parcellationmap.create_map(self,space,maptype)

    @property
    def labels(self):
        return self.regiontree.labels

    @property
    def names(self):
        return self.regiontree.names

    def supports_space(self,space : Space):
        """
        Return true if this parcellation supports the given space, else False.
        """
        return space in self.volume_src.keys()

    def decode_region(self,regionspec,mapindex=None):
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
        mapindex : integer, or None (optional)
            Some parcellation maps are defined over multiple 3D parcellation
            volumes with overlapping labelindices (e.g. splitting the
            hemispheres). For those, the optional mapindex can be used to 
            further restrict the matching regions.

        Return
        ------
        Region object
        """
        candidates = self.regiontree.find(regionspec,select_uppermost=True,mapindex=mapindex)
        if not candidates:
            raise ValueError("Regionspec {} could not be decoded under '{}'".format(
                regionspec,self.name))
        elif len(candidates)==1:
            return candidates[0]
        else:
            return Region._build_grouptree(candidates,self)


    def find_regions(self,regionspec):
        """
        Find regions with the given specification in this parcellation.

        Parameters
        ----------
        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex
            - a region object

        Yield
        -----
        list of matching regions
        """
        return self.regiontree.find(regionspec)


    def __str__(self):
        return self.name

    def __repr__(self):
        return  self.name

    def __eq__(self,other):
        """
        Compare this parcellation with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other,Parcellation):
            return self.id==other.id
        elif isinstance(other,str):
            return any([
                self.name==other, 
                self.key==other,
                self.id==other])
        else:
            raise ValueError("Cannot compare object of type {} to Parcellation".format(type(other)))

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this parcellation
        """
        return self.regiontree.__iter__()

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a Parcellation
        object from a json string.
        """
        required_keys = ['@id','name','shortName','volumeSrc','regions']
        if any([k not in obj for k in required_keys]):
            return obj

        # create the parcellation, it will create a parent region node for the regiontree.
        p = Parcellation(obj['@id'], obj['shortName'], regiondefs=obj['regions'])
        
        if 'volumeSrc' in obj:
            p.volume_src = { spaces[space_id] : {
                key : [
                    volumesrc.from_json(v_src) for v_src in v_srcs
                ] for key, v_srcs in key_vsrcs.items()
            } for space_id, key_vsrcs in obj['volumeSrc'].items() }
        
        if '@version' in obj:
            p.version = ParcellationVersion.from_json(obj['@version'])

        if 'modality' in obj:
            p.modality = obj['modality']

        if 'description' in obj:
            p.description = obj['description']
        if 'publications' in obj:
            p.publications = obj['publications']
        logger.debug(f'Adding parcellation "{str(p)}"')
        return p

REGISTRY = ConfigurationRegistry('parcellations', Parcellation)
