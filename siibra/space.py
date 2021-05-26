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

from .commons import create_key
from .config import ConfigurationRegistry
import numpy as np
from . import volumesrc
from . import logger
import copy
from cloudvolume import Bbox
from typing import Tuple

class Space:
    """
    A particular brain reference space.
    """

    def __init__(self, identifier, name, template_type=None, src_volume_type=None, volume_src={}):
        self.id = identifier
        self._rename(name)
        self.type = template_type
        self.src_volume_type = src_volume_type
        self.volume_src = volume_src
        self._assign_volume_sources(volume_src)

    def _assign_volume_sources(self,volume_src):
        self.volume_src = copy.deepcopy(volume_src)
        for volsrc in self.volume_src:
            try:
                volsrc.space = self
            except AttributeError as e:
                print(volsrc)
                raise(e)

    def _rename(self,newname):
        self.name = newname
        self.key = create_key(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def get_template(self, resolution_mm=None ):
        """
        Get the volumetric reference template image for this space.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        candidates = [vsrc for vsrc in self.volume_src if vsrc.volume_type==self.type]
        if not len(candidates)==1:
            raise RuntimeError(f"Could not resolve template image for {self.name}. This is most probably due to a misconfiguration of the volume src.")
        return candidates[0]

    def __getitem__(self,slices:Tuple[slice,slice,slice]):
        """
        Get a volume of interest specification from this space.

        Arguments
        ---------
        slices: triple of slice
            defines the x, y and z range
        """
        if len(slices)!=3:
            raise TypeError("Slice access to spaces needs to define x,y and z ranges (e.g. Space[10:30,0:10,200:300])")
        return SpaceVOI(self,[s.start for s in slices],[s.stop for s in slices])

    def get_voi(self,minpt:Tuple[float,float,float],maxpt:Tuple[float,float,float]):
        """
        Get a rectangular volume of interest specification for this space.

        Arguments
        ---------

        minpt: 3-tuple
            smaller 3D point defining the VOI
        maxpt: 3-tuple
            larger 3D point defining the VOI
        """
        return self[
            minpt[0]:maxpt[0],
            minpt[1]:maxpt[1],
            minpt[2]:maxpt[2]]

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a Space
        object from a json stream.
        """
        required_keys = ['@id','name','shortName','templateType']
        if any([k not in obj for k in required_keys]):
            return obj

        if "minds/core/referencespace/v1.0.0" not in obj['@id']:
            return obj

        volume_src = [volumesrc.from_json(v) for v in obj['volumeSrc']] if 'volumeSrc' in obj else []
        return Space(obj['@id'], obj['shortName'], template_type = obj['templateType'],
                src_volume_type = obj.get('srcVolumeType'),
                volume_src = volume_src)


class SpaceVOI(Bbox):

    def __init__(self,space:Space,minpt:Tuple[float,float,float],maxpt:Tuple[float,float,float]):
        super().__init__(minpt,maxpt)
        self.space = space

    def transform_bbox(self,transform):
        assert(transform.shape==(4,4))
        return Bbox(
            np.dot(transform,np.r_[self.minpt,1])[:3].astype('int'),
            np.dot(transform,np.r_[self.maxpt,1])[:3].astype('int') )

REGISTRY = ConfigurationRegistry('spaces', Space)
