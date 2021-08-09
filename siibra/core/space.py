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

from .core import SemanticConcept

from ..retrieval import HttpRequest
from ..arrays import bbox3d

import numpy as np
from cloudvolume import Bbox
from typing import Tuple
import nibabel as nib
from urllib.parse import quote
import json

@SemanticConcept.provide_registry
class Space(SemanticConcept,bootstrap_folder="spaces",type_id="minds/core/referencespace/v1.0.0"):
    """
    A particular brain reference space.
    """

    def __init__(self, identifier, name, template_type=None, src_volume_type=None, dataset_specs=[]):
        SemanticConcept.__init__(self,identifier,name,dataset_specs)
        self.src_volume_type=src_volume_type
        self.type = template_type

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

    @classmethod
    def _from_json(cls,obj):
        """
        Provides an object hook for the json library to construct a Space
        object from a json stream.
        """
        required_keys = ['@id','name','shortName','templateType']
        if any([k not in obj for k in required_keys]):
            return obj
        if "minds/core/referencespace/v1.0.0" not in obj['@id']:
            return obj

        result = cls(
            identifier = obj['@id'], 
            name = obj['shortName'], 
            template_type = obj['templateType'],
            src_volume_type = obj.get('srcVolumeType'),
            dataset_specs=obj.get('datasets',[]) )

        return result


class SpaceVOI(Bbox):

    def __init__(self,space:Space,minpt:Tuple[float,float,float],maxpt:Tuple[float,float,float]):
        assert(len(minpt)==3 and len(maxpt)==3)
        super().__init__(minpt,maxpt)
        self.space = space

    @staticmethod
    def from_map(space:Space,roi:nib.Nifti1Image):
        # construct from a roi mask or map
        bbox = bbox3d(roi.dataobj,affine=roi.affine)
        return SpaceVOI(space,bbox[:3,0],bbox[:3,1])

    def overlaps(self,img):
        """
        Determines wether the given image overlaps with this volume of interest, 
        that is, wheter at least one nonzero voxel is inside the voi.
        """
        # nonzero voxel coordinates
        X,Y,Z = np.where(img.get_fdata()>0)
        h = np.ones(len(X))
        # array of homogenous physcial coordinates
        coords = np.dot(img.affine,np.vstack((X,Y,Z,h)))[:3,:].T
        minpt = [min(self.minpt[i],self.maxpt[i]) for i in range(3)]
        maxpt = [max(self.minpt[i],self.maxpt[i]) for i in range(3)]
        inside = np.logical_and.reduce([coords>minpt,coords<=maxpt]).min(1)
        return any(inside)        

    def transform_bbox(self,transform):
        assert(transform.shape==(4,4))
        return Bbox(
            np.dot(transform,np.r_[self.minpt,1])[:3].astype('int'),
            np.dot(transform,np.r_[self.maxpt,1])[:3].astype('int') )

    def __iter__(self):
        """
        Iterator over the min- and maxpt of the volume of interest.
        """
        return (p for p in [self.minpt,self.maxpt])

    def __str__(self):
        return f"Bounding box {self.minpt}mm -> {self.maxpt}mm defined in {self.space.name}"

    def __repr__(self):
        return str(self)


class SpaceWarper():
    
    SPACE_IDS = {
        Space.REGISTRY.MNI152_2009C_NONL_ASYM : "MNI 152 ICBM 2009c Nonlinear Asymmetric",
        Space.REGISTRY.MNI_COLIN_27 : "MNI Colin 27",
        Space.REGISTRY.BIG_BRAIN : "Big Brain (Histology)"
    }

    SERVER="https://hbp-spatial-backend.apps.hbp.eu/v1"

    @staticmethod
    def convert(from_space,to_space,coord):
        if any (s not in SpaceWarper.SPACE_IDS for s in [from_space,to_space]):
            raise ValueError(f"Cannot convert coordinates between {from_space} and {to_space}")
        url='{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}'.format(
            server=SpaceWarper.SERVER,
            src=quote(SpaceWarper.SPACE_IDS[Space.REGISTRY[from_space]]),
            tgt=quote(SpaceWarper.SPACE_IDS[Space.REGISTRY[to_space]]),
            x=coord[0], y=coord[1], z=coord[2] )
        response = HttpRequest(url,lambda b:json.loads(b.decode())).get()
        return tuple(response["target_point"])
