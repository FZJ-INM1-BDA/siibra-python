# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1),
# Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .. import spaces,logger
from ..bigbrain import BigBrainVolume,bbox3d
from ..atlas import Atlas
import numpy as np
from numpy.linalg import inv
from cloudvolume import Bbox
from tqdm import tqdm
from nilearn.image import math_img,resample_to_img

def get_chunk(spimg,bbox):
    """
    Reads a 3D chunk from the spatial image,
    given the bounding box object
    """
    # the bounding box needs to start inside the image
    assert(all([v>=0 for v in bbox.minpt]))
    assert(all([a<b for a,b in zip(bbox.minpt,spimg.dataobj.shape[:3])]))
    x0,y0,z0 = bbox.minpt
    x1,y1,z1 = bbox.maxpt
    return spimg.dataobj[x0:x1,y0:y1,z0:z1,:]

def chunks3d(xyzt,maxshape):
    """
    Returns a list of 3D chunks for the given 3D grid points, 
    formatted as cloudvolume bounding box objects.
    NOTE: This assumes a regular grid and infers the chunksize 
    from the distances off the first entries
    """
    chunksizes = [
        np.diff(np.unique(sorted(xyzt[i,:]))[:2])[0]
        for i in range(3) ]
    return [
        Bbox(
            np.minimum(np.maximum(xyz,0),np.array(maxshape)-1).astype('int'),
            np.minimum(xyz+chunksizes+1,maxshape).astype('int')
        ) for xyz in xyzt.T[:,:3]]


class BigBrainExtractor:
    """
    A class for convenient grayvalue extraction of BigBrain voxels via masks
    through neuroglancer precomputed data of the BigBrain template and
    parcellations.
    """
    
    def __init__(self,maskres:int=320,fullres=20):
        """
        Generate a new Extractor with a neuroglancer image volume as the basis.
        """
        self.space = spaces.BIG_BRAIN_HISTOLOGY
        self.tpl = BigBrainVolume(self.space.url)
        self.mask = None
        self.maskvolume = None
        self.mask_mip = self.tpl.determine_mip(maskres)
        self.full_mip = self.tpl.determine_mip(fullres)
        
    def apply_mask(self,atlas : Atlas):
        """
        Use the selected region of interest in the given atlas object to define a mask
        on the BigBrain Volume.
        
        There are two ways: either the region itself has mask, 
        or it has a labelindex which needs to be applied to the parcellation mask.
        """
        region = atlas.selected_region
        if hasattr(region,'maps') and (self.space.id in region.maps.keys()):
            self.__intersect_mask(region.maps[self.space.id])
        elif self.space in region.parcellation.maps.keys():
            for ngsite in region.parcellation.maps[self.space].values():
                self.__intersect_mask(ngsite,region.labelindex)
        else:
            logger.warn("Cannot instantiate a neuroglancer site from this selection. Mask is not applied.")
            return False
        return True
            
    def __intersect_mask(self,ngsite,labelindex=-1):
        
        new_volume = BigBrainVolume(ngsite)
        new_mask = new_volume.Image(self.mask_mip)

        if labelindex>0:
            logger.info("Loading mask from {} (index {})".format(
                ngsite,labelindex))
            new_mask.dataobj[new_mask.dataobj!=labelindex] = 0
            new_mask.dataobj[new_mask.dataobj>0] = 1
        else:
            logger.info("Loading binary mask from {}".format(ngsite))

        if self.mask is None:
            self.mask = new_mask
            self.maskvolume = new_volume
            logger.info("New mask defined: {} voxels ({} foreground)".format(
                np.prod(self.mask.shape), np.count_nonzero(self.mask.dataobj)))
            return

        # if there is already a mask, we compute the intersection 
        # by multiplying them in the same voxel space.
        # Of course, we choose the smaller one as the basis.
        (m0,v0),(m1,_) = sorted(
                [(self.mask,self.maskvolume),(new_mask,new_volume)],
                key = lambda x : np.prod(x[0].shape))
        m1r = resample_to_img(m1,m0,interpolation='nearest')
        self.mask = math_img("img1 * img2",img1=m0, img2=m1r)
        self.maskvolume = v0
        
        logger.info("Mask intersection: {} voxels ({} foreground)".format(
            np.prod(self.mask.shape), np.count_nonzero(self.mask.dataobj)))

    def regionstats(self,include_chunk_overview=False):
        """
        Extracts grayvalue statistics of bigbrain in the given mask at high/full resolution.
        """
          
        bbox_phys = np.dot(self.mask.affine,bbox3d(self.mask.dataobj))
        
        # get enclosing chunk grid in the full-res template space
        grid_tpl = self.tpl._enclosing_chunkgrid(self.full_mip,bbox_phys)
        chunks_tpl = chunks3d(
            grid_tpl, self.tpl.info['scales'][self.full_mip]['size'])
        grid_phys = np.dot(self.tpl.affine(self.full_mip),grid_tpl)

        # project the grid to the mask's voxel space
        grid_mask_lo = np.dot(inv(self.mask.affine),grid_phys)
        chunks_mask_lo = chunks3d(grid_mask_lo,self.mask.shape[:3])

        # Find those chunks that include foreground pixels
        fg_indices = []
        for i,bbox in enumerate(chunks_mask_lo):
            chunk = get_chunk(self.mask,bbox)
            if np.any(chunk):
                fg_indices.append(i)

        # project the grid to the mask voxel space at full resolution
        # for the final masking
        grid_mask = np.dot(inv(self.maskvolume.affine(self.full_mip,clip=False)),grid_phys)
        chunks_mask = chunks3d(grid_mask,
                self.maskvolume.info['scales'][self.full_mip]['size'])

        # Produce a histogram of the grayvalues of the template in the mask, as well as a mask of the chunks.
        overview = None
        if include_chunk_overview:
            overview = self.maskvolume.Image(self.mask_mip,clip=True)
            overview.dataobj[:,:,:] = 0
        hist = np.zeros((256))
        logger.info("Extracting {} chunks at resolution {} micron".format(
            len(fg_indices),np.min(self.tpl.volume.scales[0]['resolution']/1000))
        for i in tqdm(fg_indices):
                        
            # get full-res chunk and mask it to update histogram
            t = self.tpl.Image(self.full_mip,clip=chunks_tpl[i]).dataobj
            m = self.maskvolume.Image(self.full_mip,clip=chunks_mask[i]).dataobj
            try:
                # Note that the mask chunk could be smaller than the data
                # chunk, as it had been clipped.
                values, counts = np.unique(
                    t[:m.shape[0],:m.shape[1]][m>0],
                    return_counts=True)
            except Exception as e:
                print(t.shape,m.shape)
                raise(e)

            hist[values] += counts
            
            if overview is not None:
                # update the stamp image
                x,y,z =  chunks_mask_lo[i].minpt
                X,Y,Z =  chunks_mask_lo[i].maxpt
                overview.dataobj[x:X,y:Y,z:Z] = 1

        if include_chunk_overview:
            return hist,overview
        else:
            return hist

