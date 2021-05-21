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

from . import logger
from . import retrieval 
import os
import requests
import json
import numpy as np
from cloudvolume import CloudVolume,Bbox
from memoization import cached
import nibabel as nib

@cached
def is_ngprecomputed(url):
    # Check if the given URL is likely a neuroglancer precomputed cloud store
    try: 
        r = requests.get(url+"/info")
        info = json.loads(r.content)
        return info['type'] in ['image','segmentation']
    except Exception as _:
        return False


@cached
def load_ngprecomputed(url,resolution):
    """
    Shortcut for loading data from a neuroglancer precomputed volume directly
    into a spatial image object
    """
    V = BigBrainVolume(url)
    return V.build_image(resolution,clip=True)

def bbox3d(A):
    # https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    r = np.any(A, axis=(1, 2))
    c = np.any(A, axis=(0, 2))
    z = np.any(A, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return np.array([
        [rmin, rmax], 
        [cmin, cmax], 
        [zmin, zmax],
        [1,1]
    ])


class BigBrainVolume:
    """
    TODO use siibra requests cache
    
    """
    # function to switch x/y coordinates on a vector or matrix.
    # Note that direction doesn't matter here since the inverse is the same.
    switch_xy = lambda X : np.dot(np.identity(4)[[1,0,2,3],:],X) 

    # Gigabyte size that is considered feasible for ad-hoc downloads of
    # BigBrain data. This is used to avoid accidental huge downloads.
    gbyte_feasible = 0.5
    
    def __init__(self,ngsite,fill_missing=True):
        """
        ngsite: base url of neuroglancer http location
        """
        with requests.get(ngsite+'/transform.json') as r:
            self._translation_nm = np.array(json.loads(r.content))[:,-1]
        with requests.get(ngsite+'/info') as r:
            self.info = json.loads(r.content)
        self.volume = CloudVolume(ngsite,fill_missing=fill_missing,progress=False)
        self.ngsite = ngsite
        self.nbits = np.iinfo(self.volume.info['data_type']).bits
        self.bbox_phys = self._bbox_phys()
        self.resolutions_available = {
                np.min(v['resolution'])/1000 : {
                    'mip':i,
                    'GBytes':np.prod(v['size'])*self.nbits/(8*1024**3)
                    }
                for i,v in enumerate(self.volume.scales) }
        self.helptext = "\n".join(["{:7.0f} micron {:10.4f} GByte".format(k,v['GBytes']) 
            for k,v in self.resolutions_available.items()])

    def largest_feasible_resolution(self):
        # returns the highest resolution in micrometer that is available and
        # still below the threshold of downloadable volume sizes.
        return min([res
            for res,v in self.resolutions_available.items()
            if v['GBytes']<self.gbyte_feasible ])
        
    def affine(self,mip,clip=False):
        """
        Builds the affine matrix that maps voxels 
        at the given mip to physical space in mm.
        Parameters:
        -----------
        clip : Boolean, or Bbox
            If true, clip by computing the bounding box from nonempty pixels
            if False, get the complete data of the selected mip
            If Bbox, clip by this bounding box
        """

        # correct clipping offset, if needed
        voxelshift = np.identity(4)
        if (type(clip)==bool) and clip is True:
            voxelshift[:3,-1] = self._clipcoords(mip)[:3,0]
        elif isinstance(clip,Bbox):
            voxelshift[:3,-1] = clip.minpt

        # retrieve the pixel resolution
        resolution_nm = self.info['scales'][mip]['resolution']

        # build affine matrix in nm physical space
        affine = np.identity(4)
        for i in range(3):
            affine[i,i] = resolution_nm[i]
            affine[i,-1] = self._translation_nm[i]
            
        # warp from nm to mm   
        affine[:3,:]/=1000000.
    
        return np.dot(affine,voxelshift)
        #return BigBrainVolume.switch_xy(np.dot(affine,voxelshift))

    def _clipcoords(self,mip):
        # compute clip coordinates in voxels for the given mip 
        # from the pre-computed physical bounding box coordinates
        
        logger.debug("Computing bounding box coordinates at mip {}".format(mip))
        phys2vox = np.linalg.inv(self.affine(mip))
        clipcoords = np.dot(phys2vox,self.bbox_phys).astype('int')
        # clip bounding box coordinates to actual shape of the mip
        clipcoords[:,0] = np.maximum(clipcoords[:,0],0)
        clipcoords[:,1] = np.minimum(clipcoords[:,1],self.volume.mip_shape(mip))
        return clipcoords

    def _load_data(self,mip,clip=False,force=False):
        """
        Actually load image data.
        TODO: Check amount of data beforehand and raise an Exception if it is over a reasonable threshold.
        NOTE: this function caches chunks as numpy arrays (*.npy) to the
        CACHEDIR defined in the retrieval module.
        
        Parameters:
        -----------
        clip : Boolean, or Bbox
            If true, clip by computing the bounding box from nonempty pixels
            if False, get the complete data of the selected mip
            If Bbox, clip by this bounding box
        force : Boolean (default: False)
            if true, will start downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable.
        """
        if (type(clip)==bool) and clip is True:
            clipcoords = self._clipcoords(mip)
            bbox = Bbox(clipcoords[:3,0],clipcoords[:3,1])
        elif isinstance(clip,Bbox):
            # make sure the bounding box is integer, some are not
            bbox = Bbox(
                    np.array(clip.minpt).astype('int'),
                    np.array(clip.maxpt).astype('int'))
        else:
            bbox = Bbox([0, 0, 0],self.volume.mip_shape(mip))
        gbytes = bbox.volume()*self.nbits/(8*1024**3)
        if not force and gbytes>BigBrainVolume.gbyte_feasible:
            # TODO would better do an estimate of the acutal data size
            logger.error("Data request is too large (would result in an ~{:.2f} GByte download, the limit is {}).".format(gbytes,self.gbyte_feasible))
            print(self.helptext)
            raise RuntimeError("The requested resolution is too high to provide a feasible download, but you can override this behavior with the 'force' parameter.")
        cachefile = retrieval.cachefile("{}{}{}".format(
            self.ngsite, bbox.serialize(), str(mip)).encode('utf8'),suffix='npy')
        if os.path.exists(cachefile):
            return np.load(cachefile)
        else:
            data = self.volume.download(bbox=bbox,mip=mip)
            np.save(cachefile,np.array(data))
            return np.array(data)

    def determine_mip(self,resolution=None):
        # given a resolution in micrometer, try to determine the mip that can
        # be used to move on.
        if resolution is None:
            maxres = self.largest_feasible_resolution()
            logger.info('Using the largest feasible resolution of {} micron'.format(maxres))
            return self.resolutions_available[maxres]['mip']
        elif resolution in self.resolutions_available.keys(): 
            return self.resolutions_available[resolution]['mip']
        logger.error('The requested resolution ({} micron) is not available. Choose one of:'.format(resolution))
        print(self.helptext)
        return None
        
    def build_image(self,resolution,clip=True,transform=lambda I: I, force=False):
        """
        Compute and return a spatial image for the given mip.
        
        Parameters:
        -----------
        clip : Boolean, or Bbox
            If true, clip by computing the bounding box from nonempty pixels
            if False, get the complete data of the selected mip
            If Bbox, clip by this bounding box
        force : Boolean (default: False)
            If true, will start downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable.
        """
        mip = self.determine_mip(resolution)
        if not mip:
            raise ValueError("Invalid image resolution for this neuroglancer precomputed tile source.")
        return nib.Nifti1Image(
            transform(self._load_data(mip,clip,force)),
            affine=self.affine(mip,clip)
        )
    
    def _enclosing_chunkgrid(self, mip, bbox_phys):
        """
        Produce grid points representing the chunks of the mip 
        which enclose a given bounding box. The bounding box is given in
        physical coordinates, but the grid is returned in voxel spaces of the
        given mip.
        """
        
        # some helperfunctions to produce the smallest range on a grid enclosing another range 
        cfloor = lambda x,s: int(np.floor(x/s)*s)
        cceil = lambda x,s: int(np.ceil(x/s)*s)+1
        crange = lambda x0,x1,s: np.arange(cfloor(x0,s),cceil(x1,s),s)
        
        # project the bounding box to the voxel grid of the selected mip
        bb = np.dot(np.linalg.inv(self.affine(mip)),bbox_phys)
        
        # compute the enclosing chunk grid
        chunksizes = self.volume.scales[mip]['chunk_sizes'][0]
        x,y,z = [crange(bb[i][0],bb[i][1],chunksizes[i]) 
                 for i in range(3)]
        xx,yy,zz = np.meshgrid(x,y,z)
        return np.vstack([xx.ravel(),yy.ravel(),zz.ravel(),zz.ravel()*0+1])

    def _bbox_phys(self):
        """
        Estimates the bounding box of the nonzero values 
        in the data volume, in physical coordinates. 
        Estimation is done from the lowest resolution for 
        efficiency, so it is not fully accurate.
        """
        volume = self._load_data(-1,clip=False)
        affine = self.affine(-1,clip=False)
        bbox_vox = bbox3d(volume)
        return np.dot(affine,bbox_vox)



