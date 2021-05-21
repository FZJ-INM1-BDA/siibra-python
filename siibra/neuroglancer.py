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

from cloudvolume.exceptions import OutOfBoundsError
from . import logger
from . import retrieval 
import os
import requests
import json
import numpy as np
from cloudvolume import CloudVolume,Bbox
import nibabel as nib

def bbox3d(A):
    """
    Bounding box of nonzero values in a 3D array.
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
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

class NgVolume:

    # function to switch x/y coordinates on a vector or matrix.
    # Note that direction doesn't matter here since the inverse is the same.
    switch_xy = lambda X : np.dot(np.identity(4)[[1,0,2,3],:],X) 

    # Gigabyte size that is considered feasible for ad-hoc downloads of
    # neuroglancer volume data. This is used to avoid accidental huge downloads.
    gbyte_feasible = 0.5
    
    def __init__(self,ngsite,transform_nm=np.identity(4),fill_missing=True):
        """
        ngsite: base url of neuroglancer http location
        transform_nm: optional transform to be applied after scaling voxels to nm
        """
        logger.debug(f'Constructing NgVolume with physical transform:\n{transform_nm}')
        self.transform_nm = transform_nm
        with requests.get(ngsite+'/info') as r:
            self.info = json.loads(r.content)
        self.volume = CloudVolume(ngsite,fill_missing=fill_missing,progress=False)
        self.ngsite = ngsite
        self.nbits = np.iinfo(self.volume.info['data_type']).bits
        self.mip_resolution_mm = { i:np.min(v['resolution'])/(1000**2)
                for i,v in enumerate(self.volume.scales) }
        self.resolutions_available = {
                np.min(v['resolution'])/(1000**2) : {
                    'mip':i,
                    'GBytes':np.prod(v['size'])*self.nbits/(8*1024**3)
                    }
                for i,v in enumerate(self.volume.scales) }
        self.helptext = "\n".join(["{:7.4f} mm {:10.4f} GByte".format(k,v['GBytes']) 
            for k,v in self.resolutions_available.items()])
        logger.debug(f"Available resolutions for volume:\n{self.helptext}")

    def largest_feasible_resolution(self):
        # returns the highest resolution in millimeter that is available and
        # still below the threshold of downloadable volume sizes.
        return min([res
            for res,v in self.resolutions_available.items()
            if v['GBytes']<self.gbyte_feasible ])
        
    def build_affine(self,resolution_mm=None):
        """
        Builds the affine matrix that maps voxels 
        at the given resolution to physical space.

        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm. If None, the full resolution is used.
        """
        if resolution_mm is None:
            mip = self.determine_mip(self.largest_feasible_resolution())
        else:
            mip = self.determine_mip(resolution_mm)
        if mip is None:
            raise ValueError(f"Invalid resolution of {resolution_mm} specified")

        # scaling from voxel to nm
        resolution_nm = self.info['scales'][mip]['resolution']
        scaling = np.identity(4)
        for i in range(3):
            scaling[i,i] = resolution_nm[i]

        # optional transform in nanometer space
        affine = np.dot(self.transform_nm,scaling)
            
        # warp from nm to mm   
        affine[:3,:]/=1000000.

        # if a volume of interest is given, apply the offset
        #shift_mm = np.identity(4)
        #if voi:
        #    shift_mm[:3,-1] = voi.bbox_mm.minpt
    
        #return np.dot(shift_mm,affine)
        return affine

    def _load_data(self,resolution_mm=None,force=False):
        """
        Actually load image data.
        TODO: Check amount of data beforehand and raise an Exception if it is over a reasonable threshold.
        NOTE: this function caches chunks as numpy arrays (*.npy) to the
        CACHEDIR defined in the retrieval module.
        
        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm. If none, the full resolution is used.
    
        force : Boolean (default: False)
            if true, will start downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable.
        """

        if resolution_mm is None:
            mip = self.determine_mip(self.largest_feasible_resolution())
        else:
            mip = self.determine_mip(resolution_mm)
        if mip is None:
            raise ValueError(f"Invalid resolution of {resolution_mm} specified")

        # apply voi
        #if voi:
        #    mm2vox = np.linalg.inv(self.build_affine(resolution_mm))
        #    bbox_vox = Bbox(
        #            np.dot(mm2vox,np.r_[voi.bbox_mm.minpt,1])[:3].astype('int'),
        #            np.dot(mm2vox,np.r_[voi.bbox_mm.maxpt,1])[:3].astype('int') )
        #    print("Bounding box in voxels:",bbox_vox)
        #else:
        bbox_vox = Bbox([0, 0, 0],self.volume.mip_shape(mip))

        # estimate size and check feasibility
        gbytes = bbox_vox.volume()*self.nbits/(8*1024**3)
        if not force and gbytes>NgVolume.gbyte_feasible:
            # TODO would better do an estimate of the acutal data size
            logger.error("Data request is too large (would result in an ~{:.2f} GByte download, the limit is {}).".format(gbytes,self.gbyte_feasible))
            print(self.helptext)
            raise RuntimeError("The requested resolution is too high to provide a feasible download, but you can override this behavior with the 'force' parameter.")

        # ok, retrieve data no now.
        cachefile = retrieval.cachefile("{}{}{}".format(
            self.ngsite, bbox_vox.serialize(), str(mip)).encode('utf8'),suffix='npy')
        if os.path.exists(cachefile):
            return np.load(cachefile)
        else:
            print(f"Downloading image data. mip={mip}, bbox={bbox_vox}")
            try:
                data = self.volume.download(bbox=bbox_vox,mip=mip)
                np.save(cachefile,np.array(data))
                return np.array(data)
            except OutOfBoundsError as e:
                logger.error('Bounding box does not match image.')
                print(str(e))
                return None

    def determine_mip(self,resolution_mm=None):
        # given a resolution in millimeter, try to determine the mip that can
        # be used to move on.
        if resolution_mm is None:
            maxres = self.largest_feasible_resolution()
            mip = self.resolutions_available[maxres]['mip']
            logger.info(f'Using largest feasible resolution of {maxres} mm (mip={mip})')
        elif resolution_mm in self.resolutions_available:
            mip = self.resolutions_available[resolution_mm]['mip']
        else:
            print(self.helptext)
            raise ValueError(f'Requested resolution of {resolution_mm} mm not available.')
            mip = None
        return mip
        
    def build_image(self,resolution_mm=None,transform=lambda I: I, force=False):
        """
        Compute and return a spatial image for the given mip.
        
        Parameters:
        -----------
        resolution_mm : desired resolution in mm
        force : Boolean (default: False)
            If true, will start downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable.
        """
        return nib.Nifti1Image(
            transform(self._load_data(resolution_mm,force)),
            affine=self.build_affine(resolution_mm)
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
        bb = np.dot(np.linalg.inv(self.build_affine(self.mip_resolution_mm[mip])),bbox_phys)
        
        # compute the enclosing chunk grid
        chunksizes = self.volume.scales[mip]['chunk_sizes'][0]
        x,y,z = [crange(bb[i][0],bb[i][1],chunksizes[i]) 
                 for i in range(3)]
        xx,yy,zz = np.meshgrid(x,y,z)
        return np.vstack([xx.ravel(),yy.ravel(),zz.ravel(),zz.ravel()*0+1])


