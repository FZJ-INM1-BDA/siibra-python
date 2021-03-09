from nibabel.spatialimages import SpatialImage
import requests
import json
import numpy as np
from cloudvolume import CloudVolume,Bbox
import numpy as np
from brainscapes import logger

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
            np.maximum(xyz,0).astype('int'),
            (np.minimum(xyz+chunksizes+1,maxshape)+.5).astype('int')
        ) for xyz in xyzt.T[:,:3]]

def get_chunk(spimg,bbox):
    """
    Reads a 3D chunk from the spatial image,
    given the bounding box object
    """
    x0,y0,z0 = bbox.minpt
    x1,y1,z1 = bbox.maxpt
    return spimg.dataobj[x0:x1,y0:y1,z0:z1,:]


class BigBrainVolume:
    """
    TODO: Consider deriving the class from SpatialImage
    """
    # function to switch x/y coordinates on a vector or matrix.
    # Note that direction doesn't matter here since the inverse is the same.
    switch_xy = lambda X : np.dot(np.identity(4)[[1,0,2,3],:],X) 
    max_gbytes = 0.5
    
    
    def __init__(self,ngsite):
        """
        ngsite: base url of neuroglancer http location
        """        
        with requests.get(ngsite+'/transform.json') as r:
            self._translation_nm = np.array(json.loads(r.content))[:,-1]
        with requests.get(ngsite+'/info') as r:
            self.info = json.loads(r.content)
        self.volume = CloudVolume(ngsite,progress=False)
        self.nbits = np.iinfo(self.volume.info['data_type']).bits
        self.bbox_phys = self._bbox_phys()
        self.resolutions_available = {
                np.min(v['resolution'])/1000 : {
                    'mip':i,
                    'GBytes':np.prod(v['size'])*self.nbits/(8*1024**3)
                    }
                for i,v in enumerate(self.volume.scales) }

    def largest_feasible_resolution(self):
        # returns the highest resolution in micrometer that is available and
        # still below the threshold of downloadable volume sizes.
        return min([res
            for res,v in self.resolutions_available.items()
            if v['GBytes']<self.max_gbytes ])
        
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
    
        return BigBrainVolume.switch_xy(np.dot(affine,voxelshift))

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

    def _load_data(self,mip,clip=False):
        """
        Actually load image data.
        TODO: Check amount of data beforehand and raise an Exception if it is over a reasonable threshold.
        
        Parameters:
        -----------
        clip : Boolean, or Bbox
            If true, clip by computing the bounding box from nonempty pixels
            if False, get the complete data of the selected mip
            If Bbox, clip by this bounding box
        """
        if (type(clip)==bool) and clip is True:
            clipcoords = self._clipcoords(mip)
            bbox = Bbox(clipcoords[:3,0],clipcoords[:3,1])
        elif isinstance(clip,Bbox):
            bbox = clip
        else:
            bbox = Bbox([0, 0, 0],self.volume.mip_shape(mip))
        gbytes = bbox.volume()*self.nbits/(8*1024**3)
        if gbytes>BigBrainVolume.max_gbytes:
            # TODO would better do an estimate of the acutal data size
            raise Exception("Data request is too large (would result in an ~{:.2f} GByte download, the limit is {})".format(gbytes,self.max_gbytes))
        data = self.volume.download(bbox=bbox,mip=mip)
        return np.array(data)
        
    def Image(self,mip,clip=True,transform=lambda I: I):
        """
        Compute and return a spatial image for the given mip.
        
        Parameters:
        -----------
        clip : Boolean, or Bbox
            If true, clip by computing the bounding box from nonempty pixels
            if False, get the complete data of the selected mip
            If Bbox, clip by this bounding box
        """
        return SpatialImage(
            transform(self._load_data(mip,clip)),
            affine=self.affine(mip,clip)
        )
    
    def _enclosing_chunkgrid(self, mip, bbox_phys):
        # produce grid points representing the chunks of the mip, 
        # which enclose a bounding box.
        # project the bounding box of the mask to the template voxel space
        
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
        
        volume = self._load_data(-1,clip=False)
        bbox_vox = bbox3d(volume)
        affine = self.affine(-1,clip=False)
        return np.dot(affine,bbox_vox)
