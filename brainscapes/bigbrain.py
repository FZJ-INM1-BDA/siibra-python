from . import logger,spaces
from nibabel.spatialimages import SpatialImage
import requests
import json
import numpy as np
from numpy.linalg import inv
from cloudvolume import CloudVolume,Bbox
from memoization import cached
from tqdm import tqdm
from nilearn.image import math_img,resample_to_img

@cached
def is_ngprecomputed(url):
    # Check if the given URL is likely a neuroglancer precomputed cloud store
    r = requests.get(url+"/info")
    return r.status_code==200

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
            (np.minimum(xyz+chunksizes+1,maxshape)+.5).astype('int')
        ) for xyz in xyzt.T[:,:3]]

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
    TODO use brainscapes requests cache
    
    """
    # function to switch x/y coordinates on a vector or matrix.
    # Note that direction doesn't matter here since the inverse is the same.
    switch_xy = lambda X : np.dot(np.identity(4)[[1,0,2,3],:],X) 

    # Gigabyte size that is considered feasible for ad-hoc downloads of
    # BigBrain data. This is used to avoid accidental huge downloads.
    gbyte_feasible = 0.5
    
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
            bbox = clip
        else:
            bbox = Bbox([0, 0, 0],self.volume.mip_shape(mip))
        gbytes = bbox.volume()*self.nbits/(8*1024**3)
        if not force and gbytes>BigBrainVolume.gbyte_feasible:
            # TODO would better do an estimate of the acutal data size
            logger.error("Data request is too large (would result in an ~{:.2f} GByte download, the limit is {}).".format(gbytes,self.gbyte_feasible))
            print(self.helptext)
            raise RuntimeError("Requested resolution is to high to provide a feasible download, but you can override this behavior with the 'force' parameter.")
        data = self.volume.download(bbox=bbox,mip=mip)
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
        logger.error('The requested resolution is not available. Choose one of:')
        print(self.helptext)
        return None
        
    def Image(self,mip,clip=True,transform=lambda I: I, force=False):
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
        assert(type(mip)==int)
        return SpatialImage(
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



class BigBrainExtractor:
    """
    A class for convenient grayvalue extraction of BigBrain voxels via masks
    through neuroglancer precomputed data of the BigBrain template and
    parcellations.
    """
    
    def __init__(self,maskres:int=320,fullres=160):
        """
        Generate a new Extractor with a neuroglancer image volume as the basis.
        """
        self.space = spaces.BIG_BRAIN_HISTOLOGY
        self.tpl = BigBrainVolume(self.space.url)
        self.masks = []
        self.mask_mip = self.tpl.determine_mip(maskres)
        self.full_mip = self.tpl.determine_mip(fullres)
        
    def set_mask(self,atlas):
        """
        Use the selected region of interest in the atlas define the mask volume.
        
        There are two ways: either the region itself has mask, 
        or it has a labelindex which needs to be applied to the parcellation mask.
        """
        region = atlas.selected_region
        if hasattr(region,'maps') and (self.space.id in region.maps.keys()):
            self.__add_mask(region.maps[self.space.id])
        elif self.space in region.parcellation.maps.keys():
            for ngsite in region.parcellation.maps[self.space].values():
                self.__add_mask(ngsite,region.labelindex)
        else:
            print("Cannot instantiate a neuroglancer site from this selection")
            return
            
    def __add_mask(self,ngsite,labelindex=-1):
        
        print("Adding mask from ng site",ngsite,"with index",labelindex)
        
        self.masks.append(BigBrainVolume(ngsite))
        self.masks[-1].labelindex=labelindex
        
    def __mask_intersection(self):
        
        mask = None
        vol = None
        
        for i,vol_i in enumerate(self.masks):
            
            mask_i = vol_i.Image(self.mask_mip)
            if vol_i.labelindex>0:
                mask_i.dataobj[mask_i.dataobj!=vol_i.labelindex] = 0
                mask_i.dataobj[mask_i.dataobj>0] = 1
            if mask is None:
                mask,vol = mask_i,vol_i
            else:
                # if there is already a mask, we compute the intersection 
                # by multiplying them in the same voxel space.
                # Of course, we choose the smaller one as the basis.
                (m0,v0),(m1,_) = sorted([(mask,vol),(mask_i,vol_i)],
                        key = lambda x : np.prod(x[0].shape))
                m1r = resample_to_img(m1,m0,interpolation='nearest')
                mask = math_img("img1 * img2",img1=m0, img2=m1r)
                vol = v0
                
            print("Mask intersection: #{}: {} voxels ({} foreground)".format(
                i,np.prod(mask.shape), np.count_nonzero(mask.dataobj)))
            
        return (mask,vol)
        
    def regionstats(self):
        """
        Extracts grayvalue statistics of bigbrain in the given mask at high/full resolution.
        """
          
        mask_lo,maskvol = self.__mask_intersection()
        bbox_phys = np.dot(mask_lo.affine,bbox3d(mask_lo.dataobj))
        
        # get enclosing chunk grid in the full-res template space
        grid_tpl = self.tpl._enclosing_chunkgrid(self.full_mip,bbox_phys)
        chunks_tpl = chunks3d(
            grid_tpl, self.tpl.info['scales'][self.full_mip]['size'])
        grid_phys = np.dot(self.tpl.affine(self.full_mip),grid_tpl)

        # project the grid to the mask's voxel space
        grid_mask_lo = np.dot(inv(mask_lo.affine),grid_phys)
        chunks_mask_lo = chunks3d(grid_mask_lo,mask_lo.shape[:3])

        # Find those chunks that include foreground pixels
        fg_indices = []
        for i,bbox in enumerate(chunks_mask_lo):
            chunk = get_chunk(mask_lo,bbox)
            if np.any(chunk):
                fg_indices.append(i)

        # project the grid to the mask voxel space at full resolution
        # for the final masking
        grid_mask = np.dot(inv(maskvol.affine(self.full_mip,clip=False)),grid_phys)
        chunks_mask = chunks3d(grid_mask,maskvol.info['scales'][self.full_mip]['size'])

        # Produce a histogram of the grayvalues of the template in the mask, as well as a mask of the chunks.
        roimask = maskvol.Image(self.mask_mip,clip=True)
        roimask.dataobj[:,:,:] = 0
        hist = np.zeros((256))
        print("Extracting {} full-resolution chunks".format(len(fg_indices)))
        for i in tqdm(fg_indices):
                        
            # get full-res chunk and mask it to update histogram
            t = self.tpl.Image(self.full_mip,clip=chunks_tpl[i]).dataobj
            m = maskvol.Image(self.full_mip,clip=chunks_mask[i]).dataobj
            values, counts = np.unique(
                t[:m.shape[0],:m.shape[1]][m>0],
                return_counts=True)
            hist[values] += counts
            
            # update the stamp image
            x,y,z =  chunks_mask_lo[i].minpt
            X,Y,Z =  chunks_mask_lo[i].maxpt
            roimask.dataobj[x:X,y:Y,z:Z] = 1

        return hist,roimask

