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

from . import logger
from . import retrieval 
import numpy as np
import nibabel as nib
from cloudvolume.exceptions import OutOfBoundsError
from cloudvolume import CloudVolume,Bbox
import os
import json

def from_json(obj):
    """
    Provides an object hook for the json library to construct a VolumeSrc
    object from a json stream.
    """
    if obj.get('@type')!="fzj/tmp/volume_type/v0.0.1":
        raise NotImplementedError(f"Cannot build VolumeSrc from this json spec: {obj}")

    volume_type=obj.get('volume_type')
    detail = obj.get('detail')
    url = obj.get('url')

    if volume_type=='nii':
        return NiftiVolume( obj['@id'], obj['name'],url=url,
                detail=detail,zipped_file=obj.get('zipped_file') )

    if volume_type=='neuroglancer/precomputed':
        transform_nm = np.identity(4)
        vsrc_labelindex = None
        if detail is not None and 'neuroglancer/precomputed' in detail:
            if 'transform' in detail['neuroglancer/precomputed']:
                transform_nm = np.array(detail['neuroglancer/precomputed']['transform'])
            if 'labelIndex' in detail['neuroglancer/precomputed']:
                vsrc_labelindex = int(detail['neuroglancer/precomputed']['labelIndex'])
        return NgVolume( obj['@id'], obj['name'],url=url,
                detail=detail, transform_nm=transform_nm)
        #img = volume.build_image(resolution_mm=resolution_mm)
        #if vsrc_labelindex is not None:
        #    # this volume src has a local labelindex - we apply it
        #    img.dataobj[img.dataobj!=vsrc_labelindex] = 0
        #return img

    if volume_type=='detailed maps':
        vsrc = VolumeSrc(obj['@id'], obj['name'],url=None,detail=detail)
        vsrc.volume_type=volume_type
        return vsrc

    if volume_type=='neuroglancer/precompmesh':
        return VolumeSrc( obj['@id'], obj['name'],url=url,
                detail=detail)

    
    return obj

class VolumeSrc:

    def __init__(self, identifier, name, url, detail=None ):
        """
        Construct a new volume source.

        Parameters
        ----------
        identifier : str
            A unique identifier for the source
        name : str
            A human-readable name
        volume_type : str
            Type of volume source, clarifying the data format. Typical names: "nii", "neuroglancer/precomputed".
            TODO create a fixed list (enum?) of supported types, or better: derive types from VolumeSrc object
        url : str
            The URL to the volume src, typically a url to the corresponding image or tilesource.
        detail : dict
            Detailed information. Currently only used to store a transformation matrix  for neuroglancer tilesources.
        zipped_file : str
            The filename to be extracted from a zip file. If given, the url is
            expected to point to a downloadable zip archive. Currently used to
            extreact niftis from zip archives, as for example in case of the
            MNI reference templates.
        """
        self.id = identifier
        self.name = name
        self.url = url
        if 'SIIBRA_URL_MODS' in os.environ and url:
            mods = json.loads(os.environ['SIIBRA_URL_MODS'])
            for old,new in mods.items():
                self.url = self.url.replace(old,new)
            if self.url!=url:
                logger.warning(f'Applied URL modification\nfrom {url}\nto   {self.url}') 
        self.volume_type = None
        self.detail = {} if detail is None else detail
        self.space = None

    def __str__(self):
        return f'{self.volume_type} {self.url}'

    def get_url(self):
        return self.url

    def fetch(self,resolution_mm=None):
        raise NotImplementedError("Fetching image data is not available for pure VolumeSrc objects, choose a specific derived class like NiftiVolume instead.")


class NiftiVolume(VolumeSrc):

    def __init__(self,identifier, name, url, detail=None, zipped_file=None):
        super().__init__(identifier, name, url, detail=detail)
        self.zipped_file = zipped_file
        self.volume_type = 'nii'

    def fetch(self,resolution_mm=None):
        """
        Loads and returns a Nifti1Image object representing the volume source.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for neuroglancer volumes.
        """
        filename = retrieval.download_file(self.url,ziptarget=self.zipped_file)
        if filename:
            return nib.load(filename)
        else:
            return None


class NgVolume(VolumeSrc):

    # Gigabyte size that is considered feasible for ad-hoc downloads of
    # neuroglancer volume data. This is used to avoid accidental huge downloads.
    gbyte_feasible = 0.5
    _cached_volume = None
    
    def __init__(self,identifier,name,url,detail=None,transform_nm=np.identity(4)):
        """
        ngsite: base url of neuroglancer http location
        transform_nm: optional transform to be applied after scaling voxels to nm
        """
        super().__init__(identifier,name,url,detail)
        logger.debug(f'Constructing NgVolume "{name}"')
        self.volume_type = "neuroglancer/precomputed"
        self.transform_nm = transform_nm
        r = retrieval.cached_get(url+'/info')
        self.info = json.loads(r.decode())
        self.url = url
        self.nbytes = np.dtype(self.info['data_type']).itemsize
        self.num_scales = len(self.info['scales'])
        self.mip_resolution_mm = { i:np.min(v['resolution'])/(1000**2)
                for i,v in enumerate(self.info['scales']) }
        self.resolutions_available = {
                np.min(v['resolution'])/(1000**2) : {
                    'mip':i,
                    'GBytes':np.prod(v['size'])*self.nbytes/(1024**3)
                    }
                for i,v in enumerate(self.info['scales']) }
        self.helptext = "\n".join(["{:7.4f} mm {:10.4f} GByte".format(k,v['GBytes']) 
            for k,v in self.resolutions_available.items()])

    @property
    def volume(self):
        """
        We implement this as a property so that the CloudVolume constructor is only called lazily.
        """
        if not self._cached_volume:
            self._cached_volume = CloudVolume(self.url,fill_missing=True,progress=False)
        return self._cached_volume
    
    def largest_feasible_resolution(self):
        # returns the highest resolution in millimeter that is available and
        # still below the threshold of downloadable volume sizes.
        return min([res
            for res,v in self.resolutions_available.items()
            if v['GBytes']<self.gbyte_feasible ])

    def _resolution_to_mip(self,resolution_mm):
        """
        Given a resolution in millimeter, try to determine the mip that can
        be applied.

        Parameters
        ----------
        resolution_mm : float or None
            Physical resolution in mm.
            If None, the smallest availalbe resolution is used (lowest image size)
            If -1, tha largest feasible resolution is used.
        """
        mip = None
        if resolution_mm is None:
            mip = self.num_scales-1
        elif resolution_mm==-1:
            maxres = self.largest_feasible_resolution()
            mip = self.resolutions_available[maxres]['mip']
            logger.info(f'Using largest feasible resolution of {maxres} mm (mip={mip})')
        elif resolution_mm in self.resolutions_available:
            mip = self.resolutions_available[resolution_mm]['mip']
        if mip is None:
            raise ValueError(f'Requested resolution of {resolution_mm} mm not available.\n{self.helptext}')
        return mip
            
    def build_affine(self,resolution_mm=None):
        """
        Builds the affine matrix that maps voxels 
        at the given resolution to physical space.

        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm. 
            If None, the smallest is used.
            If -1, the largest feasible is used.
        """
        mip = self._resolution_to_mip(resolution_mm)

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

    def _load_data(self,resolution_mm=None):
        """
        Actually load image data.
        TODO: Check amount of data beforehand and raise an Exception if it is over a reasonable threshold.
        NOTE: this function caches chunks as numpy arrays (*.npy) to the
        CACHEDIR defined in the retrieval module.
        
        Parameters:
        -----------
        resolution_mm : float, or None
            desired resolution in mm. If none, the full resolution is used.
        """
        mip = self._resolution_to_mip(resolution_mm)

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
        gbytes = bbox_vox.volume()*self.nbytes/(1024**3)
        if gbytes>NgVolume.gbyte_feasible:
            # TODO would better do an estimate of the acutal data size
            logger.error("Data request is too large (would result in an ~{:.2f} GByte download, the limit is {}).".format(gbytes,self.gbyte_feasible))
            print(self.helptext)
            raise NotImplementedError(f"Request of the whole full-resolution volume in one piece is prohibited as of now due to the estimated size of ~{gbytes:.0f} GByte.")

        # ok, retrieve data no now.
        cachefile = retrieval.cachefile("{}{}{}".format(
            self.url, bbox_vox.serialize(), str(mip)).encode('utf8'),suffix='npy')
        if os.path.exists(cachefile):
            return np.load(cachefile)
        else:
            try:
                data = self.volume.download(bbox=bbox_vox,mip=mip)
                np.save(cachefile,np.array(data))
                return np.array(data)
            except OutOfBoundsError as e:
                logger.error('Bounding box does not match image.')
                print(str(e))
                return None
        
    def fetch(self,resolution_mm=None):
        """
        Compute and return a spatial image for the given mip.
        
        Parameters:
        -----------
        resolution_mm : desired resolution in mm
        """
        return nib.Nifti1Image(
            self._load_data(resolution_mm),
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


