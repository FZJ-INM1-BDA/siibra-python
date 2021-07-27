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

import numpy as np
from gitlab import Gitlab
import gzip
import nibabel as nib
import json

from .feature import RegionalFeature,SpatialFeature
from .query import FeatureQuery
from .. import logger,spaces
from ..retrieval import cached_gitlab_query,cached_get,LazyLoader


class CorticalCellDistribution(RegionalFeature):
    """
    Represents a cortical cell distribution dataset. 
    Implements lazy and cached loading of actual data. 
    """

    def __init__(self, region, urlscheme):
        """
        Parameters
        ----------
        region : string
            specification of brain region
        urlscheme : format string 
            with formatting field 'file' for downloading individual files for this datasets
        """
        RegionalFeature.__init__(self,region)
        self._urlscheme = urlscheme

        # construct lazy data loaders
        self._info_loader = LazyLoader(
            urlscheme.format(file="info.txt"),
            lambda b:dict(l.split(' ') for l in b.decode('utf8').strip().split("\n")) )
        self._segments_loader = LazyLoader(
            urlscheme.format(file="segments.txt"),
            lambda b:np.array([
                [float(w) for w in l.strip().split(' ')]
                for l in b.decode().strip().split('\n')[1:]]) )
        self._image_loader = LazyLoader(
            urlscheme.format(file="image.nii.gz"),
            lambda b:nib.Nifti1Image.from_bytes(gzip.decompress(b)) )
        self._layerinfo_loader = LazyLoader(
            urlscheme.format(file="layerinfo.txt"),
            lambda b:np.array([
                tuple(l.split(' ')[1:])
                for l in b.decode().strip().split('\n')[1:]],
                dtype=[
                    ('layer','U10'),
                    ('area (micron^2)','f'),
                    ('avg. thickness (micron)','f')
                ]) )
        self._segmentation_loaders = [
            LazyLoader(
                urlscheme.format(file="image.nii.gz"),
                lambda b:nib.Nifti1Image.from_bytes(gzip.decompress(b)) )
        ]

    def load_segmentations(self):
        from PIL import Image
        from io import BytesIO
        index = 0
        result = []
        logger.info(f'Loading cell instance segmentation masks for {self}...')
        while True:
            try:
                target = f'segments_cpn_{index:02d}.tif'
                bytestream = cached_get(self._urlscheme.format(file=target))
                result.append(np.array(Image.open(BytesIO(bytestream))))
                index+=1
            except RuntimeError:
                break
        return result

    @property
    def info(self):
        return self._info_loader.data

    @property
    def cells(self):
        """
        Nx5 array with attributes of segmented cells:
            x  y area(micron^2)  layer instance_label
        """
        return self._segments_loader.data
    
    @property 
    def layers(self):
        """
        6x4 array of cortical layer attributes: 
            Number Name Area(micron**2) AvgThickness(micron)
        """
        return self._layerinfo_loader.data

    @property
    def image(self):
        """
        Nifti1Image representation of the original image patch, 
        with an affine matching it to the histological BigBrain space.
        """
        return self._image_loader.data

    @property
    def coordinate(self):
        """
        Coordinate of this image patch in BigBrain histological space in mm.
        """
        A = self.image.affine
        return spaces.BIG_BRAIN,np.dot(A,[0,0,0,1])[:3]

    def __str__(self):
        return f"BigBrain cortical cell distribution in {self.regionspec} (section {self.info['section_id']}, patch {self.info['patch_id']})"

   
class RegionalCellDensityExtractor(FeatureQuery):

    _FEATURETYPE = CorticalCellDistribution
    URLSCHEME="https://{server}/s/{share}/download?path=%2F{area}%2F{section}%2F{patch}&files={file}"
    SERVER="fz-juelich.sciebo.de"
    SHARE="yDZfhxlXj6YW7KO"

    def __init__(self):
        FeatureQuery.__init__(self)
        server = "https://jugit.fz-juelich.de"
        projectid = 4790
        reftag = 'v1.0a1'

        # determine available region subfolders in the dataset
        fulltree = json.loads(cached_gitlab_query(
            server,projectid,reftag,None,None,skip_branchtest=True,recursive=True))
        cellfiles = [ e['path'] for e in fulltree
                        if e['type']=="blob" 
                        and e['name']=='segments.txt' ]

        for cellfile in cellfiles:
            region_folder,section_id,patch_id,_ = cellfile.split("/")
            regionspec = " ".join(region_folder.split('_')[1:])
            urlscheme = self.URLSCHEME.format(
                server=self.SERVER, share=self.SHARE,
                area=region_folder, section=section_id, patch=patch_id, file="{file}")
            self.register(CorticalCellDistribution(regionspec,urlscheme))

if __name__ == '__main__':

    pass
