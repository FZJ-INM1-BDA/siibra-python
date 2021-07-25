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

from .feature import RegionalFeature
from .extractor import FeatureExtractor
from .. import logger
from ..retrieval import cached_gitlab_query,cached_get

class LazyLoader:
    def __init__(self,url,decfnc=lambda x:x):
        """
        Initialize a lazy data loader. It gets a URL and decding function, 
        and will only retrieve the data when its data property is accessed.

        Parameters
        ----------
        url : string
            URL for loading the data
        decfnc : function pointer
            function to be applied for decoding the retrieved data
        """
        self.url = url
        self._data_cached = None
        self._decfnc = decfnc

    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = self._decfnc(
                cached_get(self.url))
        return self._data_cached

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

        # construct lazy data loaders
        self._info_loader = LazyLoader(
            urlscheme.format(file="info.txt"),
            lambda b:dict(l.split(' ') for l in b.decode('utf8').strip().split("\n")) )
        self._segments_loader = LazyLoader(
            urlscheme.format(file="segments.txt"),
            lambda b:np.loadtxt(b,skiprows=1) )
        self._image_loader = LazyLoader(
            urlscheme.format(file="image.nii.gz"),
            lambda b:nib.Nifti1Image.from_bytes(gzip.decompress(b)) )
        self._layerinfo_loader = LazyLoader(
            urlscheme.format(file="layerinfo.txt"),
            lambda b:np.loadtxt(b,skiprows=1) )

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

    def __str__(self):
        return ",".join(f"{k}:{v}" for (k,v) in self.info.items())

   
class RegionalCellDensityExtractor(FeatureExtractor):

    _FEATURETYPE = CorticalCellDistribution
    URLSCHEME="https://{server}/s/{share}/download?path=%2F{area}%2F{section}%2F{patch}&files={file}"
    SERVER="fz-juelich.sciebo.de"
    SHARE="yDZfhxlXj6YW7KO"

    def __init__(self):
        FeatureExtractor.__init__(self)
        server = "https://jugit.fz-juelich.de"
        projectid = 4790
        reftag = 'v1.0a1'
        query=lambda folder,fname:cached_gitlab_query(
            server,projectid,reftag,folder,fname,skip_branchtest=True)

        # determine available region subfolders in the dataset
        #project = Gitlab(gitlab_server).projects.get(4790)
        tree = json.loads(query(None,None))
        region_folders = [ e['name'] for e in tree
                        if e['type']=="tree" 
                        and not e['name'].startswith('.') ]

        for region_folder in region_folders:
            regionspec = " ".join(region_folder.split('_')[1:])
            logger.debug(f"Found cell density data for region spec {regionspec}.")
            tree = json.loads(query(region_folder,None))
            section_ids = [e['name'] for e in tree if e['type']=="tree"]
            for section_id in section_ids:
                section_folder = f"{region_folder}/{section_id}"
                tree = json.loads(query(section_folder,None))
                patch_ids = [e['name'] for e in tree if e['type']=="tree"]
                for patch_id in patch_ids:
                    urlscheme = self.URLSCHEME.format(
                        server=self.SERVER, share=self.SHARE,
                        area=region_folder, section=section_id, patch=patch_id, file="{file}")
                    self.register(CorticalCellDistribution(regionspec,urlscheme))


if __name__ == '__main__':

    pass
