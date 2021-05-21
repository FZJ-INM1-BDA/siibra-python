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
import numpy as np
from .retrieval import download_file
from .neuroglancer import NgVolume
import nibabel as nib

from os import environ
import json

class VolumeSrc:

    def __init__(self, identifier, name, volume_type, url, detail=None, zipped_file=None):
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
        if 'SIIBRA_URL_MODS' in environ and url:
            mods = json.loads(environ['SIIBRA_URL_MODS'])
            for old,new in mods.items():
                self.url = self.url.replace(old,new)
            if self.url!=url:
                logger.warning(f'Applied URL modification\nfrom {url}\nto   {self.url}') 
        self.volume_type = volume_type
        self.detail = {} if detail is None else detail
        self.zipped_file = zipped_file
        self.space = None

    def __str__(self):
        return f'{self.volume_type} {self.url}'

    def get_url(self):
        return self.url

    def fetch(self,resolution_mm=None):
        """
        Loads and returns a Nifti1Image object representing the volume source.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        """
        if self.volume_type=='nii':
            filename = download_file(self.url,ziptarget=self.zipped_file)
            if filename:
                return nib.load(filename)
            else:
                return None

        elif self.volume_type=='neuroglancer/precomputed':
            transform_nm = np.identity(4)
            vsrc_labelindex = None
            if 'neuroglancer/precomputed' in self.detail:
                if 'transform' in self.detail['neuroglancer/precomputed']:
                    transform_nm = np.array(self.detail['neuroglancer/precomputed']['transform'])
                if 'labelIndex' in self.detail['neuroglancer/precomputed']:
                    vsrc_labelindex = int(self.detail['neuroglancer/precomputed']['labelIndex'])
            volume = NgVolume(self.url,transform_nm=transform_nm)
            img = volume.build_image(resolution_mm=resolution_mm)
            if vsrc_labelindex is not None:
                # this volume src has a local labelindex - we apply it
                img.dataobj[img.dataobj!=vsrc_labelindex] = 0
            return img

        else:
            logger.error(f'Cannot fetch a volume for type "{self.volume_type}".')
            return None

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a VolumeSrc
        object from a json stream.
        """
        if "@type" in obj and obj['@type'] == "fzj/tmp/volume_type/v0.0.1":
            return VolumeSrc(obj['@id'], obj['name'],
                    volume_type=obj['volume_type'],
                    url=obj['url'],
                    detail=obj.get('detail'),
                    zipped_file=obj.get('zipped_file')
                    )
        
        return obj
