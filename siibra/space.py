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

from .commons import create_key
from .config import ConfigurationRegistry
from .retrieval import download_file
from .bigbrain import BigBrainVolume
from . import logger
import nibabel as nib

class Space:

    def __init__(self, identifier, name, template_type=None, template_url=None, ziptarget=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.type = template_type
        self.url = template_url
        self.ziptarget = ziptarget

    def __str__(self):
        return self.name

    def get_template(self, resolution=None ):
        """
        Get the volumetric reference template image for this space.

        Parameters
        ----------
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if self.type == 'nii':
            logger.debug('Loading template image for space {}'.format(self.name))
            filename = download_file( self.url, ziptarget=self.ziptarget )
            if filename is not None:
                return nib.load(filename)
            else:
                return None

        if self.type == 'neuroglancer':
            return BigBrainVolume(self.url).build_image(resolution)

        logger.error('Downloading the template image for the requested reference space is not supported.')
        logger.error('- Requested space: {}'.format(self.name))
        return None

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        required_keys = ['@id','name','shortName','templateUrl','templateType']
        if any([k not in obj for k in required_keys]):
            logger.warning('Could not parse Space object')
            return obj

        if '@id' in obj and "minds/core/referencespace/v1.0.0" in obj['@id']:
            if 'templateFile' in obj:
                return Space(obj['@id'], obj['shortName'], 
                        template_url = obj['templateUrl'], 
                        template_type = obj['templateType'],
                        ziptarget=obj['templateFile'])
            else:
                return Space(obj['@id'], obj['shortName'], 
                        template_url = obj['templateUrl'], 
                        template_type = obj['templateType'])
        return obj

REGISTRY = ConfigurationRegistry('spaces', Space)
