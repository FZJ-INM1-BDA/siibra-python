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
from .volume_src import VolumeSrc
from . import logger
import copy

class Space:
    """
    A particular brain reference space.
    """

    def __init__(self, identifier, name, template_type=None, src_volume_type=None, volume_src={}):
        self.id = identifier
        self._rename(name)
        self.type = template_type
        self.src_volume_type = src_volume_type
        self.volume_src = volume_src
        self.template = None
        self._assign_volume_sources(volume_src)

    def _assign_volume_sources(self,volume_src):
        self.volume_src = copy.deepcopy(volume_src)
        for volsrc in self.volume_src:
            volsrc.space = self
            if volsrc.volume_type==self.type:
                self.template = volsrc

    def _rename(self,newname):
        self.name = newname
        self.key = create_key(self.name)

    def __str__(self):
        return self.name

    def get_template(self, resolution_mm=None ):
        """
        Get the volumetric reference template image for this space.

        Parameters
        ----------
        resolution_mm : float or None (Default: None)
            Request the template at a particular physical resolution in mm. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if self.template:
            return self.template.fetch(resolution_mm)
        else:
            logger.error(f'Downloading template for reference space "{self.name}" not supported.')
            return None

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a Space
        object from a json stream.
        """
        required_keys = ['@id','name','shortName','templateType']
        if any([k not in obj for k in required_keys]):
            return obj

        if "minds/core/referencespace/v1.0.0" not in obj['@id']:
            return obj

        volume_src = [VolumeSrc.from_json(v) for v in obj['volumeSrc']] if 'volumeSrc' in obj else []
        
        return Space(obj['@id'], obj['shortName'], template_type = obj['templateType'],
                src_volume_type = obj['srcVolumeType'] if 'srcVolumeType' in obj else None,
                volume_src = volume_src)


REGISTRY = ConfigurationRegistry('spaces', Space)
