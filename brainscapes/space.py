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

class Space:

    def __init__(self, identifier, name, template_url=None, ziptarget=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.url = template_url
        self.ziptarget = ziptarget

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if '@id' in obj and "minds/core/referencespace/v1.0.0" in obj['@id']:
            if 'templateFile' in obj:
                return Space(obj['@id'], obj['name'], obj['templateUrl'], 
                        ziptarget=obj['templateFile'])
            else:
                return Space(obj['@id'], obj['name'], obj['templateUrl'])
        return obj

REGISTRY = ConfigurationRegistry('spaces', Space)
