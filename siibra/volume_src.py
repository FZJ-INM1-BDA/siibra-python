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

class VolumeSrc:

    def __init__(self, identifier, name, volume_type, url, detail=None):
        self.id = identifier
        self.name = name
        self.url = url
        self.volume_type = volume_type
        self.detail=detail

    def __str__(self):
        return f'{self.volume_type} {self.url}'

    def get_url(self):
        return self.url

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if "@type" in obj and obj['@type'] == "fzj/tmp/volume_type/v0.0.1":
            return VolumeSrc(obj['@id'], obj['name'],
                    volume_type=obj['volume_type'],
                    url=obj['url'],
                    detail=obj.get('detail'))
        
        return obj
