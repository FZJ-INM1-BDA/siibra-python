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

from .logging import logger

class OriginDataInfo:
    def __init__(self, name, description=None, urls=[]):
        self.name=name
        self.description=description
        self.urls=urls

    @staticmethod
    def from_json(jsonstr):
        json_type=jsonstr.get('@type')
        if json_type == 'fzj/tmp/simpleOriginInfo/v0.0.1':
            return OriginDataInfo(name=jsonstr.get('name'),
                        description=jsonstr.get('description'),
                        urls=jsonstr.get('url', []))
        elif json_type == 'minds/core/dataset/v1.0.0':
            from .ebrains import EbrainsOriginDataInfo
            print(jsonstr.keys())
            return OriginDataInfo(id=jsonstr.get('kgId'),name=jsonstr.get('name'))
        logger.warn(f'Cannot parse {jsonstr}')
        return None


class Dataset:
    """
    Parent class for all types of datasets. 
    A dataset is supposed to originate from some form of online repository, 
    and thus carries information about its origin(s).
    """
    def __init__(self,identifier,name):
        self.id = identifier
        self.name = name
        self.origin_datainfos = []

    def _add_originDatainfo(self,spec:OriginDataInfo):
        """
        Build and OriginDataInfo object from a json specification, and add it if successful.
        """
        obj = OriginDataInfo.from_json(spec)
        if obj is None:
            logger.warn("Could not create OriginDataInfo object from json spec.")
        else:
            self.origin_datainfos.append(obj)
