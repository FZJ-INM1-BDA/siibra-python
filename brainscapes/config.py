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

import json
from os import path
from . import logger
from .commons import create_key
try:
    from importlib.resources import contents as pkg_contents,path as pkg_path
except ImportError as e:
    logger.info("importlib.resources not found. Will use importlib_resources instead.")
    from importlib_resources import contents as pkg_contents,path as pkg_path
from . import retrieval

class ConfigurationRegistry:
    """
    A class that registers configurations from json files by converting
    them to a specific object class based on the object construction function
    provided as constructor parameter. Used for atlas, space, and parcellation
    configurations.
    """

    def __init__(self,pkgpath,cls):
        """
        Populate a new registry from the json files in the package path, using
        the "from_json" function of the provided class as hook function.
        """
        logger.debug("Initializing registry of type {} for {}".format(
            cls,pkgpath))
        object_hook = cls.from_json
        self.items = []
        self.by_key = {}
        self.by_id = {}
        self.by_name = {}
        self.cls = cls
        config_files = retrieval.get_config_files_by_type(str(pkgpath).split('.')[-1])
        for configfile in config_files:
            obj = retrieval.get_json_for_config_file(configfile, object_hook)
            key = create_key(str(obj))
            identifier = obj.id
            logger.debug("Defining object '{}' with key '{}'".format( obj,key))
            self.items.append(obj)
            self.by_key[key] = len(self.items)-1
            self.by_id[identifier] = len(self.items)-1
            self.by_name[obj.name] = len(self.items)-1
        

    def __getitem__(self,index):
        """
        Item access is implemented either by sequential index, key or id.
        """
        if isinstance(index,int) and index<len(self.items):
            return self.items[index]
        elif isinstance(index,self.cls) and (index in self.items):
            # index is itself already an object of this registry - forward
            return index
        elif index in self.by_key:
            return self.items[self.by_key[index]]
        elif index in self.by_id:
            return self.items[self.by_id[index]]
        elif index in self.by_name:
            return self.items[self.by_name[index]]
        else:
            raise ValueError("Cannot access this item in the {} Registry:".format(
                self.cls),index)

    def __dir__(self):
        return list(self.by_key.keys()) + list(self.by_id.keys())

    def __str__(self):
        return "\n".join([i.key for i in self.items])

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self.by_key.keys():
            return self.items[self.by_key[name]]
        else:
            raise AttributeError("No such attribute: {}".format(name))

