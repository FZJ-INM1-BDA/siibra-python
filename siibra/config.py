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

from . import logger,__version__
from .commons import create_key
from .retrieval import GitlabQuery #cached_gitlab_query
from tqdm import tqdm
import os
import re
import json

# Until openminds is fully supported, 
# we store atlas configurations in a gitlab repo.
# We tag the configuration with each release
GITLAB_PROJECT_TAG=os.getenv("SIIBRA_CONFIG_GITLAB_PROJECT_TAG", "siibra-{}".format(__version__))

class ConfigurationRegistry:
    """
    Registers atlas configurations from siibra configuration files 
    managed in a separately maintained gitlab configuration repository.
    Each json file is converted to a specific object class based on the object construction
    function provided as constructor parameter.  
    
    The target class is determined from the folder of the configuration 
    repository (atlases, parcellations, spaces).    
    """

    logger.debug(f"Configuration: {GITLAB_PROJECT_TAG}")
    uses_default_tag = not "SIIBRA_CONFIG_GITLAB_PROJECT_TAG" in os.environ
    _QUERIES = ( 
        # we use an iterator we we only instantiate the one[s] used
        GitlabQuery(
            'https://jugit.fz-juelich.de',3484,
            GITLAB_PROJECT_TAG,skip_branchtest=uses_default_tag),
        GitlabQuery(
            'https://gitlab.ebrains.eu',93,
            GITLAB_PROJECT_TAG,skip_branchtest=uses_default_tag),
    )
    
    def __load_config(self,config_folder):
        """
        Find, load and cache siibra configuration files from the separately maintained gitlab configuration repository.
        """
        msg=f"Retrieving configuration '{GITLAB_PROJECT_TAG}' for {config_folder:15.15}"
        for query in self._QUERIES:
            try:
                config = {}
                for configfile,data in tqdm(query.iterate_files(config_folder,'.json'),desc=msg):
                    config[configfile] = json.loads(data)
                break                
            except Exception as e:
                print(str(e))
                logger.error(f"Cannot connect to configuration server {query.server}, trying a different mirror")
        else:
            # we get here only if the loop is not broken
            raise RuntimeError(f"Cannot initialize atlases: No configuration data found for '{GITLAB_PROJECT_TAG}'.")

        return config

    def __init__(self,config_subfolder,cls):
        """
        Populate a new registry from the json files in the package path, using
        the "from_json" function of the provided class as hook function.
        """
        logger.debug("Initializing registry of type {} for {}".format(
            cls,config_subfolder))

        config = self.__load_config(config_subfolder)
        
        self.items = []
        self.by_key = {}
        self.by_id = {}
        self.by_name = {}
        self.cls = cls
        loglevel = logger.getEffectiveLevel()
        logger.setLevel("ERROR") # be quiet when initializing the object
        for fname,spec in config.items():
            obj = cls.from_json(spec)
            if not isinstance(obj,cls):
                raise RuntimeError(f'Could not generate object of type {cls} from configuration {fname}')
            key = create_key(str(obj))
            identifier = obj.id
            self.items.append(obj)
            self.by_key[key] = len(self.items)-1
            self.by_id[identifier] = len(self.items)-1
            self.by_name[obj.name] = len(self.items)-1
        logger.setLevel(loglevel)

        
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
        elif isinstance(index,str):
            # if a string is specified, we check if each word is matched in a space name
            matches = self.find(index)
            if len(matches)==1:
                return matches[0]
            elif len(matches)>1:
                # see if the matches are from the same collection - then we disambiguate to the newest version
                try:
                    collections = {m.version.collection for m in matches}
                    if len(collections)==1:
                        return sorted(matches,key=lambda m:m.version,reverse=True)[0]
                except Exception as e:
                    pass
                namelist = ", ".join(m.name for m in matches)
                logger.warning(f"Specification '{index}' yielded {len(matches)} different matches: {namelist}")
        raise IndexError(f"Cannot identify item '{index}' in {self.cls.__name__} registry")

    def find(self,spec:str):
        """
        Find items in the registry by matching words in the name or id.
        """
        words = [w for w in re.split('[ -]',spec)]
        squeezedname = lambda item: item.name.lower().replace(" ","")
        return [i for i in self.items if any([
            all(w.lower() in squeezedname(i) for w in words),
            spec.replace(" ","") in squeezedname(i),
            spec==i.id]) ]
    
    def __len__(self):
        return len(self.items)

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

