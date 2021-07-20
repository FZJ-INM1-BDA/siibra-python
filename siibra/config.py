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
from . import logger,__version__
from .commons import create_key
from .retrieval import CACHEDIR
from gitlab import Gitlab, exceptions as gitlab_exceptions
from tempfile import mkstemp
from tqdm import tqdm
import os
from collections import defaultdict
import re

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

    GITLAB_CONFIGS=[{
        'SERVER': 'https://jugit.fz-juelich.de',
        'PROJECT_ID': 3484,
    }, {
        'SERVER': 'https://gitlab.ebrains.eu',
        'PROJECT_ID': 93,
    }]
    
    logger.info(f"Configuration: {GITLAB_PROJECT_TAG}")

    def __load_config(self,config_folder):
        """
        Find, load and cache siibra configuration files from the separately maintained gitlab configuration repository.
        """

        for gitlab_config in self.GITLAB_CONFIGS:
            GITLAB_SERVER=gitlab_config.get('SERVER')
            GITLAB_PROJECT_ID=gitlab_config.get('PROJECT_ID')
            cachefile = os.path.join(CACHEDIR,f"config_{GITLAB_PROJECT_TAG}_{config_folder}.json")
            if os.path.isfile(cachefile):
                # we do have a cache! read and return
                logger.debug(f"Loading cached configuration '{GITLAB_PROJECT_TAG}' for {config_folder}")
                with open(cachefile,'r') as f:
                    return json.load(f)

        # No cached configuration found. 
        # Parse the gitlab repositories for atlas configurations.
        # Cache a configuration only if GITLAB_PROJECT_TAG is really a fixed tag.
        activate_caching = False
        for gitlab_config in self.GITLAB_CONFIGS:
            try:
                GITLAB_SERVER=gitlab_config.get('SERVER')
                GITLAB_PROJECT_ID=gitlab_config.get('PROJECT_ID')
                if GITLAB_SERVER is None or GITLAB_PROJECT_ID is None:
                    raise ValueError('Both SERVER and PROJECT_ID are required')
                logger.debug(f'Attempting to connect to {GITLAB_SERVER}')
                # 10 second timeout
                project=Gitlab(gitlab_config['SERVER'], timeout=10).projects.get(GITLAB_PROJECT_ID)
                if GITLAB_PROJECT_TAG in map(lambda t:t.name,project.tags.list()):
                    activate_caching = True
                break
            except gitlab_exceptions.GitlabError:
                # Gitlab server down. Try the next one.
                logger.info(f'Gitlab server at {GITLAB_SERVER} is unreachable. Trying another mirror...')
            except ValueError:
                logger.warn('Gitlab configuration malformed')
        else:
            # will not be reached if the for loop is broken
            raise ValueError('No Gitlab server with siibra configurations can be reached')
            
        config = {}
        for node in project.repository_tree(ref=GITLAB_PROJECT_TAG):
            if node['type']!='tree' or node['name']!=config_folder:
                continue
            files = list(filter(
                lambda v: v['type']=='blob' and v['name'].endswith('.json'),
                project.repository_tree(path=config_folder,ref=GITLAB_PROJECT_TAG,all=True) ))
            msg=f"Retrieving configuration '{GITLAB_PROJECT_TAG}' for {config_folder:15.15}"
            for configfile in tqdm(files,total=len(files),desc=msg,unit=" files"):
                fname = configfile['name']
                # retrieve the config file contents and store to temporary or cache file.
                if activate_caching:
                    localfile = os.path.join(CACHEDIR,f"config_{GITLAB_PROJECT_TAG}_{config_folder}_{fname}")
                    f = open(localfile,'wb')
                else:
                    handle,localfile = mkstemp()
                    f = os.fdopen(handle, "wb")
                config[fname] = localfile
                p = project.files.get(file_path=config_folder+"/"+fname, ref=GITLAB_PROJECT_TAG)
                f.write(p.decode())
                f.close()

        # activate cache only if the gitlab project tag was a protected tag. 
        # For other tags (e.g. if the project tag is a branch), 
        # the cache will be updated each time siibra is loaded.
        if activate_caching:
            cachefile = os.path.join(CACHEDIR,f"config_{GITLAB_PROJECT_TAG}_{config_folder}.json")
            with open(cachefile,'w') as f:
                json.dump(config,f,indent='\t')

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
        for fname,cachefile in config.items():
            with open(cachefile,'r') as f:
                obj = json.load(f, object_hook=cls.from_json)
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
            words = [w for w in re.split('[ -]',index)]
            squeezedname = lambda item: item.name.lower().replace(" ","")
            matches = [i for i in self.items
                        if all(w.lower() in squeezedname(i) for w in words)
                        or index.replace(" ","") in squeezedname(i)]
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

