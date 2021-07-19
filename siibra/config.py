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
from tqdm import tqdm
import os
from collections import defaultdict
import re

# Until openminds is fully supported, 
# we store atlas configurations in a gitlab repo.
# We tag the configuration with each release
GITLAB_PROJECT_TAG=os.getenv("SIIBRA_CONFIG_GITLAB_PROJECT_TAG", "siibra-{}".format(__version__))

logger.info(f"Loading configuration: {GITLAB_PROJECT_TAG}")


class ConfigurationRegistry:
    """
    Registers atlas configurations from json files managed in EBRAINS, by
    converting them to a specific object class based on the object construction
    function provided as constructor parameter. Used for atlas, space, and parcellation
    configurations. 

    This will be migrated to atlas ontology and openMINDS elememts from the KG in the future.

    TODO provide documentation and mechanisms to keep this fixed to a certain release.
    """

    GITLAB_CONFIGS=[{
        'SERVER': 'https://jugit.fz-juelich.de',
        'PROJECT_ID': 3484,
    }, {
        'SERVER': 'https://gitlab.ebrains.eu',
        'PROJECT_ID': 93,
    }]

    def __load_config(self):
        """
        Determine, load and cache config files.
        """
        sfmt = lambda s: re.sub('_+','_',re.sub('[:/.-]','_',s))
        for gitlab_config in self.GITLAB_CONFIGS:
            GITLAB_SERVER=gitlab_config.get('SERVER')
            GITLAB_PROJECT_ID=gitlab_config.get('PROJECT_ID')
            cachefile = os.path.join(CACHEDIR,f"{sfmt(GITLAB_SERVER)}_{GITLAB_PROJECT_ID}_{GITLAB_PROJECT_TAG}_files.json")
            if os.path.isfile(cachefile):
                # we do have a cache! read and return
                with open(cachefile,'r') as f:
                    return json.load(f)

        # No cached config folders. Parse gitlab repositories with atlas configurations
        for gitlab_config in self.GITLAB_CONFIGS:
            try:
                GITLAB_SERVER=gitlab_config.get('SERVER')
                GITLAB_PROJECT_ID=gitlab_config.get('PROJECT_ID')
                if GITLAB_SERVER is None or GITLAB_PROJECT_ID is None:
                    raise ValueError('Both SERVER and PROJECT_ID are required')
                logger.debug(f'Attempting to connect to {GITLAB_SERVER}')
                # 10 second timeout
                project=Gitlab(gitlab_config['SERVER'], timeout=10).projects.get(GITLAB_PROJECT_ID)
                break
            except gitlab_exceptions.GitlabError:
                # Gitlab server down. Try the next one.
                logger.info(f'Gitlab server at {GITLAB_SERVER} is unreachable. Trying another mirror...')
            except ValueError:
                logger.warn('Gitlab configuration malformed')
        else:
            # will not be reached if the for loop is broken
            raise ValueError('No Gitlab server with siibra configurations can be reached')
            
        config = defaultdict(dict)
        for node in project.repository_tree(ref=GITLAB_PROJECT_TAG):
            if node['type']!='tree' or node['name'].startswith('_'):
                continue
            folder = node['name']
            files = list(filter(
                lambda v: v['type']=='blob' and v['name'].endswith('.json'),
                project.repository_tree(path=folder,ref=GITLAB_PROJECT_TAG,all=True) ))
            msg=f"Initializing configuration of {folder:15.15}"
            for configfile in tqdm(files,total=len(files),desc=msg,unit=" files"):
                fname = configfile['name']
                # cache the individual config file; store cachefile name in dict
                cachefile = os.path.join(CACHEDIR,f"{sfmt(GITLAB_SERVER)}_{GITLAB_PROJECT_ID}_{GITLAB_PROJECT_TAG}_{folder}_{fname}")
                config[folder][fname] = cachefile
                p = project.files.get(file_path=folder+"/"+fname, ref=GITLAB_PROJECT_TAG)
                with open(cachefile,'wb') as f:
                    f.write(p.decode())

        # store dict of config cachefiles
        cachefile = os.path.join(CACHEDIR,f"{sfmt(GITLAB_SERVER)}_{GITLAB_PROJECT_ID}_{GITLAB_PROJECT_TAG}_files.json")
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

        config = self.__load_config()
        assert(config_subfolder in config.keys())
        
        config_cachefiles = config[config_subfolder] 
        self.items = []
        self.by_key = {}
        self.by_id = {}
        self.by_name = {}
        self.cls = cls
        loglevel = logger.getEffectiveLevel()
        logger.setLevel("ERROR") # be quiet when initializing the object
        for fname,cachefile in config_cachefiles.items():
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

