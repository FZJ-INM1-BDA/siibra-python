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
from gitlab import Gitlab
from gitlab.exceptions import GitlabError
from requests.exceptions import ConnectionError
from tempfile import mkstemp
from tqdm import tqdm
import os
from collections import defaultdict
import re
from datetime import datetime
from glob import glob

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

    GITLAB_CONFIGURATION_REPOSITORIES=[{
        'SERVER': 'https://jugit.fz-juelich.de',
        'PROJECT_ID': 3484,
    }, {
        'SERVER': 'https://gitlab.ebrains.eu',
        'PROJECT_ID': 93,
    }]
    
    logger.debug(f"Configuration: {GITLAB_PROJECT_TAG}")

    def __load_config(self,config_folder):
        """
        Find, load and cache siibra configuration files from the separately maintained gitlab configuration repository.
        """

        # try to connect to a configuration server
        want_branch_commit = None
        for gitlab_config in self.GITLAB_CONFIGURATION_REPOSITORIES:
            try:
                GITLAB_SERVER=gitlab_config.get('SERVER')
                GITLAB_PROJECT_ID=gitlab_config.get('PROJECT_ID')
                if GITLAB_SERVER is None or GITLAB_PROJECT_ID is None:
                    raise ValueError('Both SERVER and PROJECT_ID are required to determine a siibra configuration repository')
                logger.debug(f'Attempting to connect to {GITLAB_SERVER}')
                project = Gitlab(gitlab_config['SERVER'], timeout=10).projects.get(GITLAB_PROJECT_ID)
                matched_branches = list(filter(lambda b:b.name==GITLAB_PROJECT_TAG,project.branches.list()))
                if len(matched_branches)>0:
                    want_branch_commit = matched_branches[0].commit
                repository_reached = True
                break
            except (ConnectionError,GitlabError):
                # Gitlab server down. Try the next one.
                logger.debug(f'Gitlab server at {GITLAB_SERVER} unreachable. Trying next mirror.')
            except ValueError:
                logger.warn('Gitlab configuration malformed')
        else:
            # will not be reached if the for loop is broken.
            repository_reached = False
            logger.debug(f'No access to any repository with configurations for {config_folder}.')

        # Decide wether to access a possibly cached configuration
        cachefile = None
        basename = f"{config_folder}_{GITLAB_PROJECT_TAG}"
        if want_branch_commit is not None:
            sid = want_branch_commit['short_id']
            tstamp = datetime.fromisoformat(want_branch_commit['created_at']).strftime('%Y%m%d%H%M%S')
            basename += f"_commit{sid}_{tstamp}"
        if repository_reached:
            # we did connect to a configuration repository above, 
            # so we can rely on our information on the project tag.
            cachefile = os.path.join(CACHEDIR,f"config_{basename}.json")
        else:
            # We seem to be offline, so we take the best we can get from the local cache.
            cachefiles_available = glob(os.path.join(CACHEDIR,f"config_{basename}*.json"))
            if len(cachefiles_available)>0:
                logger.debug(f"Cannot connect to repository. Looking for most recently cached configuration '{GITLAB_PROJECT_TAG}' for {config_folder}.")
                if len(cachefiles_available)==1:
                    # this is either the unique cached version of a configuration tag, 
                    # or the single cached version of a branch.
                    cachefile = cachefiles_available[0]
                else:
                    # We have multiple commits cached from the same branch.
                    # Choose the one with newest timestamp.
                    get_tstamp = lambda fn: fn.replace('.json','').split('_')[-1]
                    sorted_cachefiles = sorted(cachefiles_available,key=get_tstamp)
                    cachefile = sorted_cachefiles[-1]

        if cachefile is not None and os.path.isfile(cachefile):
            # we do have a cache! read and return
            logger.info(f"Loading cached configuration '{GITLAB_PROJECT_TAG}' for {config_folder}")
            with open(cachefile,'r') as f:
                return json.load(f)

        # No cached configuration found. 
        if not repository_reached:
            raise RuntimeError(f"Cannot initialize atlases: No cached configuration data for '{GITLAB_PROJECT_TAG}'', and no access to any of the configuration repositories either.")

        # Load configuration from  repository.
        config = {}
        for node in project.repository_tree(ref=GITLAB_PROJECT_TAG):
            if node['type']!='tree' or node['name']!=config_folder:
                continue
            files = list(filter(
                lambda v: v['type']=='blob' and v['name'].endswith('.json'),
                project.repository_tree(path=config_folder,ref=GITLAB_PROJECT_TAG,all=True) ))
            msg=f"Retrieving configuration '{GITLAB_PROJECT_TAG}' for {config_folder:15.15}"
            for configfile in tqdm(files,total=len(files),desc=msg,unit=" files"):
                # retrieve the config file contents and store in cache.
                fname = configfile['name']
                config[fname] = os.path.join(CACHEDIR,f"{basename}_{fname}")
                p = project.files.get(file_path=config_folder+"/"+fname, ref=GITLAB_PROJECT_TAG)
                with open(config[fname],'wb') as f:
                    f.write(p.decode())

        logger.debug("Creating cachefile",cachefile)
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

