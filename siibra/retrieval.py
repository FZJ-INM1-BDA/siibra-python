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
from zipfile import LargeZipFile, ZipFile
from urllib.parse import quote
import requests
import hashlib
import os
from . import logger
import base64
from tqdm import tqdm
from appdirs import user_cache_dir
from nibabel import Nifti1Image
import gzip

# TODO manage the cache: limit total memory used, remove old zips,...

class Cache:

    _instance = None
    folder = user_cache_dir('.'.join(__name__.split('.')[:-1]),"")

    def __init__(self):
        raise RuntimeError(f'Call instance() to access {self.__class__.__name__}')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            if 'SIIBRA_CACHEDIR' in os.environ:
                cls.folder = os.environ['SIIBRA_CACHEDIR']
            # make sure cachedir exists and is writable
            try:
                if not os.path.isdir(cls.folder):
                    os.makedirs(cls.folder)
                assert(os.access(cls.folder, os.W_OK))
                logger.debug(f'Setup cache at {cls.folder}')
            except Exception as e:
                print(str(e))
                raise PermissionError(f'Cannot create cache at {cls.folder}. Please define a writable cache directory in the environment variable SIIBRA_CACHEDIR.')
            cls._instance = cls.__new__(cls)
        return cls._instance
        
    def clear(self):
        import shutil
        logger.info(f"Clearing siibra cache at {self.folder}")
        shutil.rmtree(self.folder)

    def build_filename(self,str_rep,suffix=None):
        hashfile = os.path.join(
            self.folder,
            str(hashlib.sha256(str_rep.encode('ascii')).hexdigest()))
        if suffix is None:
            return hashfile
        else:
            return hashfile+"."+suffix

DECODERS = {
    ".nii.gz" : lambda b:Nifti1Image.from_bytes(gzip.decompress(b)),
    ".nii" : lambda b:Nifti1Image.from_bytes(gzip.decompress(b)),
    ".json" : lambda b:json.loads(b.decode())
}

class HttpLoader:

    cache = Cache.instance()

    def __init__(self,url,func=None,status_code_messages={},**kwargs):    
        """
        Initialize a cached http data loader. 
        It takes a URL and optional data conversion function.
        For loading, the http request is only performed if the 
        result is not yet available in the disk cache.
        Leaves the interpretation of the returned content to the caller.

        Parameters
        ----------
        url : string, or None
            URL for loading raw data, which is then fed into `func` 
            for creating the output. 
            If None, `func` will be called without arguments.
        func : function pointer
            Function for constructing the output data 
            (called on the data retrieved from `url`, if supplied)
        status_code_messages : dict
            Optional dictionary of message strings to output in case of error, 
            where keys are http status code.
        """
        assert(url is not None)
        self.url = url
        self.func = func
        self.kwargs = kwargs
        self.status_code_messages=status_code_messages
        self.cachefile = self.cache.build_filename(self.url+json.dumps(kwargs))

    def _retrieve(self):
        # Loads the data from http if required. 
        # If the data is already cached, None is returned,
        # otherwise data (as it is already in memory anyway).
        # The caller should load the cachefile only
        # if None is returned.
        if os.path.isfile(self.cachefile):
            # in cache. Just load the file 
            logger.debug(f"Already in cache at {os.path.basename(self.cachefile)}: {self.url}")
            return
        else:
            # not yet in cache, perform http request.
            logger.debug(f"Loading {self.url} to {os.path.basename(self.cachefile)}")
            r = requests.get(self.url,**self.kwargs)
            if r.ok:
                with open(self.cachefile,'wb') as f:
                    f.write(r.content)
                return r.content
            elif r.status_code in self.status_code_messages:
                raise RuntimeError(self.status_code_messages[r.status_code])
            else:
                raise RuntimeError(f'Could not retrieve data.\nhttp status code: {r.status_code}\nURL: {self.url}') 

    def get(self):
        data = self._retrieve() # returns the data if 
        if data is None:
            with open(self.cachefile,'rb') as f:
                data = f.read() 
        return data if self.func is None else self.func(data)

class LazyHttpLoader(HttpLoader):

    def __init__(self,url,func=None,status_code_messages={},**kwargs):    
        """
        Initialize a lazy and cached http data loader. 
        It stores a URL and optional data conversion function, 
        but loads the actual data only when its 'data' property 
        is accessed the first time.
        For loading, the http request is only performed if the 
        result is not yet available in the disk cache.
        Leaves the interpretation of the returned content to the caller.

        Parameters
        ----------
        url : string, or None
            URL for loading raw data, which is then fed into `func` 
            for creating the output. 
            If None, `func` will be called without arguments.
        func : function pointer
            Function for constructing the output data 
            (called on the data retrieved from `url`, if supplied)
        status_code_messages : dict
            Optional dictionary of message strings to output in case of error, 
            where keys are http status code.
        """
        HttpLoader.__init__(self,url,func,status_code_messages,**kwargs)
        self._data_cached = None

    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = self.get()
        return self._data_cached        


class ZipLoader(LazyHttpLoader):
    
    def __init__(self,url,filename,func=None):
        LazyHttpLoader.__init__(self,url)
        self.filename = filename
        suitable_decoders = [dec for sfx,dec in DECODERS.items() if filename.endswith(sfx)]
        if (func is None) and (len(suitable_decoders)>0):
            assert(len(suitable_decoders)==1)
            self.func = suitable_decoders[0]
        else:
            self.func = func

    @property 
    def data(self):
        if self._data_cached is None:
            self._retrieve()
            with ZipFile(self.cachefile,'r') as zip_ref:
                try:
                    with zip_ref.open(self.filename) as f:
                        self._data_cached = f.read()
                except Exception as e:
                    logger.error(f"Could not get file {self.filename} from cached zip container at {self.cachefile}")
                    raise(e)
        return self._data_cached


class GitlabLoader():

    def __init__(self,server:str,project:int,reftag:str,skip_branchtest=False):
        # TODO: the query builder needs to check wether the reftag is a branch, and then not cache.
        assert(server.startswith("http"))
        self.server = server
        self.project = project
        self.reftag = reftag
        self._per_page = 100
        self._base_url = "{s}/api/v4/projects/{p}/repository".format(s=server,p=project)
        self._branchloader = LazyHttpLoader(
            f"{self._base_url}/branches",DECODERS['.json'])
        self._tag_checked = True if skip_branchtest else False
        self._want_commit_cached = None

    @property
    def want_commit(self):
        if not self._tag_checked:
            try:
                matched_branches = list(filter(lambda b:b['name']==self.reftag,self.branches))
                if len(matched_branches)>0:
                    self._want_commit_cached = matched_branches[0]['commit']
                    print(f"{self.reftag} is a branch of {self.server}/{self.project}! Want last commit {self._want_commit_cached['short_id']} from {self._want_commit_cached['created_at']}")
                self._tag_checked = True
            except Exception as e:
                print(str(e))
                print("Could not connect to gitlab server!")
        return self._want_commit_cached

    @property
    def branches(self):
        return self._branchloader.data

    def _build_url(self,folder="",filename=None,recursive=False,page=1):
        ref = self.reftag if self.want_commit is None else self.want_commit['short_id']
        if filename is None:
            pathstr = "" if len(folder)==0 else f"&path={quote(folder,safe='')}"
            return f"{self._base_url}/tree?ref={ref}{pathstr}&per_page={self._per_page}&page={page}&recursive={recursive}"
        else:
            pathstr = filename if folder=="" else f"{folder}/{filename}"
            filepath = quote(pathstr,safe='')
            return f"{self._base_url}/files/{filepath}?ref={ref}"

    def get_file(self,filename,folder=""):
        loader = LazyHttpLoader(
            self._build_url(folder,filename),DECODERS['.json'])
        content = base64.b64decode(loader.data['content'].encode('ascii'))
        return content.decode()

    def iterate_files(self,folder="",suffix=None,progress=None,recursive=False):
        """
        Returns an iterator over files in a given folder. 
        In each iteration, a tuple (filename,file content) is returned.
        """
        page = 1
        results = []
        while True:
            loader = LazyHttpLoader(
                self._build_url(folder,recursive=recursive,page=page),
                DECODERS['.json'])
            results.extend(loader.data)
            if len(loader.data)<self._per_page:
                # no more pages
                break
            page+=1
        end = "" if suffix is None else suffix
        elements = [ e for e in results
                    if e['type']=='blob' and e['name'].endswith(end) ]
        it = ( (e['path'],self.get_file(e['path'])) for e in elements) 
        if progress is None:
            return it
        else:
            return tqdm(it,total=len(elements),desc=progress)


class OwncloudLoader():

    def __init__(self,server:str,share:int):
        assert(server.startswith("http"))
        self.server = server
        self.share = share
        self._base_url = f"{server}/s/{share}"

    def _build_query(self,filename,folder=""):
        fpath = "" if folder=="" else f"path={quote(folder,safe='')}&"
        fpath += f"files={quote(filename)}"
        url= f"{self._base_url}/download?{fpath}"
        return url

    def load_file(self,filename,folder=""):
        return LazyHttpLoader(self._build_query(filename,folder)).data

    def build_lazyloader(self,filename,folder,func=None):
        suitable_decoders = [dec for sfx,dec in DECODERS.items() if filename.endswith(sfx)]
        if (func is None) and (len(suitable_decoders)>0):
            assert(len(suitable_decoders)==1)
            func = suitable_decoders[0]
        return LazyHttpLoader(self._build_query(filename,folder),func)
