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
from zipfile import ZipFile
import requests
import hashlib
import os
from gitlab import Gitlab
from gitlab.exceptions import GitlabError
from . import logger
from datetime import datetime
from glob import glob
import re

# TODO unify download_file and cached_get
# TODO manage the cache: limit total memory used, remove old zips,...

# Ideas:
#
# this module can detect several flavors of download file specifactions, for example:
# - A nifti file on a standard http URL, no auth required -> just download it
# - A zip file on a standard http URL -> We need additional specificcation of the desired file in side the zip
#   (needs to be reflected in the metadata scheme for e.g. spaces and parcellations)
# - a UID of data provider like EBRAINS -> need to detecto the provider, and use an auth key
#   that is setup during package installation
#


def __compile_cachedir(suffix=None):
    from os import path,makedirs,environ
    from appdirs import user_cache_dir
    if "SIIBRA_CACHEDIR" in environ:
        cachedir = environ['SIIBRA_CACHEDIR']
    else:
        dname = __name__ if suffix is None else __name__+"_"+suffix
        cachedir = user_cache_dir(__name__,"")
    # make sure cachedir exists and is writable
    try:
        if not path.isdir(cachedir):
            makedirs(cachedir)
        assert(os.access(cachedir, os.W_OK))
    except Exception as e:
        print(str(e))
        raise PermissionError('Cannot create local cache folder at {}. Please set a writable cache directory using the environment variable SIIBRA_CACHEDIR, and reload siibra.'.format(cachedir))
    return cachedir

CACHEDIR = __compile_cachedir()
logger.debug('Using cache: {}'.format(CACHEDIR))

hashstr = lambda s: str(hashlib.sha256(s).hexdigest())

def cachefile(str_rep,suffix=None):
    hashfile = os.path.join(CACHEDIR,hashstr(str_rep))
    if suffix is None:
        return hashfile
    else:
        return hashfile+"."+suffix


def download_file(url, ziptarget=None, targetname=None ):
    """
    Downloads a file from a URL to local disk, and returns the local filename.

    Parameters
    ----------
    url : string
        Download link
    ziptarget : string
        [Optional] If the download gives a zip archive, this paramter gives the
        filename to be extracted from the archive.
    targetname : string (optional)
        Desired filename after download.

    TODO Handle and write tests for non-existing URLs, password-protected URLs, too large files, etc.
    """

    # Existing downloads are indicated by a hashfile generated from the URL,
    # which includes the filename of the actual image. This is a workaround to
    # deal with the fact that we do not know the filetype prior to downloading,
    # so we cannot determine the suffix in advance.
    url_str = url if ziptarget is None else url+ziptarget
    hashfile = cachefile(str.encode(url_str))
    if os.path.exists(hashfile):
        with open(hashfile, 'r') as f:
            filename,cachename = f.read().split(';')
            if os.path.exists(cachename):
                return cachename

    # No valid hash and corresponding file found - need to download
    req = requests.get(url)
    if req is not None and req.status_code == 200:

        # try to figure out the original filename for the requested file
        if targetname is not None:
            original_filename = targetname
        elif 'X-Object-Meta-Orig-Filename' in req.headers:
            original_filename = req.headers['X-Object-Meta-Orig-Filename']
        else:
            original_filename = os.path.basename(url)

        # build a uid-based alternative filename for the cache which is not redundant
        # (but keep the original filename for reference)
        namefields = os.path.basename(original_filename).split(".")
        suffix =  ".".join(namefields[1:]) if len(namefields)>1 else ".".join(namefields)
        cachename = hashfile+"."+suffix
        filename = original_filename

        # now save the file
        with open(cachename, 'wb') as code:
            code.write(req.content)

        # if this was a zip file, and a particular target file in the zip was
        # requested, we need to extract it now. We will later drop the zipfile.
        if suffix.endswith("zip") and (ziptarget is not None):
            filename = ziptarget
            cachename = get_from_zip(
                    cachename, ziptarget)
        with open(hashfile, 'w') as f:
            f.write(filename+";")
            f.write(cachename)
        return cachename
    '''
        - error on response status != 200
        - error on file read
        - Nibable error
        - handle error, when no filename header is set
        - error or None when space not known
        - unexpected error
        '''


def get_from_zip(zipfile, ziptarget ):
    # Extract temporary zip file
    # TODO catch problem if file is not a nifti
    targetname = None
    with ZipFile(zipfile, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = os.path.basename(zip_info.filename)
            if zip_info.filename == ziptarget:
                downloadname = CACHEDIR + '/' + zip_info.filename
                zip_ref.extract(zip_info, CACHEDIR)
                targetname = "{}.{}".format(
                    os.path.splitext(zipfile)[0],
                    os.path.basename(ziptarget) )
                if not os.path.exists(targetname):
                    os.rename(downloadname,targetname)
    os.remove(zipfile)
    if targetname is not None:
        return targetname
    raise Exception("target file",ziptarget,"not found in zip archive",zipfile)


def get_json_from_url(url):
    req = requests.get(url)
    if req is not None and req.status_code == 200:
        return json.loads(req.content)
    else:
        return {}


def cached_get(url,msg_if_not_cached=None,**kwargs):
    """
    Performs a requests.get if the result is not yet available in the local
    cache, otherwise returns the result from the cache.
    This leaves the interpretation of the returned content to the caller.
    TODO we might extend this as a general tool for the siibra library, and make it a decorator
    TODO this hashes the authentication token as part of kwargs, which might not be desired
    """
    logger.debug(f"Get from URL\n{url}")
    logger.debug(f"kwargs: {kwargs}")
    url_hash = hashlib.sha256((url+json.dumps(kwargs)).encode('ascii')).hexdigest()
    cachefile_content = os.path.join(CACHEDIR,url_hash)+".content"
    cachefile_url = os.path.join(CACHEDIR,url_hash)+".url"

    if os.path.isfile(cachefile_content):
        # This URL target is already in the cache - just return it
        logger.debug("Returning cached response of url {} at {}".format(url,cachefile_content))
        with open(cachefile_content,'rb') as f:
            r = f.read()
            return(r)
    else:
        if msg_if_not_cached:
            print(msg_if_not_cached)
        r = requests.get(url,**kwargs)
        if r.ok:
            with open(cachefile_content,'wb') as f:
                f.write(r.content)
            with open(cachefile_url,'w') as f:
                f.write(url)
            return r.content
        elif any(d in url for d in ["ebrains","humanbrainproject"]):
            if r.status_code == 401:
                raise RuntimeError('The provided EBRAINS authentication token is not valid')
            elif r.status_code == 403:
                raise RuntimeError('No permission to access the given query')
            elif r.status_code == 404:
                raise RuntimeError('Query with this id not found')
        raise RuntimeError(f'Could not retrieve data.\nhttp status code: {r.status_code}\nURL: {url}')

def clear_cache():
    import shutil
    logger.info("Clearing siibra cache.")
    shutil.rmtree(CACHEDIR)
    __compile_cachedir()


def cached_gitlab_query(server,project_id,ref_tag,folder=None,filename=None,recursive=False,return_hidden_files=False,skip_branchtest=False):
    """
    Cached retrieval of folder list or file content from a public gitlab repositoriy.

    Arguments
    ---------
    server : string
        Gitlab server main url
    project_id : int
        Id of the requested project (you can find this in the project settings of the gitlab repository)
    ref_tag : string
        Reference tag, e.g. name of a branch, tag, or checksum.
        NOTE branches are not cached, since the branch name 
        does not determine the version of the repository.
    folder : string or None
        Optional name of repository subfolder to use
    filename : string or None
        Optional name of file requested. 
        If None, folder contents are returned as a dictionary, 
        otherwise the bytestream of the file contents.
    skip_branchtest: bool, default: False
        If true, the ref_tag is assumed not to be a branch, so that
        it will point to a unique version. This will prevent an initial
        test to classify the tag into branch or not, which requires an
        additional query to the server.
    """

    def connect(server,project_id):
        project = None
        try:
            logger.debug(f'Attempting to connect to {server}')
            project = Gitlab(server, timeout=10).projects.get(project_id)
        except (ConnectionError,GitlabError):
            # Gitlab server down.
            logger.warn(f'Gitlab server at {server} unreachable.')
        except ValueError as e:
            print(str(e))
            logger.warn('Gitlab configuration malformed')
        return project

    # try to connect to the server and classify the ref_tag (is it a branch?)
    want_branch_commit = None
    project = None
    connection_failed = False
    if not skip_branchtest:
        project = connect(server,project_id)
        if project is None:
            connection_failed = True
        else:
            matched_branches = list(filter(lambda b:b.name==ref_tag,project.branches.list()))
            if len(matched_branches)>0:
                want_branch_commit = matched_branches[0].commit

    # construct base of cache file names
    pathspec = ""
    if folder is not None:
        pathspec += f"-{folder}"
    if filename is not None:
        pathspec += f"-{filename}"
    if recursive:
        pathspec += "-recursive"
    basename = re.sub( "_+","_",
        re.sub("[/:.]","_",
        f"{server}-{project_id}{pathspec}-{ref_tag}"))
    if want_branch_commit is not None:
        sid = want_branch_commit['short_id']
        tstamp = datetime.fromisoformat(want_branch_commit['created_at']).strftime('%Y%m%d%H%M%S')
        basename += f"_commit{sid}_{tstamp}"

    # Determine desired cachefile
    cachefile = None
    if connection_failed:
        # No connection to server, so look for existing cache content.
        cachefiles_available = glob(os.path.join(CACHEDIR,f"{basename}*"))
        if len(cachefiles_available)>0:
            logger.debug(f"Cannot connect to repository. Looking for most recently cached content for '{ref_tag}'.")
            if len(cachefiles_available)==1:
                # this is either the unique cached version of a configuration tag, 
                # or the single cached version of a branch.
                cachefile = cachefiles_available[0]
            else:
                # We have multiple commits cached from the same branch.
                # Choose the one with newest timestamp.
                get_tstamp = lambda fn: fn.split('_')[-1]
                sorted_cachefiles = sorted(cachefiles_available,key=get_tstamp)
                cachefile = sorted_cachefiles[-1]
    else:
        cachefile = os.path.join(CACHEDIR,f"{basename}")

    # If cached data is available, read and return
    if cachefile is not None and os.path.isfile(cachefile):
        with open(cachefile,'r') as f:
            return f.read()

    # No cached data found, so we need to get the data from the server.
    if skip_branchtest:
        project = connect(server,project_id)
    if project is None:
        raise RuntimeError(f"No connection or cached data for '{server}' project '{project_id}' under reftag '{ref_tag}'.")
    relpath = "" if folder is None else f"{folder}/"
    if filename is not None:
        # A particular file was requested. 
        # This is a bytestream which could be any type of file.
        result = project.files.get(file_path=relpath+filename, ref=ref_tag).decode()
    else:
        # list of folder contents was requested. This is a list of dictionaries, 
        # which we serialize to a json string and then encode to bytes.
        entries = project.repository_tree(path=relpath,ref=ref_tag,all=True,recursive=recursive)
        if not return_hidden_files:
            entries = list(filter(lambda e:not e['name'].startswith('.'),entries))
        result = json.dumps(entries).encode('utf8')
        
    logger.debug(f"Caching gitlab content to {cachefile}")
    assert(isinstance(result,bytes))
    with open(cachefile,'wb') as f:
        f.write(result)

    return result.decode()


class LazyLoader:
    def __init__(self,url=None,func=None):
        """
        Initialize a lazy data loader. It stores a URL and optional function, 
        and will retrieve and decode the actual data only when its 'data' property 
        is accessed the first time.

        Parameters
        ----------
        url : string, or None
            URL for loading raw data, which is then fed into `func` 
            for creating the output. 
            If None, `func` will be called without arguments.
        func : function pointer
            Function for constructing the output data 
            (called on the data retrieved from `url`, if supplied)
        """
        if (url is None) and (func is None):
            logger.warn("Request to build a LazyLoader with an identity function - this makes no sense.")
        self.url = url
        self._data_cached = None
        if func is None:
            self._func = lambda x:x
        else:
            self._func = func

    @property
    def data(self):
        if self._data_cached is None:
            if self.url is None:
                self._data_cached = self._func()
            else:
                raw = cached_get(self.url)
                self._data_cached = self._func(raw)
        return self._data_cached