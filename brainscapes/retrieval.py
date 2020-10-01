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
from os import path
from . import logger

# TODO unifiy download_file and cached_get
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


def __compile_cachedir():
    from os import path,makedirs,environ
    from appdirs import user_cache_dir
    if "BRAINSCAPES_CACHEDIR" in environ:
        cachedir = environ['BRAINSCAPES_CACHEDIR']
    else:
        cachedir = user_cache_dir(__name__,"")
    if not path.isdir(cachedir):
        makedirs(cachedir)
    return cachedir

CACHEDIR = __compile_cachedir()
logger.debug('Using cache: {}'.format(CACHEDIR))


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
    hashfile = path.join(CACHEDIR,str(hashlib.sha256(str.encode(url)).hexdigest()))
    if path.exists(hashfile):
        with open(hashfile, 'r') as f:
            filename = f.read()
            if path.exists(filename):
                return filename

    # No valid hash and corresponding file found - need to download
    req = requests.get(url)
    if req is not None and req.status_code == 200:
        if targetname is not None:
            filename = path.join(CACHEDIR,targetname)
        elif 'X-Object-Meta-Orig-Filename' in req.headers:
            filename = path.join(CACHEDIR,req.headers['X-Object-Meta-Orig-Filename'])
        else:
            filename = path.join(CACHEDIR,path.basename(url))
        with open(filename, 'wb') as code:
            code.write(req.content)
        suffix = path.splitext(filename)[-1]
        if (suffix == ".zip") and (ziptarget is not None):
            filename = get_from_zip(
                    filename, ziptarget)
        with open(hashfile, 'w') as f:
            f.write(filename)
        return filename
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
    with ZipFile(zipfile, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = path.basename(zip_info.filename)
            if zip_info.filename == ziptarget:
                zip_ref.extract(zip_info, CACHEDIR)
                return CACHEDIR + '/' + zip_info.filename


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
    TODO we might extend this as a general tool for the brainscapes library, and make it a decorator
    """
    url_hash = hashlib.sha256(url.encode('ascii')).hexdigest()
    cachefile_content = path.join(CACHEDIR,url_hash)+".content"
    cachefile_url = path.join(CACHEDIR,url_hash)+".url"

    if path.isfile(cachefile_content):
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
        elif r.status_code == 401:
            print('The provided authentication token is not valid')
        elif r.status_code == 403:
            print('No permission to access the given query')
        elif r.status_code == 404:
            print('Query with this id not found')
        else:
            print('Problem with "get" protocol on url: %s ' % url )
        raise Exception('Could not retrieve data')
