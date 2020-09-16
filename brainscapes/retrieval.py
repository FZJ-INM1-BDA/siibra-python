import json
from zipfile import ZipFile

import requests
import hashlib
from tempfile import mkdtemp
from os import path, makedirs, environ

# Ideas:
#
# this module can detect several flavors of download file specifactions, for example:
# - A nifti file on a standard http URL, no auth required -> just download it
# - A zip file on a standard http URL -> We need additional specificcation of the desired file in side the zip
#   (needs to be reflected in the metadata scheme for e.g. spaces and parcellations)
# - a UID of data provider like EBRAINS -> need to detecto the provider, and use an auth key
#   that is setup during package installation
#


def __check_and_get_token():
    """
    Check for a token in the environment variables.
    A token must be provided for different http calls, to be authenticated

    TODO Method and therefore the token is not yet used. Clarification needed how to use token against object.cscs store
    """
    try:
        token = environ['HBP_AUTH_TOKEN']
        print('token: ' + token)
    except KeyError:
        print('An authentication token must be set as an environment variable: HBP_AUTH_TOKEN')


def download_file(url, download_folder, ziptarget=None, targetname=None ):
    """
    Downloads a file from a URL to local disk, and returns the local filename.

    Parameters
    ----------
    url : string
        Download link
    download_folder : string
        Path to a folder on the local filesystem for storing the image
    ziptarget : string
        [Optional] If the download gives a zip archive, this paramter gives the
        filename to be extracted from the archive.
    targetname : string (optional)
        Desired filename after download.

    TODO Handle and write tests for non-existing URLs, password-protected URLs, too large files, etc.
    """

    if not path.isdir(download_folder):
        print("Creating download folder:", download_folder)
        makedirs(download_folder)

    # Existing downloads are indicated by a hashfile generated from the URL,
    # which includes the filename of the actual image. This is a workaround to
    # deal with the fact that we do not know the filetype prior to downloading,
    # so we cannot determine the suffix in advance.
    hashfile = download_folder + '/' + str(hashlib.sha256(str.encode(url)).hexdigest())
    if path.exists(hashfile):
        with open(hashfile, 'r') as f:
            filename = f.read()
            if path.exists(filename):
                print("Loading from cache:", filename)
                return filename

    # No valid hash and corresponding file found - need to download
    print('Downloading from', url)
    req = requests.get(url)
    if req is not None and req.status_code == 200:
        if targetname is not None:
            filename = download_folder + "/" + targetname
        elif 'X-Object-Meta-Orig-Filename' in req.headers:
            filename = download_folder + '/' + req.headers['X-Object-Meta-Orig-Filename']
        else:
            filename = path.basename(url)
        with open(filename, 'wb') as code:
            code.write(req.content)
        print("Filename is", filename)
        suffix = path.splitext(filename)[-1]
        if (suffix == ".zip") and (ziptarget is not None):
            filename = get_from_zip(
                    filename, ziptarget, download_folder)
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


def get_from_zip(zipfile, ziptarget, targetdirectory):
    # Extract temporary zip file
    # TODO catch problem if file is not a nifti
    with ZipFile(zipfile, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = path.basename(zip_info.filename)
            if zip_info.filename == ziptarget:
                tmpdir = mkdtemp()
                zip_ref.extract(zip_info, targetdirectory)
                return targetdirectory + '/' + zip_info.filename


def get_json_from_url(url):
    req = requests.get(url)
    if req is not None and req.status_code == 200:
        return json.loads(req.content)
    else:
        return {}


if __name__ == '__main__':
    __check_and_get_token()
