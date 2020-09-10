from zipfile import ZipFile

import requests
import hashlib
import nibabel as nib
from os import path

# Ideas:
#
# this module can detect several flavors of download file specifactions, for example:
# - A nifti file on a standard http URL, no auth required -> just download it
# - A zip file on a standard http URL -> We need additional specificcation of the desired file in side the zip (needs to be reflected in the metadata scheme for e.g. spaces and parcellations)
# - a UID of data provider like EBRAINS -> need to detecto the provider, and use an auth key that is setup during package installation
#

_allowed_templates = [
    'mni_icbm152_t1_tal_nlin_asym_09c.nii',
    'colin27_t1_tal_lin.nii'
]
_tmp_directory = 'brainscapes_tmp'


def download_file(url, download_folder):
    """
    Downloads a file from a URL to local disk, and returns the filename

    TODO Handle and write tests for non-existing URLs, password-protected URLs, too large files, etc.
    """
    filename = download_folder + '/' + str(hashlib.sha256(str.encode(url)).hexdigest())
    if path.exists(filename):
        return filename
    else:
        req = requests.get(url)
        if req is not None and req.status_code == 200:
            if 'X-Object-Meta-Orig-Filename' in req.headers:
                filename = download_folder + '/' + req.headers['X-Object-Meta-Orig-Filename']
            with open(filename, 'wb') as code:
                code.write(req.content)
            return filename
        #     if 'X-Object-Meta-Orig-Filename' in req.headers:
        #         filename = download_folder + '/' + req.headers['X-Object-Meta-Orig-Filename']
        #     else:
        #         filename = download_folder + '/' + req.headers['X-Object-Meta-Orig-Filename']
        #     if not path.exists(filename):
        #         with open(filename, 'wb') as code:
        #             code.write(req.content)
        #     return filename
        # return None
        # throw error TODO
    '''
        - error on response status != 200
        - error on file read
        - Nibable error
        - handle error, when no filename header is set
        - error or None when space not known
        - unexpected error
        '''


def get_file_from_zip(zipfile):
    # Extract temporary zip file
    # TODO shall go to the data retrieval module
    with ZipFile(zipfile, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == '/':
                continue
            zip_info.filename = path.basename(zip_info.filename)
            if zip_info.filename in _allowed_templates:
                zip_ref.extract(zip_info, _tmp_directory)
                return nib.load(_tmp_directory + '/' + zip_info.filename)
