import requests
from os import path

# Ideas:
#
# this module can detect several flavors of download file specifactions, for example:
# - A nifti file on a standard http URL, no auth required -> just download it
# - A zip file on a standard http URL -> We need additional specificcation of the desired file in side the zip (needs to be reflected in the metadata scheme for e.g. spaces and parcellations)
# - a UID of data provider like EBRAINS -> need to detecto the provider, and use an auth key that is setup during package installation
#

def download_file(url,download_folder):
    """
    Downloads a file from a URL to local disk, and returns the filename

    TODO Handle and write tests for non-existing URLs, password-protected URLs, too large files, etc.
    """
    req = requests.get(url)
    if req is not None and req.status_code == 200:
        filename = download_folder + '/' + req.headers['X-Object-Meta-Orig-Filename']
        if not path.exists(filename):
            with open(filename, 'wb') as code:
                code.write(req.content)
        return filename
    return None
    # throw error TODO
    '''
        - error on response status != 200
        - error on file read
        - Nibable error
        - handle error, when no filename header is set
        - error or None when space not known
        - unexpected error
        '''


