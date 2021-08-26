import os
import unittest
import siibra
from ..get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestRetrievalDownloadFile(unittest.TestCase):

    def test_download_zipped_file(self):
        url="https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/test_stefan_destination/MPM.zip"
        ziptarget="JulichBrain_v25.xml"
        loader = siibra.retrieval.requests.ZipfileRequest(url,ziptarget)
        self.assertIsNotNone(loader.data)
    # TODO Clear cache folder after test (for local testing)

if __name__ == "__main__":
    unittest.main()
