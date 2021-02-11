import os
import unittest
import brainscapes as bs

class TestRetrievalDownloadFile(unittest.TestCase):

    def test_download_file(self):
        bs.retrieval.download_file(
            "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/test_stefan_destination/MPM.zip",
            ziptarget="JulichBrain_v25.xml"
        )
    # TODO Clear cache folder after test (for local testing)

if __name__ == "__main__":
    unittest.main()
