import unittest
import siibra


class TestretrievalDownloadFile(unittest.TestCase):
    def test_download_zipped_file(self):
        url = "https://data-proxy.ebrains.eu/api/v1/buckets/d-37258979-8b9f-4817-9d83-f009019a6c38/Semi-quantitative-analysis-siibra-csv.zip"
        ziptarget = "F9-BDA.csv"
        loader = siibra.retrieval.requests.ZipfileRequest(url, ziptarget, refresh=True)
        self.assertIsNotNone(loader.data)


if __name__ == "__main__":
    unittest.main()
