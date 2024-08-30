from siibra.operations.file_fetcher.zip_fetcher import ZipDataOp
from siibra.operations.base import DataOp

hcp_zip_url = "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000059_Atlas_based_HCP_connectomes_v1.1_pub/096-HarvardOxfordMaxProbThr0.zip"
hcp_zip_filename = (
    "096-HarvardOxfordMaxProbThr0/2FunctionalConnectivity/000/EmpCorrFC_REST1-RL.csv"
)


def test_zipdataop():
    dataop = ZipDataOp.from_url(hcp_zip_url, hcp_zip_filename)
    Runner = DataOp.get_runner(dataop)
    runner = Runner()
    assert Runner is ZipDataOp
    result = runner.run(None, **dataop)
    assert isinstance(result, bytes)
    text = result.decode("utf-8")
