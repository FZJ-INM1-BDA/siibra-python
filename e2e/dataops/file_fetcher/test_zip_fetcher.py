import pytest
from tempfile import mkstemp
import zipfile
import os

from siibra.operations.file_fetcher.zip_fetcher import ZipDataOp
from siibra.operations.base import DataOp

hcp_zip_url = "https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000059_Atlas_based_HCP_connectomes_v1.1_pub/096-HarvardOxfordMaxProbThr0.zip"
hcp_zip_filename = (
    "096-HarvardOxfordMaxProbThr0/2FunctionalConnectivity/000/EmpCorrFC_REST1-RL.csv"
)


def test_zipdataop():
    dataop = ZipDataOp.generate_specs(url=hcp_zip_url, filename=hcp_zip_filename)
    runner = DataOp.get_runner(dataop)
    assert isinstance(runner, ZipDataOp)
    result = runner.run(None, **dataop)
    assert isinstance(result, bytes)
    text = result.decode("utf-8")


@pytest.fixture
def local_zipfile():
    _, filename = mkstemp(".zip")
    zf = zipfile.ZipFile(filename, "w")
    zf.writestr("test.txt", "foobar")
    zf.close()
    yield filename
    os.unlink(filename)


def test_local_dataop(local_zipfile):
    dataop = ZipDataOp.generate_specs(url=local_zipfile, filename="test.txt")
    runner = DataOp.get_runner(dataop)
    assert isinstance(runner, ZipDataOp)
    result = runner.run(None, **dataop)
    assert result == b"foobar"
