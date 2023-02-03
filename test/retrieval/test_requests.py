import siibra
from siibra.retrieval.requests import EbrainsRequest

def test_device_flow():
    assert hasattr(EbrainsRequest, "device_flow")
