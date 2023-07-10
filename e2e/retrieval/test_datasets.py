import siibra

dsv_id = "37258979-8b9f-4817-9d83-f009019a6c38"


def test_dataset_name():
    dsv = siibra.retrieval.datasets.EbrainsV3DatasetVersion(id=dsv_id)
    assert "None" not in dsv.name
    assert dsv.description
