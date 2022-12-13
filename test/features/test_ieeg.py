import pytest
from siibra import parcellations
from siibra.features.legacy.ieeg import IEEG_Session, IEEG_SessionQuery, IEEGSessionModel

ieeg_query = IEEG_SessionQuery()

@pytest.mark.parametrize('ieeg', ieeg_query.features)
def test_ieeg_to_model(ieeg: IEEG_Session):
    model: IEEGSessionModel = ieeg.to_model()
    import re
    assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"
    assert model.dataset.metadata.accessibility is not None, f"expecting model.dataset.metadata.accessibility is defined"
    assert model.dataset.metadata.accessibility.get("@id") == "https://nexus.humanbrainproject.org/v0/data/minds/core/embargostatus/v1.0.0/3054f80d-96a8-4dce-9b92-55c68a8b5efd", f"expecting the accessibility of the dataset to be restricted access"

hoc1right = parcellations['2.9'].get_region('hoc1 right')
models = [f.to_model(roi=hoc1right, detail=True) for f in ieeg_query.features]

def test_some_ieeg_matches():
    assert any(
        m.in_roi for m in models
    ), f"expecting some ieeg sessions in hoc1 right"
    assert any(
        not m.in_roi for m in models
    ), f"expecting some ieeg sessions not in hoc1 right"

@pytest.mark.parametrize('ieeg_model', [m for m in models if m.in_roi])
def test_matched_session(ieeg_model: IEEGSessionModel):
    assert any(
        electrodes.in_roi for electrodes in ieeg_model.electrodes.values()
    ), f"expecting matched session to have some electrodes matched"

    assert any(
        contact_pts.in_roi
        for electrodes in ieeg_model.electrodes.values() if electrodes.in_roi
        for contact_pts in electrodes.contact_points.values()
    ), f"expecting matched session & electrodes to have some contact point matched"


@pytest.mark.parametrize('ieeg_model', [m for m in models if not m.in_roi])
def test_unmatched_session(ieeg_model: IEEGSessionModel):
    assert all(
        not electrodes.in_roi for electrodes in ieeg_model.electrodes.values()
    ), f"expecting unmatched session to have no electrodes matched"
    assert all(
        not contact_pts.in_roi
        for electrodes in ieeg_model.electrodes.values()
        for contact_pts in electrodes.contact_points.values()
    ), f"expecting unmatched session to have no contact point matched"
