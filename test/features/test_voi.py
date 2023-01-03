import pytest
import numpy as np
from siibra.features.basetypes.volume_of_interest import VolumeOfInterestQuery, VolumeOfInterest


query = VolumeOfInterestQuery()


@pytest.mark.parametrize("feature", query.features)
def test_voi_features(feature: VolumeOfInterest):
    model = feature.to_model()
    import re

    assert re.match(
        r"^[\w/\-.:]+$", model.id
    ), f"model_id should only contain [\\w/\\-.:]+, but is instead {model.id}"


def test_pli_volume_transform():
    feat = [f for f in query.features if "3D-PLI" in f.name]
    assert len(feat) == 1, "expecting 1 PLI data"  # may need to fix in future
    feat = feat[0]
    assert all(
        (
            np.array(vol.detail.get("neuroglancer/precomputed").get("transform"))
            == vol.transform_nm
        ).all()
        for vol in feat.volumes
    ), "expecting transform in neuroglance/precomputed be adopted as transform_nm, but was not."

    assert any(
        vol.url
        == "https://neuroglancer.humanbrainproject.eu/precomputed/data-repo/HSV-FOM"
        for vol in feat.volumes
    ), "Expect RGB PLI volume to be present"

    assert (
        len(feat.volumes) > 1
    ), "expecting more than 1 volume (incl. blockface, MRS label etc)"
