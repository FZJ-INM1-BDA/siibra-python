import pytest
import numpy as np
import siibra


vois = siibra.features.fibres.PLIVolumeOfInterest.get_instances()


def test_pli_volume_transform():

    modalities = [v.modality for v in vois]
    print(modalities)
    for word in ['transmittance', 'fibre orientation map']:
        assert any(word in n for n in modalities)

    feat = [f for f in vois if "fibre orientation map" in f.modality]
    assert len(feat) == 1, "expecting 1 FOM volume"  # may need to fix in future
    feat = feat[0]

    assert any(
        p._url
        == "https://neuroglancer.humanbrainproject.eu/precomputed/data-repo/HSV-FOM"
        for p in feat._providers.values()
    ), "Expect RGB PLI volume to be present"


