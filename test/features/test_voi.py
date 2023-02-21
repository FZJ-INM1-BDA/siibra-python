import pytest
import numpy as np
import siibra


vois = siibra.features.image.VolumeOfInterest.get_instances()


def test_pli_volume_transform():

    voinames = [v.name for v in vois]
    for word in ['Transmittance', 'Blockface', 'fiber orientation map', 'Segmentation', 'T2-weighted MRI']:
        assert any(word in n for n in voinames)

    feat = [f for f in vois if "fiber orientation map" in f.name]
    assert len(feat) == 1, "expecting 1 FOM volume"  # may need to fix in future
    feat = feat[0]

    assert any(
        p._url
        == "https://neuroglancer.humanbrainproject.eu/precomputed/data-repo/HSV-FOM"
        for p in feat._providers.values()
    ), "Expect RGB PLI volume to be present"


