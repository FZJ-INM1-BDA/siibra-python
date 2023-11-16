import siibra


vois = siibra.features.fibres.PLIVolumeOfInterest.get_instances()


def test_pli_volume_transform():
    modalities = [v.modality for v in vois]
    print(modalities)
    for word in ["transmittance", "fibre orientation map"]:
        assert any(word in n for n in modalities)

    fibre_orient_feats = [f for f in vois if "fibre orientation map" in f.modality]
    assert len(fibre_orient_feats) == 2, "expecting 1 FOM volume"  # may need to fix in future

    assert any(
        p._url
        == "https://neuroglancer.humanbrainproject.eu/precomputed/data-repo/HSV-FOM"
        for f in fibre_orient_feats
        for p in f._providers.values()
    ), "Expect RGB PLI volume to be present"
