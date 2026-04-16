import numpy as np
import nibabel
import siibra


def create_synthetic_data():
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9",
        space="mni152",
        maptype="statistical"
    )
    length = 20
    template_img = julich_pmaps.space.get_template().fetch()
    arr = np.zeros(list(template_img.shape) + [length])
    for i, img in enumerate(julich_pmaps.fetch_iter()):
        if i == length:
            break
        arr[:, :, :, i] += img.dataobj
    return siibra.volumes.from_nifti(
        nibabel.nifti1.Nifti1Image(arr, affine=template_img.affine),
        time_index=np.asanyarray(range(length)),
        space='mni152',
        name="synthetic timeseries volume"
    )


def test_timeseries_volume_assignment():
    difumo128 = siibra.get_map(
        parcellation="difumo 128",
        space="mni152",
        maptype="statistical"
    )
    synthetic_vol = create_synthetic_data()
    assignments = difumo128.assign(synthetic_vol, split_components=False)
    print(assignments)
