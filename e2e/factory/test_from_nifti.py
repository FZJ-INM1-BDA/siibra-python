import pytest
import nibabel as nib
import numpy as np
from siibra.factory import imageprovider_from_nifti


ICBM152_SPACE_ID = (
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
)


@pytest.fixture
def dummy_nifti():
    yield nib.Nifti1Image(np.random.random(10).reshape(5, 2, 1), affine=np.identity(4))


def test_imageprovider_from_nifti(dummy_nifti):
    image_prov = imageprovider_from_nifti(dummy_nifti, space="mni 152")
    assert image_prov.space_id == ICBM152_SPACE_ID
