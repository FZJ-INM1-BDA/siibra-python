import pytest
import nibabel as nib
import numpy as np
from tempfile import mktemp
import hashlib
import os

from siibra.attributes.locations import BoundingBox, Point
from siibra.attributes.datarecipes.volume import ImageRecipe
from siibra.operations.volume_fetcher.nifti import NiftiExtractVOI
from siibra.operations.volume_fetcher.neuroglancer_precomputed import (
    NgPrecomputedFetchCfg,
)


JBA_31_ICBM152_LABELLED_URL = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-f1fe19e8-99bd-44bc-9616-a52850680777/maximum-probability-maps_MPMs_207-areas/JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.nii.gz"
JBA_HOC1_ICBM152_STATMAP = "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-f1fe19e8-99bd-44bc-9616-a52850680777/probabilistic-maps_PMs_207-areas/Area-hOc1/Area-hOc1_lh_MNI152.nii.gz"
NG_PRECOMP_BB_20UM = (
    "https://neuroglancer.humanbrainproject.eu/precomputed/BigBrainRelease.2015/8bit"
)


ICBM152_SPACEID = (
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
)

BIGBRAIN_ID = "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588"

# TODO update with link to data-proxy of the generated nii (n.b. affected by gzip compression level. Thus using default level)
EXPECTED_FILE_MD5_HASH = "257d7d1549f9dbff5622f9e4147f996a"


@pytest.fixture
def jba31_icbm152_labelled_improv():
    yield ImageRecipe(
        format="nii", url=JBA_31_ICBM152_LABELLED_URL, space_id=ICBM152_SPACEID
    )


@pytest.fixture
def jba_hoc1_lh_icbm152_stat_imgprov():
    yield ImageRecipe(
        format="nii",
        url=JBA_HOC1_ICBM152_STATMAP,
        space_id=ICBM152_SPACEID,
    )


@pytest.fixture
def bb_template():
    yield ImageRecipe(
        format="neuroglancer/precomputed", url=NG_PRECOMP_BB_20UM, space_id=BIGBRAIN_ID
    )


@pytest.fixture
def bb_test_bbox():
    yield BoundingBox(
        minpoint=[-21.152, 14, 48.92],
        maxpoint=[-12.246, 23, 53.302],
        space_id=BIGBRAIN_ID,
    )


@pytest.fixture
def hoc1_lh_bbox(jba31_icbm152_labelled_improv):
    data = jba31_icbm152_labelled_improv.get_data()
    assert isinstance(data, nib.Nifti1Image)
    xmin, ymin, zmin, xmax, ymax, zmax = [59, 23, 55, 99, 87, 108]

    bbox = BoundingBox(minpoint=[xmin, ymin, zmin], maxpoint=[xmax, ymax, zmax])
    yield bbox.transform(data.affine, space_id=jba31_icbm152_labelled_improv.space_id)


@pytest.fixture
def fp1_lh_bbox(jba31_icbm152_labelled_improv):

    data = jba31_icbm152_labelled_improv.get_data()
    assert isinstance(data, nib.Nifti1Image)
    xmin, ymin, zmin, xmax, ymax, zmax = [50, 177, 48, 98, 206, 112]

    bbox = BoundingBox(minpoint=[xmin, ymin, zmin], maxpoint=[xmax, ymax, zmax])
    yield bbox.transform(data.affine, space_id=jba31_icbm152_labelled_improv.space_id)


@pytest.fixture
def tmp_nii_filename():
    filename = mktemp(".nii.gz")
    yield filename
    os.unlink(filename)


def test_get_data_labelledmap(jba31_icbm152_labelled_improv):
    data = jba31_icbm152_labelled_improv.get_data()
    assert isinstance(data, nib.Nifti1Image)
    assert np.max(data.dataobj) == 207
    assert np.min(data.dataobj) == 0
    assert len(np.unique(data.dataobj)) == 208
    assert data.dataobj.shape == (193, 229, 193)


def test_get_data_statledmap(jba_hoc1_lh_icbm152_stat_imgprov):
    data = jba_hoc1_lh_icbm152_stat_imgprov.get_data()
    assert isinstance(data, nib.Nifti1Image)
    assert np.max(data.dataobj) < 1
    assert np.min(data.dataobj) == 0
    assert data.dataobj.shape == (193, 229, 193)


def test_get_data_labelledmap_voi(jba31_icbm152_labelled_improv, hoc1_lh_bbox):
    jba31_icbm152_labelled_improv.transformation_ops.append(
        NiftiExtractVOI.generate_specs(voi=hoc1_lh_bbox)
    )
    data = jba31_icbm152_labelled_improv.get_data()
    assert isinstance(data, nib.Nifti1Image)
    assert np.max(data.dataobj) < 207
    assert np.min(data.dataobj) == 0
    assert len(np.unique(data.dataobj)) < 208
    assert (
        91 in np.unique(data.dataobj).tolist()
    )  # bbox is of hoc1, 91 is label index of hoc1
    assert data.dataobj.shape == (40, 64, 53)


def test_get_data_statmap_voi(jba_hoc1_lh_icbm152_stat_imgprov, fp1_lh_bbox):
    jba_hoc1_lh_icbm152_stat_imgprov.transformation_ops.append(
        NiftiExtractVOI.generate_specs(voi=fp1_lh_bbox)
    )
    data = jba_hoc1_lh_icbm152_stat_imgprov.get_data()
    assert isinstance(data, nib.Nifti1Image)
    assert np.max(data.dataobj) < 0.01
    assert data.dataobj.shape == (48, 29, 64)


def test_ng_nifti_extractvoi(bb_template, bb_test_bbox, tmp_nii_filename):
    bb_template.transformation_ops.extend(
        [
            NiftiExtractVOI.generate_specs(voi=bb_test_bbox),
            NgPrecomputedFetchCfg.generate_specs(
                fetch_config={"max_download_GB": 1, "resolution_mm": 8e-2}
            ),
        ]
    )
    nii = bb_template.get_data()
    nii.to_filename(tmp_nii_filename)
    with open(tmp_nii_filename, "rb") as fp:
        assert hashlib.md5(fp.read()).hexdigest() == "257d7d1549f9dbff5622f9e4147f996a"


def test_ng_nifti_fetch_kwargs(bb_template, bb_test_bbox, tmp_nii_filename):
    bb_template.transformation_ops.append(
        NgPrecomputedFetchCfg.generate_specs(
            fetch_config={
                "max_download_GB": 1,
                "resolution_mm": 8e-2,
                "bbox": bb_test_bbox,
            }
        ),
    )
    nii = bb_template.get_data()
    nii.to_filename(tmp_nii_filename)
    with open(tmp_nii_filename, "rb") as fp:
        assert hashlib.md5(fp.read()).hexdigest() == "257d7d1549f9dbff5622f9e4147f996a"
