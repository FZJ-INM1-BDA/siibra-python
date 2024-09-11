import siibra
import numpy as np
from tqdm import tqdm
from siibra.attributes.locations import Point
from siibra.atlases.sparsemap import SparseIndex
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path


sparse_map_retrieval = [
    (
        [-4.077, -79.717, 11.356],
        [
            {
                "Area hOc2 (V2, 18) - left hemisphere": 0.33959856629371643,
                "Area hOc1 (V1, 17, CalcS) - left hemisphere": 0.6118946075439453,
            }
        ],
    ),
]


@pytest.fixture(scope="session")
def freshlocal_jba29_icbm152():
    mp = siibra.get_map("2.9", "icbm 152", "statistical")
    spi = SparseIndex("icbm152_julich2_9", mode="w")

    progress = tqdm(total=len(mp.regionnames), leave=True)
    for regionname in mp.regionnames:
        image_provider = mp._extract_regional_map_volume_provider(regionname)

        spi.add_img(image_provider.get_data(), regionname)
        progress.update(1)
    progress.close()
    spi.save()
    yield SparseIndex("icbm152_julich2_9", mode="r")


@pytest.fixture(scope="session")
def remote_jba29_icbm152():
    remote_spi = SparseIndex(
        "https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/sparse-indices/mni152-jba29",
        mode="r",
    )
    yield remote_spi


@pytest.fixture(scope="session")
def savedlocal_jba29_icbm152(remote_jba29_icbm152):
    with TemporaryDirectory() as dir:
        local_file = Path(dir) / "foo"
        remote_jba29_icbm152.save_as(str(local_file))
        yield SparseIndex(str(local_file), mode="r")


spix_fxt_names = [
    pytest.param("savedlocal_jba29_icbm152", id="savedlocal"),
    pytest.param("remote_jba29_icbm152", id="remote"),
    pytest.param("freshlocal_jba29_icbm152", id="freshlocal"),
]


@pytest.mark.parametrize("spix_fxt_name", spix_fxt_names)
@pytest.mark.parametrize("pt_phys, expected_value", sparse_map_retrieval)
def test_spidx(pt_phys, expected_value, spix_fxt_name, request):
    spidx: SparseIndex = request.getfixturevalue(spix_fxt_name)
    space = siibra.get_space("icbm 152")
    pt = Point(coordinate=pt_phys, space_id=space.ID)
    affine = np.linalg.inv(spidx.affine)
    pt_voxel = pt.transform(affine)
    voxelcoord = np.array(pt_voxel.coordinate).astype("int")
    assert spidx.read([voxelcoord]) == expected_value
