import siibra
import numpy as np
from tqdm import tqdm
from siibra.attributes.locations import Point
from siibra.atlases.sparsemap import SparseIndex
import pytest


sparse_map_retrieval = [
    ([-4.077, -79.717, 11.356], [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]),
]

@pytest.fixture(scope="session")
def freshlocal_jba29_icbm152():
    mp = siibra.get_map("2.9", "icbm 152", "statistical")
    spi = SparseIndex("icbm152_julich2_9", mode="w")

    progress = tqdm(total=len(mp.regions), leave=True)
    for regionname in mp.regions:
        volumes = mp.find_volumes(regionname)
        assert len(volumes) == 1
        volume = volumes[0]
        spi.add_img(volume.fetch(), regionname)
        progress.update(1)
    progress.close()
    spi.save()
    yield SparseIndex("icbm152_julich2_9", mode="r")

@pytest.fixture(scope="session")
def remote_jba29_icbm152():
    remote_spi = SparseIndex("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/icbm152_julich2_9", mode="r")
    yield remote_spi
    

@pytest.mark.parametrize("pt_phys, expected_value", sparse_map_retrieval)
def test_remote_sparsemap_jba29_icbm152(pt_phys, expected_value, remote_jba29_icbm152):
    space = siibra.get_space("icbm 152")
    pt = Point(coordinate=pt_phys, space_id=space.ID)
    affine = np.linalg.inv(remote_jba29_icbm152.affine)
    pt_voxel = pt.transform(affine)
    voxelcoord = np.array(pt_voxel.coordinate).astype("int")
    assert remote_jba29_icbm152.read([voxelcoord]) == expected_value


@pytest.mark.parametrize("pt_phys, expected_value", sparse_map_retrieval)
def test_freshlocal_sparsemap_jba29_icbm152(pt_phys, expected_value, freshlocal_jba29_icbm152):
    space = siibra.get_space("icbm 152")
    pt = Point(coordinate=pt_phys, space_id=space.ID)
    affine = np.linalg.inv(freshlocal_jba29_icbm152.affine)
    pt_voxel = pt.transform(affine)
    voxelcoord = np.array(pt_voxel.coordinate).astype("int")
    assert freshlocal_jba29_icbm152.read([voxelcoord]) == expected_value
