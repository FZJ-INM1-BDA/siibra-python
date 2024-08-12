import siibra
import numpy as np
from siibra.attributes.locations import Point
from siibra.atlases.sparsemap import SparseIndex


def test_remote_sparsemap():

    pt_phys = [-4.077, -79.717, 11.356]
    space = siibra.get_space("icbm 152")
    remote_spi = SparseIndex("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/icbm152_julich2_9", mode="r")
    pt = Point(coordinate=pt_phys, space_id=space.ID)
    affine = np.linalg.inv(remote_spi.affine)
    pt_voxel = pt.transform(affine)
    voxelcoord = np.array(pt_voxel.coordinate).astype("int")

    expected_value = [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]
    assert remote_spi.read([voxelcoord]) == expected_value

