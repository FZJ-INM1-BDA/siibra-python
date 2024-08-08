import siibra
from siibra.attributes import locations

def test_intersection_ptset_outside_bbox():
    space = siibra.get_space("icbm 152")
    pts = locations.pointcloud.PointCloud(coordinates=[[-9.184, -85.865, 2.414], [-9.184, -99.865, 3.114]],
                                          space_id=space.ID)
    region = siibra.get_region("julich brain 3.0.3", "hoc1 right")
    
    xtersect = locations.ops.intersect(region.get_boundingbox(space.ID), pts)
    assert xtersect is None

