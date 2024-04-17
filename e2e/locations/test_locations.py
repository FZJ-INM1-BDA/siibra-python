import siibra


def test_intersection_ptset_outside_bbox():
    pts = siibra.PointSet(
        [[-9.184, -85.865, 2.414], [-9.184, -99.865, 3.114]], space='mni152'
    )
    bbox = siibra.get_region('julich 3', 'hoc1 right').get_boundingbox('mni152')
    assert pts.intersection(bbox) is None

    siibra.core.structure.BrainStructure._ASSIGNMENT_CACHE = dict()
    assert bbox.intersection(pts) is None
