import siibra
from siibra.volumes import Volume
from siibra.core.region import Region
import numpy as np
import pytest
# TODO write a test for the volume-region and volume-volume intersection


# add more to the list when centroids can be calculated for non-convex regions as well
selected_regions = [
    (siibra.get_region('julich 2.9', 'CA2 (Hippocampus) right'), 'mni152'),
    (siibra.get_region('julich 2.9', 'CA2 (Hippocampus) left'), 'colin27'),
    (siibra.get_region('julich 2.9', 'hoc1 left'), 'mni152'),
    (siibra.get_region('julich 3', 'superior temporal sulcus'), 'mni152'),
]


@pytest.mark.parametrize("region, space", selected_regions)
def test_region_intersection_with_its_own_volume(region, space):
    assert isinstance(region, Region)
    volume = region.get_regional_mask(space)
    intersection = region.intersection(volume)
    assert isinstance(intersection, Volume)
    assert np.all(
        np.equal(intersection.fetch().dataobj, volume.fetch().dataobj)
    ), "Intersection of a regional map with its region object should be the same volume."


@pytest.mark.parametrize("region, space", selected_regions)
def test_region_intersection_with_its_centroid(region, space):
    assert isinstance(region, Region)
    centroids = region.compute_centroids(space)
    assert centroids in region
    assert region.intersection(centroids) == centroids
    assert centroids.intersection(region) == centroids
