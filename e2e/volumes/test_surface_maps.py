"""
Surface maps label regions per hemisphere, so the same label commonly denotes two
different regions. These tests pin the behaviour that makes them usable anyway.
"""

import numpy as np
import pytest
import siibra

surface_maps = [
    siibra.get_map("julich 2.9", "fsaverage"),
    siibra.get_map("julich 3.1", "fsaverage"),
]


@pytest.mark.parametrize("siibramap", surface_maps)
def test_lookup_table_has_unique_indices(siibramap):
    """Without compression the labels repeat across hemispheres and the table is invalid."""
    lut = siibramap.to_BIDS_lookup_table()
    assert lut["index"].is_unique
    assert set(lut["name"]) == set(siibramap.regions)


@pytest.mark.parametrize("siibramap", surface_maps)
def test_compressed_labels_separate_the_hemispheres(siibramap):
    compressed = siibramap.compress()
    labels_per_fragment = {
        fragment: set(np.unique(
            compressed.volumes[0]._providers["gii-label"].fetch(fragment=fragment)["labels"]
        )) - {0}
        for fragment in compressed.fragments
    }
    assert len(labels_per_fragment) > 1
    left, right = labels_per_fragment.values()
    assert left.isdisjoint(right)


@pytest.mark.parametrize("siibramap", surface_maps)
def test_compressed_map_can_be_fetched_per_region(siibramap):
    """The compressed map is backed by gifti files siibra wrote itself."""
    compressed = siibramap.compress()
    mesh = compressed.fetch(compressed.regions[0])
    assert np.array_equal(np.unique(mesh["labels"]), [0, 1])


@pytest.mark.parametrize("siibramap", surface_maps)
def test_labels_without_a_region_are_dropped(siibramap):
    compressed = siibramap.compress()
    prov = compressed.volumes[0]._providers["gii-label"]
    observed = set()
    for fragment in compressed.fragments:
        observed |= set(np.unique(prov.fetch(fragment=fragment)["labels"]))
    unnamed = (observed - {0}) - compressed.labels
    assert not unnamed, f"{len(unnamed)} label(s) with no region survived compression: {sorted(unnamed)}"
