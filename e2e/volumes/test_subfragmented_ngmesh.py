import pytest
import siibra

from siibra.volumes import Map

# a map is subfragemented if the parcellation and map regions are not fragmented
# but the data is stored fragmented
maps_w_subfragments = [
    (
        siibra.get_map(parcellation="isocortex", space="bigbrain"),
        {"left": 163842, "right": 163842},
    ),
]


@pytest.mark.parametrize("siibramap, frag_info", maps_w_subfragments)
def test_fetch_subfragmented(siibramap: Map, frag_info: int):
    assert len(siibramap.fragments) == 0
    for r in siibramap.regions:
        for k in frag_info:
            assert len(siibramap.fetch(region=r, format="mesh", fragment=k)["verts"]) == frag_info[k]

        assert len(siibramap.fetch(region=r, format="mesh")["verts"]) == sum(
            frag_info.values()
        )

    assert (
        len(siibramap.fragments) == 0
    )  # after fetching, since map index was altered before, this was returning fragments.
