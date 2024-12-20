import siibra
from siibra.volumes.sparsemap import SparseMap


def test_cache_prefixes():
    sparsemaps = [
        mp for mp in siibra.maps
        if isinstance(mp, siibra.volumes.sparsemap.SparseMap)
    ]
    assert len(sparsemaps) == len([mp._cache_prefix for mp in sparsemaps])


def test_sparsemap_cache_uniqueness():
    mp157 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="157")
    mp175 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="175")
    assert isinstance(mp157, SparseMap) and isinstance(mp175, SparseMap)
    assert mp157.sparse_index.probs[0] != mp175.sparse_index.probs[0]
