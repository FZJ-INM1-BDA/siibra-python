import siibra


def test_sparsemap_cache_uniqueness():
    mp157 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="157")
    mp175 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="175")
    assert mp157.sparse_index.probs[0] != mp175.sparse_index.probs[0]
