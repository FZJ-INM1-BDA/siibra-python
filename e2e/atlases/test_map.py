import siibra


def test_all_icbm152():
    found_maps = siibra.find_maps(space="icbm 152")
    assert len(found_maps) > 0


def test_all_jba29():
    found_maps = siibra.find_maps(parcellation="2.9")
    assert len(found_maps) > 0
