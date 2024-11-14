import siibra


def test_sparsemap_volume_assignment():
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9",
        space="mni152",
        maptype="statistical"
    )
    difumo_maps = siibra.get_map(
        parcellation='difumo 64',
        space='mni152',
        maptype='statistical'
    )
    region = "fusiform posterior"
    volume = difumo_maps.volumes[difumo_maps.get_index(region).volume]
    assignments = julich_pmaps.assign(volume)
    significants_assignments = assignments.query('correlation >= 0.35')
    assert len(significants_assignments) == 2
    assert all(
        assigned_region in significants_assignments['region'].values
        for assigned_region in ["Area FG2 (FusG) left", "Area FG2 (FusG) right"]
    )
