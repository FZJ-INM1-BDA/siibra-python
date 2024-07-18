from siibra.factory.livequery import brainglobe
import siibra

for atlas in brainglobe.list_all():
    print(atlas)



got_parc = siibra.get_parcellation("bg-allen_mouse_100um_v1.2")
got_space = siibra.get_space("bg-allen_mouse_100um_v1.2")

assert parc is got_parc
assert space is got_space
