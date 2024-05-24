import siibra
from siibra.assignment import iterate
from siibra.atlases import Parcellation

for parc in iterate(Parcellation):
    print(parc, parc.name)

print("getting a single parcellation")
parcellation = siibra.get_parcellation("2.9")
print(parcellation.name)

hoc1_left = parcellation.find("hoc1 left")
hoc1 = parcellation.find("hoc1")

assert len(hoc1_left) == 1
assert len(hoc1) == 3
