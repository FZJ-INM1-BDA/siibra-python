import siibra
from siibra.assignment import iterate
from siibra.atlases import Parcellation

for parc in iterate(Parcellation):
    print(parc, parc.name)

print("getting a single parcellation")
parcellation = siibra.get_parcellation("2.9")
print(parcellation.name)