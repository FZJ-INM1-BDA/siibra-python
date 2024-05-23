import siibra
from siibra.assignment import iterate
from siibra.atlases import Parcellation

for parc in iterate(Parcellation):
    print(parc, parc.name)
