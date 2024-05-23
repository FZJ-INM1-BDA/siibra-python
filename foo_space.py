import siibra
from siibra.assignment import iterate
from siibra.atlases import Space

for space in iterate(Space):
    print(space, space.name)
