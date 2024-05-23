import siibra
from siibra.assignment import iterate
from siibra.atlases import Space

for space in iterate(Space):
    print(space, space.name)

print("getting a single space")
# siibra.get_space("big brain")
# would not work
space = siibra.get_space("bigbrain")
print(space.name)