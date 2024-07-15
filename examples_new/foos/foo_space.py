import siibra
from siibra.factory import iter_collection
from siibra.atlases import Space

for space in iter_collection(Space):
    print(space, space.name)
    for pub in space.publications:
        print(f"- [{pub.text}]({pub.value})")

print("getting a single space")
# siibra.get_space("big brain")
# would not work
space = siibra.get_space("bigbrain")
print(space.name)
