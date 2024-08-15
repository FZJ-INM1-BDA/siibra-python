import siibra
from siibra.factory.iterator import iter_preconfigured_ac
from siibra.atlases import ParcellationScheme

for parc in iter_preconfigured_ac(ParcellationScheme):
    print(parc, parc.name)
    for pub in parc.publications:
        print(f"[{pub.text}]({pub.value})")


print("getting a single parcellation")
parcellation = siibra.get_parcellation("2.9")
print(parcellation.name)

hoc1_left = parcellation.find("hoc1 left")
hoc1 = parcellation.find("hoc1")

assert len(hoc1_left) == 1
assert len(hoc1) == 3
