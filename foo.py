from siibra.features_beta.loaders import get_instances
import sys

unfiltered = [
    f for f in get_instances()
]
print("unfiltered", len(unfiltered))

filtered = [
    f for f in get_instances() if f.filter(name="gaba")
]
print("unfiltered", len(unfiltered), "filtered",  len(filtered))

print(
    "\n".join([f.name for f in filtered])
)

