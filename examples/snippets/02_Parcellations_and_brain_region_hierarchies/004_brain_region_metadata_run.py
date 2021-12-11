"""
Basic brain region properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` makes no distinction between brain regions and region trees: Each
`siibra.core.Region` object represents itself a subtree with a (possibly empty)
set of child regions, and has a pointer to its parent region in the hierarchy.
It also has a pointer to the parcellation it belongs to.
"""

# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# Let's fetch the region representing the primary visual cortex.
v1 = atlas.get_region('v1', parcellation='julich')
v1

# %%
# The region belongs to a particular parcellation object:
v1.parcellation

# %%
# The primary visual cortex is part of the occipital cortex:
v1.parent

# %%
# It represents a subtree, with its children being the respective areas on each
# hemisphere:

# show the tree representation
print(repr(v1))

# return the actual list of child region objects
v1.children

# fetch one child
v1l = v1.children[0]

# %%
# Regions contain metadata. In some cases, they even represent individual EBRAINS datasets.
# In this case we can retrieve more detailed information from the EBRAINS Knowledge Graph.

# print some metadata of the brain region
for infodata in v1l.infos:
    print(infodata.description); print()
    for p in infodata.publications:
        print(p['cite']); print()

