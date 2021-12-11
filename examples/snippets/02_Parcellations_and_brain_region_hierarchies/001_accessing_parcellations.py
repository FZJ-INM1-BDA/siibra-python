"""
List predefined parcellations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as for atlas objects, `siibra` provides a general registry of predefined
parcellation objects. They will be bootstrapped when you load the library for
the first time, and stay in your local file cache for future use.
"""

# %%
# We start by loading the library
import siibra

# %%
# The registry can list the keys of all parcellations predefined in `siibra`. Note that this
# includes parcellations for all supported atlases.
dir(siibra.parcellations)

# %%
# Fetching an object works in the same way as for `siibra.atlases` (see :ref:`atlases`)
siibra.parcellations['julich']

# %%
# Typically however, we are only interested in the parcellatioins supported by
# a given atlas. Atlases provide their own parcellation registry for this
# purpose, which includes the relevant subset of the parcellations.
atlas = siibra.atlases['human']
dir(atlas.parcellations)

# %%
# These can be used like any other registry object:
atlas.parcellations['julich']

# %%
# Note that this specification matched multiple objects. Since `siibra` was
# able to sort them, it returned the first in the list, which is the one with
# the newest version. We can of course refine the specification to fetch
# another version.
print(atlas.parcellations['julich'].version)
atlas.parcellations['julich 1.18'].version

# %%
# We can also explicitely request a supported parcellation object from the
# atlas, which has the same effect as accessing the Registry.
atlas.get_parcellation('long bundles')

jubrain = atlas.get_parcellation('julich') # will return the latest version per default

# %%
# The resulting parcellation is a semantic object, including 
# 
# - general information like the parcellation name, a description and related publications
# - the region hierarchy
# - functions to access parcellation maps in different reference spaces
# - functions to find and access regions and region masks
# 
# Parcellation maps and brain  regions are covered in the next examples. For now let's
# just look at a few metadata fields:
print("Name:    ",jubrain.name)
print("Id:      ",jubrain.id)
print("Modality:",jubrain.modality)
print()
print(jubrain.description)
print()
for p in jubrain.publications:
    print(p['cite'])

