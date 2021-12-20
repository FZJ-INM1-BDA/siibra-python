"""
.. _atlases:

Selecting a predefined atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides a registry of predefined atlas objects of different species. 
"""

# %%
# The predefined atlas objects will be bootstrapped the first time a particular version
# of `siibra` is loaded, so an internet connection is required before accessing them for the
# first time. After the initial bootstrap, the atlas definitions are cached on the local
# disk, so they can be accessed offline.
import siibra

# %% 
# `siibra.atlases` is a Registry object: 
type(siibra.atlases)

# %%
# siibra uses registries not only for atlases, but also for parcellations
# (`siibra.parcellations`), reference spaces (`siibra.spaces`) or data
# modalities (`siibra.modalilities`). These will be adressed in subsequent
# examples. 
#
# Objects stored in a siibra Registry can be accessed in
# different ways:
#
#  1. You can iterate over all objects
#  2. An integer index gives sequential access to individual elements
#  3. A string index will be matched against the name of objects
#  4. Object keys can be tab-completed as attributes of the registry
#
# We can print the keys of all predefined atlas objects in the registry:
dir(siibra.atlases)

# %% 
# These keys can be used to fetch an atlas object, supported by autocompletion
# when you hit <TAB>. This makes it convenient to find the right name. 
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
atlas

# %%
# We can also select the atlas by specifying the key as an index:
siibra.atlases["MULTILEVEL HUMAN ATLAS"]

# %%
# We can fetch the first available atlas:
siibra.atlases[0]

# %%
# Most importantly, we can use arbitrary strings matching the name of atlas
# uniquely. This is usually the simplest way to select from a Registry.
siibra.atlases['human']

# %%
# An atlas has a range of properties and functions, for example is linked to a species:
atlas.species

# %%
# Furthermore, an atlas provides its own registries of supported spaces and parcellations. 
# We will cover these in the next examples.
dir(atlas.spaces)

