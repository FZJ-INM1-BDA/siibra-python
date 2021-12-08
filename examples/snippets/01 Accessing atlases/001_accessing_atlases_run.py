"""
.. _atlases:

Accessing predefined atlases
~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides a registry of predefined atlas objects. 
These objects will be bootstrapped the first time you load particular version
of `siibra`, so you need an internet connection before accessing them for the
first time. After the initial bootstrap, these objects are cached on your
computer so you can access them offline. 
"""

# sphinx_gallery_thumbnail_path = "_static/demo.png"

import siibra

# %% 
# `siibra.atlases` is a Registry object: 
type(siibra.atlases)

# %%
# We can requests the keys of all predefined objects in the registry:
dir(siibra.atlases)

# %% 
# These keys can be used to fetch an atlas object. 
# Note that in an interpreter and most IDE's, these keys will autocomplete when
# you hit <TAB>. This makes it convenient to find the right name. 
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
atlas

# %%
# `siibra` Registry objects also offer the index operator to access their predefined
# objects. There are basically three ways.
# 1.) Using the above mentioned keys: 
siibra.atlases["MULTILEVEL HUMAN ATLAS"]

# %%
# 2.) Using integer numbers for sequential access:
siibra.atlases[0]

# %%
# 3.) Using arbitrary strings that match an object uniquely. The latter is the simplest way
# to select objects from a Registry.
siibra.atlases['human']

# %%
# An atlas has a range of properties and functions. In fact it can be used to
# access much of `siibra`'s functionality, which are addressed in other examples.
# As an example, an atlas is defined for a particular species:
atlas.species['name']

# %%
# Furthermore, an atlas provides registries of its supported spaces and parcellations. 
# We will cover these in the section :ref:`parcellations`.
dir(atlas.spaces)

