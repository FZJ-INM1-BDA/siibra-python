# Copyright 2018-2021
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
.. _atlases:

Selecting a preconfigured atlas from an instance table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides a table of preconfigured atlas objects of
different species.
Atlas objects are very simple structures: They are mostly used to group
a set of preconfigured parcellations and reference spaces by species.
They do not provide much further functionality on their own.
"""

# %%
# We start by loading the library
import siibra

# %%
# Preconfigured atlas objects are accessible via the instance table `siibra.atlases`,
# which is a shortcut to `siibra.Atlas.registry()`.
# Instance tables are simple container structures, populated with preconfigured objects
# when accessed for the first time. The first call will require an internet connection to
# retrieve the initial configuration information. After initial access, the configuration
# information will cached on the local disk and can be accessed offline.
type(siibra.atlases)

# %%
# siibra uses instance tables not only for atlases, but also for parcellations
# (`siibra.parcellations`), reference spaces (`siibra.spaces`) or feature
# modalities (`siibra.modalilities`), as will be shown in later code examples.
#
# Objects stored in a siibra instance table can be accessed in
# different ways:
#
#  1. In "list-style": by iterating over all objects or using the index operator "[]"
#  2. By fuzzy keyword matching via the get() function or index operator
#  3. By tab-completion of their "keys"

# We can print the keys of all predefined atlas objects in the registry:
dir(siibra.atlases)

# %%
# The keys can be used to select atlas object by autocompletion
# when you hit <TAB>. This makes it convenient to find the right name
# in an interactive shell.
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
atlas

# %%
# We can also select an atlas by specifying the key as a string
# in the get() method:
siibra.atlases.get("MULTILEVEL HUMAN ATLAS")

# %%
# More importantly, we can use an arbitrary set of words
# matching the name or key of atlas. This is usually the simplest way
# to select from a Registry. It will fail if no unique match is found.
siibra.atlases.get('human')

# %%
# Of course, ass in a list, we can also iterate over the objects in the
# instance table or use the index operator to fetch an objects by its position:
for atlas in siibra.atlases:
    print(atlas.name)
print(siibra.atlases[0])

# %%
# Fuzyy string matching also works in the index operator.
# For legibility however, we typically prefer the "get()" form
# in our code examples.
siibra.atlases['human']

# %%
# An atlas has a range of properties and functions, for example is linked to a species:
atlas.species

# %%
# Furthermore, an atlas provides its own registries of supported spaces and parcellations.
# We will cover these in the next examples.
dir(atlas.spaces)
