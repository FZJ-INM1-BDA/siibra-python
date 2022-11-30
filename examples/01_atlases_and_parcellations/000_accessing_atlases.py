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

Selecting a predefined atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides a registry of predefined atlas objects of different species. 
"""

# %%
# Predefined atlas objects are stored in `siibra.atlases`, which is just a shortcut to
# the object registry `siibra.REGISTRY.Atlas`. This container of predefined objects
# will be populated at the first time you request it. This will require an internet connection 
# to retrieve the configuration information. After the initial call, the atlas definitions are cached 
# on the local disk, so they can be accessed offline from then on.
import siibra

# %% 
# `siibra.atlases` is an instance table, a siibra object which allows 
# to access objects by name and autocompletion.
type(siibra.atlases)

# %%
# siibra uses instance tables not only for atlases, but also for parcellations
# (`siibra.parcellations`), reference spaces (`siibra.spaces`) or data
# modalities (`siibra.modalilities`). These will be adressed in subsequent
# examples. 
#
# Objects stored in a siibra instance table can be accessed in
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


# %%
