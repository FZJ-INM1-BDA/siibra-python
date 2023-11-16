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
Selecting preconfigured parcellations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as for atlas objects, `siibra` provides an instance table of preconfigured
brain parcellations.
"""

# %%
# We start by loading the library
import siibra
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/default_thumbnail.png'
# %%
# The instance table can list the keys of all parcellations predefined in `siibra`.
# Note that this includes parcellations for all supported atlases.
siibra.parcellations.keys

# %%
# Fetching an object works in the same way as for `siibra.atlases` (see :ref:`atlases`)
julich_brain = siibra.parcellations.get('julich')

# %%
# Parcellations typically refer to entities in the EBRAINS knowledge graph,
# modelled in the MINDS and openMINDS standards. `siibra` stores their
# identifiers. Note that this holds for spaces and other concepts as well, as
# will be seen in the next examples.
print(julich_brain.id)

# %%
# Typically however, we are only interested in the parcellations supported by
# a given atlas. Atlases provide their own parcellation table for this
# purpose, which includes the relevant subset of the parcellations.
atlas = siibra.atlases.get('human')
atlas.parcellations.keys

# %%
# These are instance tables, and can be used just like siibra.parcellations:
atlas.parcellations.get('julich')  # will return the latest version per default

# %%
# Note that this specification matched multiple objects. Since `siibra` was
# able to sort them, it returned the first in the list, which is the one with
# the newest version. We can of course refine the specification to fetch
# another version.
print(atlas.parcellations.get('julich').version)
atlas.parcellations.get('julich 1.18').version

# %%
# We can also explicitly request a supported parcellation object from the
# atlas, which has the same effect as accessing the Registry.
atlas.parcellations.get('Deep fibre bundles')

julich_brain = atlas.parcellations.get('julich 2.9')

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
print("Name:    ", julich_brain.name)
print("Id:      ", julich_brain.id)
print("Modality:", julich_brain.modality)
print()
print(julich_brain.description)
print()
for p in julich_brain.publications:
    print(p['citation'])
