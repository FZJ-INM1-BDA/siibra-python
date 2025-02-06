# Copyright 2018-2025
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
Find predefined reference spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as atlas and parcellation objects, `siibra` provides an instance table of
preconfigured reference spaces that will be populated when accessed for
the first time, and stay in your local file cache for future use. Reference
spaces are purely semantic objects which define a brain coordinate system.

Associated to each reference space however are one or more reference templates,
representing a concrete reference image or surface representation of the brain.
These are demonstrated in the following example.
"""

# %%
# As for atlases and parcellations, siibra provides a registry of predefined
# reference spaces:
import siibra
siibra.spaces.keys
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/default_thumbnail.png'

# %%
# Fetching an object works in the same way as for e.g. `siibra.atlases`
# (see :ref:`atlases`)
space = siibra.spaces.get('icbm 2009c asym')
print(space)

# %%
# Typically however, we are only interested in the reference spaces supported by
# a given atlas. Atlases provide their own reference space table for this
# purpose, which includes the relevant subset of the spaces.
atlas = siibra.atlases.get('human')
dir(atlas.spaces)

# %%
# These can be used like any other registry object:
colin_space = atlas.spaces.get('colin27')
print(colin_space)

# %%
# We can also explicitly request a supported space object from the
# atlas, which has the same effect as accessing the Registry.
bigbrain_space = atlas.get_space('bigbrain')
print(bigbrain_space)
