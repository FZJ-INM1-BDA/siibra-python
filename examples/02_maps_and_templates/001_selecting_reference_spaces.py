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
Find predefined reference spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as atlas and parcellation objects, `siibra` provides a registry of predefined
reference space objects that will be bootstrapped when you load the library for
the first time, and stay in your local file cache for future use. Reference
spaces are semantic objects which define a brain coordinate system. Associated
to each reference space are one or more reference templates, representing a
concrete reference image or surface representation of the brain.
"""

# %%
# As for atlases and parcellations, siibra provides a registry of predefined
# reference spaces:
import siibra
dir(siibra.spaces)

# %%
# Fetching an object works in the same way as for e.g. `siibra.atlases` (see :ref:`atlases`)
siibra.spaces['mni152']

# %%
# Typically however, we are only interested in the reference spaces supported by
# a given atlas. Atlases provide their own space registry for this
# purpose, which includes the relevant subset of the spaces.
atlas = siibra.atlases['human']
dir(atlas.spaces)

# %%
# These can be used like any other registry object:
atlas.spaces['colin27']

# %%
# We can also explicitely request a supported space object from the
# atlas, which has the same effect as accessing the Registry.
atlas.get_space('bigbrain')

