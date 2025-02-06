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
Basic brain region properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` makes no distinction between brain regions and region trees: Each
`Region` object represents a subtree with a (possibly empty)
set of child regions, and has a pointer to its parent region in the hierarchy.
As mentioned before, Parcellation objects are also special regions,
with no parent and additional functionalities.
Consequently, the parcellation of a region can be accessed as the region's
"root" attribute (but "parcellation" is also provided as a shortcut property to this)
"""

# %%
# Start by importing the package.
import siibra
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/default_thumbnail.png'

# %%
# Let's fetch the region from the Julich-Brain parcellation
# representing the primary visual cortex.
v1 = siibra.get_region('julich 2.9', 'v1')

# %%
# The corresponding parcellation is just the root region:
print(v1.root)
print(v1.parcellation)


# %%
# The primary visual cortex is part of the occipital cortex:
v1.parent

# %%
# It represents a subtree, with its children being the respective areas on each
# hemisphere:

# show the tree representation
v1.render_tree()

# return the actual list of child region objects
v1.children

# we can access children with fuzzy string matching using "find"
# as well as by their index
v1l = v1.find("left")
print(v1l)
