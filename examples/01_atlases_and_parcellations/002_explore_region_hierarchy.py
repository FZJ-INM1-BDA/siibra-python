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
Explore brain region hierarchies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each parcellation provides access to a tree of brain regions. 
"""

# %%
# We start by selecting an atlas and a parcellation
import siibra
atlas = siibra.atlases['human']
julich_brain = atlas.get_parcellation('julich 2.9')
julich_brain

# %%
# The region hierarchy is a tree structure. Its default representation is a string of the tree.
print(julich_brain.tree2str())

# %%
# Each node is a `siibra.Region` object, including the root node. 
# We can iterate all regions in a parcellation
[region.name for region in julich_brain]

# %%
# We can iterate the brain divisions in the parcellation
[region.name for region in julich_brain.children]

# %%
# We can also iterate only the leaves of the tree
[region.name for region in julich_brain.leaves]

