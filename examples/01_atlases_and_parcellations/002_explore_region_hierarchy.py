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

# %%
# The region hierarchy is a tree structure. Its default representation is a printout of the tree.
julich_brain.regiontree

# %%
#  Each node is a `siibra.Region` object, including the root node. 
type(julich_brain.regiontree)

# %%
# We can iterate all children of a node. This can be used to iterate the
# complete tree from the rootnode.
[region.name for region in julich_brain.regiontree]

# %%
# We can also iterate only the leaves of the tree
[region.name for region in julich_brain.regiontree.leaves]

