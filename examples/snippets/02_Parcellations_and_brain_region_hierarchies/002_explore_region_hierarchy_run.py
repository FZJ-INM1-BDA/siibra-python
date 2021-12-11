"""
Exploring the brain region hierarchy of a parcellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

