"""
Finding brain regions in a parcellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use Parcellation objects to find individual brain regions.
"""


import siibra
atlas = siibra.atlases['human']
julich_brain = atlas.get_parcellation('julich 2.9')
print(julich_brain)

# %%
# The most basic way is to search for all regions matching a particular string:
julich_brain.regiontree.find('V1')

# %%
# For convenience, querying the root node can be done directly from the
# parcellation object:
julich_brain.find_regions('V1')

# %%
# For more fine grained searches, powerful regular expressions can be used. Refer to https://docs.python.org/3/library/re.html for more information about regular expression syntax.
import re
# find hOc2 or hOc4 in the right hemisphere
julich_brain.find_regions(re.compile('hOc[24].*right'))

# %%
# Searching for more general brain regions, we see that areas often appear
# three times: Julich-Brain defines them separately for the left and right
# hemisphere, and additionally defines a common parent region. 
for r in julich_brain.find_regions('amygdala'):
    print(r.name)

# %%
# Often however, we want to access one particular region, given a unique specification,
# and not obtain a list of many possible matches. This can be done using the 
# `decode_region` method. It assumes that the provided region specification is 
# unique, and attempts to return a single exact match, which might also be the most
# general common parent. Note that if the spec is not unique, it
# will raise an exception!

# region V1 in the left hemisphere, a leaf in the tree
julich_brain.decode_region('v1 left')

# the unique parent region of the amygdala subtree
julich_brain.decode_region('amygdala')

# %%
# Regions can also be search right away from the atlas object.
# However, it will return matching regions from all its known parcellations.
# search all regions known by the atlas
for region in atlas.find_regions('amygdala'):
    print(f"{region.name:30.30} {region.parcellation}")

