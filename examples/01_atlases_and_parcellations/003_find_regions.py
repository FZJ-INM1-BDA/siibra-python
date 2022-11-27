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
Find brain regions in a parcellation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use Parcellation objects to find individual brain regions.
"""

# %%
# We start by seleting an atlas and a parcellation.
import siibra
atlas = siibra.atlases['human']
julich_brain = atlas.get_parcellation('julich 2.9')

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
# Regions can also be search right away from the atlas object.
# However, it will return matching regions from all its known parcellations.
# search all regions known by the atlas
for region in atlas.find_regions('amygdala'):
    print(f"{region.name:30.30} {region.parcellation}")


# %%
# Often however, we want to access one particular region, given a unique specification,
# and not obtain a list of many possible matches. This can be done using the 
# `get_region` method. It assumes that the provided region specification is 
# unique, and returns the single exact match. 
# Note that if the specification is not unique, this method will raise an exception!
julich_brain.get_region('v1 left')

# %%
# In case that the given specification matches multiple regions, which however represent
# the children of the same parent, `get_region` will return the parent object. 
# In that case, the returned region can be a full subtree:
julich_brain.get_region('amygdala')

# %%
# Atlas objects provide direct access to the `get_region()` method of their
# parcellations. This way the above can also be done without explictly
# accessing the parcellation object:
atlas.get_region('amygdala', parcellation='julich')

