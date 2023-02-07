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

We can use Parcellation objects to find child brain regions.
"""

# %%
# We start by selecting an atlas and a parcellation.
import siibra
atlas = siibra.atlases.get('human')
julich_brain = atlas.parcellations.get('julich 2.9')
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/01-003.png'

# %%
# The most basic way is to search for all regions matching a particular string:
julich_brain.find('V1')

# %%
# You can filter the children and querying the root node by:
julich_brain.find('v1', filter_children=True)

# %%
# For more powerful searches, regular expressions can be used with '/<pattern>/<flags>' or 
# using `re.compile()`. Find hOc2 or hOc4 in the right hemisphere:
julich_brain.find('/hOc[24].*right/')

# %%
# Searching for more general brain regions, we see that areas often appear
# three times: Julich-Brain defines them separately for the left and right
# hemisphere, and additionally defines a common parent region.
for r in julich_brain.find('amygdala'):
    print(r.name)

# %%
# Regions can also be search right away from the atlas object.
# However, it will return matching regions from all its known parcellations.
# search all regions known by the atlas
for r in atlas.find_regions('amygdala'):
    print(f"{r.name:30.30} {r.parcellation}")


# %%
# In fact, siibra provides a package-level function
# to search through regions of all parcellations.
siibra.find_regions('amygdala')

# %%
# Often however, we want to access one particular region, given a unique specification,
# and not obtain a list of many possible matches. This can be done using the
# `get_region` method. It assumes that the provided region specification is
# unique, and returns the single exact match.
# Note that if the specification is not unique, this method will raise an exception!
julich_brain.get_region('v1 left')

# %%
# siibra provides a package-level shortcut function for this as well:
siibra.get_region('julich 2.9', 'v1 left')

# %%
# In case that the given specification matches multiple regions, which however represent
# the children of the same parent, `get_region` will return the parent object. 
# In that case, the returned region can be a full subtree:
julich_brain.get_region('amygdala')

# %%
# Atlas objects provide direct access to the `get_region()` method of their
# parcellations. This way the above can also be done without explicitly
# accessing the parcellation object:
atlas.get_region('amygdala', parcellation='julich')

