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
Understanding links between data features and anatomical locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

All data features are requested with the same function, ``siibra.get_features()``.
The type of feature (spatial, regional, or parcellation feature), and thus the way it is linked to an anatomical concept depends on the requested modality.
This example shows how some features are linked to anatomical regions in different ways.
""" 

import siibra


# %%
# Available modalities are defined in the registry ``siibra.modalities``. 
# Most of these represent specifically supported data modalities, and will
# be covered one by one in the next examples. 
for m in siibra.modalities:
   print(m)

# %%
# As a first example, we use brain region V2 to query for "EbrainsRegionalDataset" features.
# See see :ref:`ebrains_datasets` for more information about this modality.
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
region = atlas.get_region("v2")
features = siibra.get_features(region, siibra.modalities.EbrainsRegionalDataset)
for feature in features:
   print(f" - {feature.name}")

# %%
# Since these features represent EBRAINS datasets which are linked to brain regions, 
# they are instances of both the "RegionalFeature" class, the "EbrainsDataset" class.
print(
   f"The last returned feature was of type '{feature.__class__.__name__}', "
   f"derived from {' and '.join(b.__name__ for b in feature.__class__.__bases__)}."
)

# %%
# Mappings to brain regions can have different precision: 
# A feature may be associated with the exact region requested,
# but it may also be linked to a child region, similar region, or parent region.
# To understand the link between the query region and each requested feature, 
# ``siibra`` adds some information about the match after running the query. In particular,   
# 
#  - ``Feature.match_qualification`` returns a rating of the matching that was applied, and could be any of "exact", "approximate", "contained", "contains"
#  - ``Feature.match_description`` often contains a human-readable description about the matching
#  - ``Feature.matched_region`` returns the region object to which the feature was last matched (if any)
#  - ``Feature.matched_parcellation`` returns the parcellation object to which the feature was last matched
#  - ``Feature.matched_location`` returns the spatial primitive to which the feature was matched (if any)
#
# Let's look at a few examples. We start with a rather broad specification - the occipital lobe.
# Here we hardly encounter an exact match, but many contained and approximately corresponding features.
# We print only some of the returned datasets.
selected_ids = [
   "54996ef6-b821-447a-b34c-7f2512396c8c",
   "8cd8506e-3a6f-406e-92c8-b52dda1dea23",
   "01550c5c-291d-4bfb-a362-875fcaa21724",
   "7269d1a2-c7ad-4745-972c-10dbf5a022b7",
]
occ = atlas.get_region('occipital')
for f in siibra.get_features(occ, 'ebrains'):
   if any(f.id.endswith(i) for i in selected_ids):
      print(f"{f.name}\n -> {f.match_description}\n")

# %%
# If we query a rather specific region, we get more exact matches.
v1 = atlas.get_region('v1')
for f in siibra.get_features(v1, 'ebrains'):
   print(f" - {f.name}\n {f.id}  {f.match_description}")

# %%
# Note that the same data feature might have different qualifications when returned from different queries.
# Let's take the following dataset with fMRI data in the primary visual cortex:
dataset = "https://nexus.humanbrainproject.org/v0/data/minds/core/dataset/v1.0.0/de7a6c44-8167-44a8-9cf4-435a3dab61ed"

# %%
# When querying for datasets related to V1, the match is qualified as "exact". 
# (Note that we group the results by dataset, so we can access the selected dataset directly.)
for f in siibra.get_features(v1, 'ebrains', group_by='dataset')[dataset]:
   print(f.name)
   print(f.match_description)

# %%
# When querying for datasets related to only the left hemisphere of v1, the match is qualified as "contains":
v1l = atlas.get_region('v1 left')
for f in siibra.get_features(v1l, 'ebrains', group_by='dataset')[dataset]:
   print(f.match_description)

# %%
# Lastly, when querying for datasets related to the occipial cortex, the match is qualified as "contained":
for f in siibra.get_features(occ, 'ebrains', group_by='dataset')[dataset]:
   print(f.match_description)

# %%
# There are also data features which are linked to locations in a specific reference space.
# A typical example are contact points from physiological recordings.
sessions = siibra.get_features(v1, siibra.modalities.IEEG_Session)
sessions[0].electrodes[0].contact_points[0]
print(features[0]._match)

