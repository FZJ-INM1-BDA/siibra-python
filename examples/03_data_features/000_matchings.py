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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

All data features are requested with the same function, ``siibra.features.get()``.
The type of feature (spatial, regional, or parcellation feature), and thus the way it is linked to an anatomical concept depends on the requested modality.
This example shows how some features are linked to anatomical regions in different ways.
"""

# %%
import siibra


# %%
# Available feature types are listed in the module `siibra.features`.
# Most of these represent specifically supported data modalities, and will
# be covered one by one in the next examples.
print(siibra.features.Feature.SUBCLASSES) # TODO: This is a temporary fix. 

# %%
# Regional features
# ^^^^^^^^^^^^^^^^^
#
# As a first example, we use brain region V2 to query for "EbrainsRegionalDataset" features.
# See :ref:`ebrains_datasets` for more information about this modality.
v2 = siibra.get_region("julich 2.9", "v2")
features = siibra.features.get(v2, siibra.features.external.EbrainsDataFeature)
for feature in features:
    print(f" - {feature.name}")

# %%
# Mappings to brain regions can have different precision:
# A feature may be associated with the exact region requested,
# but it may also be linked to a child region, similar region, or parent region.
# To understand the link between the query region and each requested feature,
# ``siibra`` adds some information about the match each time a query is executed. In particular,
#
#  - ``Feature.match_qualification`` returns a rating of the matching that was applied, and could be any of "exact", "approximate", "contained", "contains"
#  - ``Feature.match_description`` often contains a human-readable description about the matching
#
# Furthermore, if applicable,
#  - ``Feature.matched_region`` returns the region object to which the feature was last matched,
#  - ``Feature.matched_parcellation`` returns the parcellation object to which the feature was last matched, and
#  - ``Feature.matched_location`` returns the spatial primitive to which the feature was matched.
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
occ = siibra.get_region("julich 2.9", "occipital")
for f in siibra.features.get(occ, "ebrains"):
    if any(_ in f.id for _ in selected_ids):
        print(f.name)
        print(" -> ", f.last_match_description)
        print()

# %%
# If we query a rather specific region, we get more exact matches.
v1 = siibra.get_region("julich 2.9", "v1")
for f in siibra.features.get(v1, "ebrains"):
    if any(_ in f.id for _ in selected_ids):
        print(f.name)
        print(" -> ", f.last_match_description)
        print()

# %%
# Qualifications differ across queries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that the same data feature might have different qualifications when returned from different queries.
# Let's take the following dataset with fMRI data in the primary visual cortex:
dataset_id = "de7a6c44-8167-44a8-9cf4-435a3dab61ed"

# %%
# When querying for datasets related to V1, the match is qualified as an exact coincidence.
for f in siibra.features.get(v1, "ebrains"):
    if f.id.endswith(dataset_id):
        print(f.name)
        print(" -> ", f.last_match_description)
        print()

# %%
# When querying for datasets related to only the left hemisphere of v1,
# the match is qualified as "contains":
v1l = siibra.get_region("julich 2.9", "v1 left")
for f in siibra.features.get(v1l, "ebrains"):
    if dataset_id in f.id:
        print(f.name)
        print(" -> ", f.last_match_description)
        print()

# %%
# Lastly, when querying for datasets related to the occipial cortex, the match is qualified as "contained":
for f in siibra.features.get(occ, "ebrains"):
    if f.id.endswith(dataset_id):
        print(f.last_match_description)

# %%
# Spatial features
# ^^^^^^^^^^^^^^^^
#
# There are also data features which are linked to locations in a specific reference space.
# For example, volumes of interest are tested for overlap with bounding boxes of regions.
# The following dataset covering the hippocampus intersects with the mask of CA1.
# Note how `siibra` deals with the fact that the volume of interest is defined in BigBrain space,
# while the region is only mapped in the MNI spaces - it warps the bounding box
# of the region to the space of the feature for the test.
ca1 = siibra.get_region("julich 2.9", "ca1")
features = siibra.features.get(ca1, siibra.features.VolumeOfInterest)
print(features[0].name)
print(features[0].last_match_description)


# %%
# Another example are gene expressions retrieved from the Allen atlas.
# These are linked by the coordinate of their tissue probes in MNI space.
# If a coordinate is inside the selected brain regions, it is an exact match.
features = siibra.features.get(v1, siibra.features.molecular.GeneExpressions, gene="TAC1")
print(features[0].last_match_description)
