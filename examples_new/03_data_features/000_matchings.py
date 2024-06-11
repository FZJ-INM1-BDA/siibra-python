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
# sphinx_gallery_thumbnail_path = '_static/example_thumbnails/default_thumbnail.png'

# %%
# Available feature types are listed in the module `siibra.features`.
# Most of these represent specifically supported data modalities, and will
# be covered one by one in the next examples.
siibra.modality_types

# %%
# Regional features
# ^^^^^^^^^^^^^^^^^
#
# As a first example, we use brain region V2 to query for "EbrainsRegionalDataset" features.
# See :ref:`ebrains_datasets` for more information about this modality.
v2 = siibra.get_region("julich 2.9", "hoc2 left")
features = siibra.find_features(v2, siibra.modality_types.STREAMLINECOUNTS)
for feature, explanation, *_ in features:
    print(f" - {feature} {explanation}")

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
occ = siibra.get_region("julich 2.9", "occipital lobe")
for f, explanation, *_ in siibra.find_features(occ, "receptor"):
    print(f)
    print(" -> ", explanation)

# %%
# If we query a rather specific region, we get more exact matches.
v1 = siibra.get_region("julich 2.9", "v1")
for f, explanation, *_ in siibra.find_features(v1, "receptor"):
    print(f)
    print(" -> ", explanation)


# %%
# Qualifications differ across queries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that the same data feature might have different qualifications when returned from different queries.
# Let's take the following dataset with fMRI data in the primary visual cortex:


# %%
# When querying for datasets related to only the left hemisphere of v1,
# the match is qualified as "contains":
v1l = siibra.get_region("julich 2.9", "v1 left")
for f, explanation, *_ in siibra.find_features(v1l, "receptor"):
    print(f)
    print(" -> ", explanation)
    print()

# %%
# Lastly, when querying for datasets related to the occipital cortex, the match is qualified as "contained":
for f, explanation, *_ in siibra.find_features(occ, "receptor"):
    print(" -> ", explanation)

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
matched_results = siibra.find_features(ca1, siibra.modality_types.PLI_HSV_FIBRE_ORIENTATION_MAP)
feature, explanation, *_ = matched_results[0]
print(feature, explanation)


# %%
# Another example are gene expressions retrieved from the Allen atlas.
# These are linked by the coordinate of their tissue probes in MNI space.
# If a coordinate is inside the selected brain regions, it is an exact match.
# For example, the gene expressions retrieved from the Allen atlas are linked by the coordinate
# For example, the gene expressions retrieved from the Allen atlas are linked by the coordinate
# of their tissue probes in MNI space. If a coordinate is inside the selected brain regions, it is an exact match.
features = siibra.find_features(v1l, siibra.modality_types.GENE_EXPRESSIONS, genes=["MAOA", "TAC1"])
assert len(features) == 1
feature, explanation, *_ = features[0]
print(explanation)

