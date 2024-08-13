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
Neurotransmitter receptor densities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EBRAINS provides transmitter receptor density measurements linked to a selection of cytoarchitectonic brain regions
in the human brain (Palomero-Gallagher, Amunts, Zilles et al.). These can be accessed by calling the 
``siibra.features.get()`` method with feature types in ``siibra.features.molecular`` modality, and by
specifying a cytoarchitectonic region. Receptor densities come as cortical profiles and regional fingerprints,
both tabular style data features.
"""


# %%
import siibra

# %%
# If we query this modality for the whole atlas instead of a particular
# brain region, all linked receptor density features
# will be returned.


parcellation = siibra.get_parcellation("julich 2.9")
# TODO region.fetch_regional_map is current broken. uncomment once fixed
# features = siibra.find_features(
#     parcellation, siibra.modality_vocab.modality.NEUROTRANSMITTER_RECEPTOR_DENSITY
# )

query_param = siibra.QueryParam(
    attributes=[siibra.modality_vocab.modality.NEUROTRANSMITTER_RECEPTOR_DENSITY]
)
features = siibra.find([query_param], siibra.Feature)
print("Receptor density fingerprints found at the following anatomical anchors:")
# print(
#     "\n".join(
#         f"{feature_attribute} {qualifcation} {parcellation_attribute}"
#         for feat in features
#         for feature_attribute, parcellation_attribute, qualifcation in feat.relates_to(
#             parcellation
#         )
#     )
# )

# %%
# You can also check all facets of the this query
siibra.Feature.find_facets(features)

# %%
# And narrow down the query by selecting specific facets:
all_5ht2_features = siibra.Feature.filter_facets(features, receptor="5-HT2")
all_5ht2_fp_queries = siibra.Feature.filter_facets(
    all_5ht2_features, {"Data Type": "cortical profile"}
)

# %%
# When providing a particular region instead, the returned list is filtered accordingly.
# So we can directly retrieve densities for the primary visual cortex:
v1_fingerprints = siibra.find_features(
    siibra.get_region("julich 2.9", "v1"),
    siibra.modality_vocab.modality.NEUROTRANSMITTER_RECEPTOR_DENSITY,
)
for feature in v1_fingerprints:
    feature.plot()


# %%
# Each feature includes a data structure for the fingerprint, with mean and
# standard values for different receptors. The following table thus gives
# us the same values as shown in the polar plot above:
for feature in v1_fingerprints:
    feature.data

# %%
# Many of the receptor features also provide a profile of density measurements
# at different cortical depths, resolving the change of
# distribution from the white matter towards the pial surface.
# The profile is stored as a dictionary of density measures from 0 to 100%
# cortical depth.
v1_profiles = siibra.find_features(
    siibra.get_region("julich 2.9", "v1"),
    siibra.modality_vocab.modality.NEUROTRANSMITTER_RECEPTOR_DENSITY,
)
for p in v1_profiles:
    print(p.facets)

print(p.data)
p.plot()


# %%
# For more advanced users, they could construct queries directly.
# For example, the code below retrieves all features with the modality "neurotransmitter receptor density"

query_param = siibra.QueryParam(
    attributes=[siibra.modality_vocab.modality.NEUROTRANSMITTER_RECEPTOR_DENSITY]
)
receptor_features = siibra.find([query_param], siibra.Feature)

# %%
# To see how the receptor_features can be further categories, one can inspect the available facets:

facets = siibra.Feature.find_facets(receptor_features)
print(facets)
print(facets[facets["key"] == "receptor"])
print(facets[facets["key"] == "Data Type"])

# %%
# Now, suppose we want to further filter the result, so that we want to only get the data related to the neurotransmitter "AMPA"

ampa_features = siibra.Feature.filter_facets(receptor_features, receptor="AMPA")
assert len(ampa_features) <= len(receptor_features)
# %%
# Now, suppose we want to further filter the results, to get only those with cortical profile as the Data Type.
# Note that Data Type contains special characters, which means we cannot pass it as kwargs.

receptor_features_with_pr = siibra.Feature.filter_facets(
    receptor_features, {"Data Type": "cortical profile"}
)
assert len(receptor_features_with_pr) <= len(receptor_features)

# %%
# We can even find out how many regions has neurotransmitter receptor data

region_specs = {
    regionspec
    for feature in receptor_features
    for regionspec in feature._find(siibra.attributes.descriptions.RegionSpec)
}

for spec in region_specs:
    print(spec)
