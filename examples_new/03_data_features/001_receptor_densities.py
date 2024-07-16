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
query_cursor = siibra.QueryCursor(parcellation, siibra.modality_types.NEUROTRANSMITTER_RECEPTOR_DENSITY)
all_features = query_cursor.exec_explain()
print("Receptor density fingerprints found at the following anatomical anchors:")
print(
    "\n".join(
        str(anchored_concept)
        for feat, explanation, qualifications in all_features
        for queried_concept, anchored_concept, qualifcation in qualifications
    )
)

# %%
# You can also check all facets of the this query
print(query_cursor.facets)

# %% 
# And narrow down the query by selecting specific facets:
all_5ht2_queries = query_cursor.filter_by_facets(receptor="5-HT2")
all_5ht2_fp_queries = all_5ht2_queries.filter_by_facets({ "Data Type": "cortical profile" })

# %%
# When providing a particular region instead, the returned list is filtered accordingly.
# So we can directly retrieve densities for the primary visual cortex:
v1_fingerprints = siibra.find_features(
    siibra.get_region("julich 2.9", "v1"),
    siibra.modality_types.NEUROTRANSMITTER_RECEPTOR_DENSITY_FINGERPRINT,
)
for feature in v1_fingerprints:
    feature.plot()


# %%
# Each feature includes a data structure for the fingerprint, with mean and
# standard values for different receptors. The following table thus gives
# us the same values as shown in the polar plot above:
for feature, *_ in v1_fingerprints:
    feature.data

# %%
# Many of the receptor features also provide a profile of density measurements
# at different cortical depths, resolving the change of
# distribution from the white matter towards the pial surface.
# The profile is stored as a dictionary of density measures from 0 to 100%
# cortical depth.
v1_profiles = siibra.find_features(
    siibra.get_region("julich 2.9", "v1"),
    siibra.modality_types.NEUROTRANSMITTER_RECEPTOR_DENSITY_PROFILE,
)
for p in v1_profiles:
    print(p.facets)

print(p.data)
p.plot()
