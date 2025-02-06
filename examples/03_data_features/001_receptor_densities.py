# Copyright 2018-2025
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
parcellation = siibra.parcellations.get('julich 2.9')
all_features = siibra.features.get(parcellation, siibra.features.molecular.ReceptorDensityFingerprint)
print("Receptor density fingerprints found at the following anatomical anchors:")
print("\n".join(str(f.anchor) for f in all_features))

# %%
# When providing a particular region instead, the returned list is filtered accordingly.
# So we can directly retrieve densities for the primary visual cortex:
v1_fingerprints = siibra.features.get(
    siibra.get_region('julich 2.9', 'v1'),
    siibra.features.molecular.ReceptorDensityFingerprint
)
v1_fingerprints[0].plot()

# %%
# Each feature includes a data structure for the fingerprint, with mean and
# standard values for different receptors. The following table thus gives
# us the same values as shown in the polar plot above:
v1_fingerprints[0].data

# %%
# Many of the receptor features also provide a profile of density measurements
# at different cortical depths, resolving the change of
# distribution from the white matter towards the pial surface.
# The profile is stored as a dictionary of density measures from 0 to 100%
# cortical depth.
v1_profiles = siibra.features.get(
    siibra.get_region('julich 2.9', 'v1'),
    siibra.features.molecular.ReceptorDensityProfile
)[0]
for p in v1_profiles:
    print(p.receptor)
    if "GABAA" in p.receptor:
        print(p.receptor)
        break
print(p.data)
p.plot()
