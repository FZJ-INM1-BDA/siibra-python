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

EBRAINS provides transmitter receptor density measurments linked to a selection of cytoarchitectonic brain regions in the human brain (Palomero-Gallagher, Amunts, Zilles et al.). These can be accessed by calling the ``siibra.get_features()`` method with the ``siibra.modalities.ReceptorDistribution`` modality (or the shorthand 'receptor'), and by specifying a cytoarchitectonic region. Receptor densities come as a structured datatype which includes a regional fingerprint with average densities for different transmitters, as well as often an additional cortical density profile and a sample autoradiograph patch. They bring their own `plot()` method to produce a quick illustration.
"""


# %%
# We start by selecting an atlas.
import siibra
atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

# %%
# If we query this modality for the whole atlas instead of a particular
# brain region, all linked receptor density features
# will be returned.
all_features = siibra.get_features( atlas, siibra.modalities.ReceptorDistribution)
print("Receptor density features found for the following regions:")
print("\n".join(f.regionspec for f in all_features))

# %%
# When providing a particular region instead, the returned list is filtered accordingly. 
# So we can directly retrieve densities for the primary visual cortex:
v1_features = siibra.get_features(atlas.get_region('v1'), 'receptor')
for f in v1_features:
    fig = f.plot()

# %%
# Each feature includes a data structure for the fingerprint, with mean and
# standard values for different receptors. The following table thus gives
# us the same values as shown in the polar plot above:
fp = v1_features[0].fingerprint
for label, mean, std in zip(fp.labels, fp.meanvals, fp.stdvals):
    print(f"{label:20.20} {mean:10.0f} {fp.unit}      +/-{std:4.0f}")

# %%
# Many of the receptor features also provide a profile of density measurements
# at different cortical depths, resolving the change of
# distribution from the white matter towards the pial surface.
# The profile is stored as a dictionary of density measures from 0 to 100%
# cortical depth.
p_ampa = v1_features[0].profiles['AMPA']
import matplotlib.pyplot as plt
plt.plot(p_ampa.densities.keys(), p_ampa.densities.values())
plt.title(f"Cortical profile of AMPA densities in V1")
plt.xlabel("Cortical depth (%)")
plt.ylabel(p_ampa.unit)
plt.grid(True)

# %%
# Lastly, many receptor features provide a sample 2D cortical patch of the
# color-coded autoradiograph for illustration.
img = v1_features[0].autoradiographs['AMPA']
plt.imshow(img)
plt.axis('off')
plt.title(f"Sample color-coded autoradiography patch for AMPA in V1")

