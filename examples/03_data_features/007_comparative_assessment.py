# Copyright 2018-2022
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
Comparative analysis of brain organisation in two brain regions¶
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`siibra` data features simplify analysis of multimodal aspects of brain regions.
In this example, we select a region from the Broca region in the inferior frontal gyrus, IFG 44,
involved in language proceessing, and a region from the visual system in the occipital cortex, V1.
"""

# %%
import siibra
import matplotlib.pyplot as plt
import numpy as np

# %%
# We start by selecting the two region objects from the parcellation,
# using simple keyword specifications.
specs = ['ifg 44', 'hoc1']
regions = [siibra.get_region('julich 2.9', spec) for spec in specs]

# %%
# Next, we choose a set of feature modalities that we're interested in.
# We start with "fingerprint" style features for cell and receptor densities,
# which provide average measurements across multiple
# samples, and can be easily compared visually.
modalities = [
    siibra.modalities.ReceptorDensityFingerprint,
    siibra.modalities.CellDensityFingerprint,
    siibra.modalities.BigBrainIntensityFingerprint,
]

# %%
# We iterate the regions and modalities to generate a grid plot.
plt.style.use('seaborn')
f, axs = plt.subplots(len(modalities), len(regions))
ymax = [4500, 150, 30000]
for i, region in enumerate(regions):
    for j, modality in enumerate(modalities):
        features = siibra.get_features(region, modality)
        print(region, modality, len(features))
        if len(features) > 0:
            fp = features[-1]
            fp.barplot(ax=axs[j, i])
            axs[j, i].set_ylim(0, ymax[j])
f.tight_layout()

# %%
# For the same measurment types, we can also sample individual cortical profiles,
# showing density distributions from the pial surface to the gray/white matter
# boundary in individual tissue samples. For the receptor measurments, we
# supply now an additional filter to choose only GABAB profiles.
modalities = [
    (siibra.modalities.ReceptorDensityProfile, lambda p: "gabab" in p.receptor.lower()),
    (siibra.modalities.CellDensityProfile, lambda p: True),
    (siibra.modalities.BigBrainIntensityProfile, lambda p: True),
]
f, axs = plt.subplots(len(modalities), len(regions))
ymax = [3500, 150, 30000]

for i, region in enumerate(regions):
    for j, (modality, filterfunc) in enumerate(modalities):
        features = list(
            filter(filterfunc, siibra.get_features(region, modality))
        )
        # fetch a random sample from the available ones
        p = features[int(np.random.rand() * (len(features)))]
        p.plot(ax=axs[j, i], layercolor="darkblue")
        axs[j, i].set_ylim(0, ymax[j])
f.tight_layout()
