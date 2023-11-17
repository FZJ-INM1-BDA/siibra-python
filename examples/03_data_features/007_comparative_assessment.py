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
Comparative analysis of brain organisation in two brain regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`siibra` data features simplify analysis of multimodal aspects of brain regions.
In this example, we select a region from the Broca region in the inferior frontal gyrus, IFG 44,
involved in language processing, and a region from the visual system in the occipital cortex, V1.
"""

# %%
import siibra
import matplotlib.pyplot as plt
import numpy as np
# sphinx_gallery_thumbnail_number = -1

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
    siibra.features.molecular.ReceptorDensityFingerprint,
    siibra.features.cellular.LayerwiseCellDensity,
    siibra.features.cellular.LayerwiseBigBrainIntensities,
]

# %%
# We iterate the regions and modalities to generate a grid plot.
fig, axs = plt.subplots(len(modalities), len(regions))
ymax = [4500, 150, 30000]
for i, region in enumerate(regions):
    for j, modality in enumerate(modalities):
        features = siibra.features.get(region, modality)
        print(region, modality, len(features))
        if len(features) > 0:
            fp = features[-1]
            fp.plot(ax=axs[j, i])
            axs[j, i].set_ylim(0, ymax[j])
fig.tight_layout()

# %%
# For the same measurement types, we can also sample individual cortical profiles,
# showing density distributions from the pial surface to the gray/white matter
# boundary in individual tissue samples. Cortical profiles queries form
# :ref:`sphx_glr_examples_03_data_features_009_compound_features.py` for easy
# handling of the profiles. We can browse through the elements with integer
# index. Let us select the first ones for comparison.
modalities = [
    siibra.features.molecular.ReceptorDensityProfile,
    siibra.features.cellular.CellDensityProfile,
    siibra.features.cellular.BigBrainIntensityProfile,
]

fig, axs = plt.subplots(len(modalities), len(regions))

for i, region in enumerate(regions):
    for j, modality in enumerate(modalities):
        cf = siibra.features.get(region, modality)[0]
        p = cf[np.random.randint(0, len(cf) - 1)]
        p.plot(ax=axs[j, i])

fig.set(figheight=15, figwidth=10)
fig.tight_layout()
