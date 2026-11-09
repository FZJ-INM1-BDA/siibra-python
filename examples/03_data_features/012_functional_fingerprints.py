# Copyright 2018-2026
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
.. _012_functional_fingerprints
:bdg-secondary:`Intermediate`

Regional functional activation and functional fingerprints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`siibra` provides the mean activation of 300 task-fMRI activation maps from
the Individual Brain Charting (IBC) project within every region of each
labelled parcellation map in MNI152 space. For one activation map, this gives
a value per region, which can colorize the parcellation. For one region, it
gives the region's functional fingerprint across all activation maps.
"""

# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 1

# %%
# We start with the whole parcellation. Querying Julich-Brain v3.1 for
# regional functional activation returns one feature whose data has
# activation maps as rows and regions as columns.
julich_brain = siibra.parcellations.get("julich 3.1")
julichbrain_func_activation = siibra.features.get(
    julich_brain, siibra.features.functional.RegionalFunctionalActivation
)[0]
print(julichbrain_func_activation.name)
julichbrain_func_activation.data.iloc[:5, :4]

# %%
# Descriptive columns, such as the task and contrast of each activation map,
# are kept apart from the regional values.
julichbrain_func_activation.activation_maps.head()

# %%
# To compare activation maps meaningfully, we pick three with strong and
# distinct regional patterns. Among the 30 maps that vary most across
# regions, we start with the most variable one and then repeatedly add the
# map least correlated with those already chosen.
variability = julichbrain_func_activation.data.std(axis=1)
candidates = variability.nlargest(30).index
correlation = julichbrain_func_activation.data.loc[candidates].T.corr()
selected_activations = [candidates[0]]
while len(selected_activations) < 3:
    redundancy = correlation.drop(index=selected_activations)[selected_activations].abs().max(axis=1)
    selected_activations.append(redundancy.idxmin())
print(selected_activations)

# %%
# We colorize the parcellation map with each selected activation map.
labelled_map = julich_brain.get_map(space="mni152")
for name in selected_activations:
    values = julichbrain_func_activation.data.loc[name].dropna()
    colorized = labelled_map.colorize(values.to_dict())
    plotting.plot_glass_brain(colorized, title=name, cmap="RdBu_r", colorbar=True, plot_abs=False)

# %%
# Next, choose two regions to compare the functional fingerprint. Bars are
# colored by their value.
for region in ['hoc1 left', 'area 4p right']:
    fig = plt.figure()
    fingerprint = siibra.features.get(
        julich_brain.get_region(region),
        siibra.features.functional.FunctionalFingerprint,
    )[0]
    fingerprint.plot(fig=fig)
