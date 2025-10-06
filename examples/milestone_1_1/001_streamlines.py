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
dMRI streamline fiber bundles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# %%
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# All fiber bundles are connected to several regions where the fiber ends are
# connected to. Therefore, siibra can query the bundles using these regions.
# We choose a cortical region from Julich Brain and find fiber bundles
# that have an ending at this region.
julich_brain = siibra.parcellations["julich 3.1"]
area3b_left = julich_brain.get_region("Area 3b (PostCG) left")
bundles_3bleft = siibra.features.get(
    area3b_left, siibra.features.connectivity.StreamlineFiberBundle
)
print("Bundles found:", len(bundles_3bleft))


# %%
# The streamline fiber bundles are categorized as short and long bundles, which
# is encoded in the modality. So we filter the short bundles using the modality.
print({b.modality for b in bundles_3bleft})
short_bundles_3bleft = [b for b in bundles_3bleft if "short" in b.modality]

# %%
# We can heck all the terminal regions a bundle is connected to by it's anchor
# attribute..
bundle = short_bundles_3bleft[0]
terminal_regions = list(bundle.anchor.regions.keys())
terminal_region_map = siibra.volumes.volume.merge(
    [r.get_regional_map("mni152") for r in terminal_regions],
    labels=[1, 2]
)
print("Terminal regions:")
for r in terminal_regions:
    print(r.name)


# %%
# Each bundle is represented as pandas DataFrames where each fiber is indexed
# with an integer and coordinates on MNI 152 reference space for fiiber are
# consecutively listed.
print("Fiber count:", len(bundle.data.index.unique()))
bundle.data


# %%
# Using nilearn, display the coordinates of each fiber over the terminal regions.
fig = plt.figure(figsize=(10, 4), dpi=600)
display = plotting.plot_markers(
    node_values=bundle.data.index,
    node_coords=bundle.data.values,
    node_cmap="turbo",
    node_size=0.5,
    colorbar=False,
    figure=fig,
    node_kwargs={"marker": "o", "linewidths": 0},
)
terminal_regions = list(bundle.anchor.regions.keys())
print("Terminal regions:")
for r in terminal_regions:
    print(r.name)
terminal_region_map = siibra.volumes.volume.merge(
    [r.get_regional_map("mni152") for r in terminal_regions],
    labels=[1, 2]
)
display.add_overlay(terminal_region_map.fetch(), cmap="winter")


# %%
# Similarly, long bundles can be filtered out. Long bundles have more
# desctriptive names, allowing further filtering options.
long_bundles_3bleft = [b for b in bundles_3bleft if "long" in b.modality]
for b in long_bundles_3bleft:
    print(b.name)

# %%
# We can easily filter "Left_Arcuate" and plot it over its terminal regions
# such that each fiber has a ditinct color as above.
bundle = [b for b in long_bundles_3bleft if "Left_Arcuate" in b.name][0]
print("Fiber count:", len(bundle.data.index.unique()))
fig = plt.figure(figsize=(10, 4), dpi=300)
display = plotting.plot_markers(
    node_values=bundle.data.index,
    node_coords=bundle.data.values,
    node_cmap="turbo",
    node_size=0.5,
    colorbar=False,
    figure=fig,
    node_kwargs={"marker": "o", "linewidths": 0},
)
terminal_regions = list(bundle.anchor.regions.keys())
print("Terminal regions:")
for r in terminal_regions:
    print(r.name)
terminal_region_map = siibra.volumes.volume.merge(
    [r.get_regional_map("mni152") for r in terminal_regions],
    labels=list(range(len(terminal_regions)))
)
display.add_overlay(terminal_region_map.fetch(), cmap="winter")
