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
# We start by loading the library
import siibra
from nilearn import plotting
import matplotlib.pyplot as plt

# %%
# We choose a cortical region from Julich Brain and find fiber bundles
# overlapping with this region
julich_brain = siibra.parcellations["julich 3.1"]
area3b_left = julich_brain.get_region("Area 3b (PostCG) left")
bundles_3bleft = siibra.features.get(
    area3b_left, siibra.features.connectivity.StreamlineFiberBundle
)
print("Bundles found:", len(bundles_3bleft))
print({b.modality for b in bundles_3bleft})


# %%
short_bundles_3bleft = [b for b in bundles_3bleft if "short" in b.modality]

# %%
# Each bundle is represented as a dictionary of fibers which in turn are
# represented as Contour objects. Contours are just PointClouds where the order
# of the coordinates is important. (This enables warping the coordinates to
# other spaces efficiently). Let us choose a bundle to demonstrate
bundle = short_bundles_3bleft[0]
bundle.data


# %%
print("Fiber count:", len(bundle.data.index.unique()))
fig = plt.figure(figsize=(10, 4), dpi=600)
display = plotting.plot_markers(
    node_values=bundle.data.index,
    node_coords=bundle.data.values,
    node_cmap="jet",
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
display.add_overlay(terminal_region_map.fetch())

# %%
long_bundles_3bleft = [b for b in bundles_3bleft if "long" in b.modality]
for b in long_bundles_3bleft:
    print(b.name)

# %%
bundle = [b for b in long_bundles_3bleft if "Left_Arcuate" in b.name][0]
print("Fiber count:", len(bundle.data.index.unique()))
fig = plt.figure(figsize=(10, 4), dpi=300)
display = plotting.plot_markers(
    node_values=bundle.data.index,
    node_coords=bundle.data.values,
    node_cmap="jet",
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
display.add_overlay(terminal_region_map.fetch())
